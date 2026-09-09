"""LLM output parser.

LLM is asked to output two sections in order:

    ## Description
    <2-4 sentences about paradigm, key data structures, distinguishing trick>

    ## Code
    ```python
    <code>
    ```

Parser extracts both. If description is missing or too short but code is
present and runnable, caller may invoke a fallback summarizer (mutation
model) — see ``OutputParser.needs_fallback_summary``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


_DESC_HEADER = re.compile(r"^\s*##\s*Description\s*$", re.IGNORECASE | re.MULTILINE)
_CODE_HEADER = re.compile(r"^\s*##\s*Code\s*$", re.IGNORECASE | re.MULTILINE)
_FENCED_PY = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class ParserConfig:
    min_description_chars: int = 20
    max_description_chars: int = 800


@dataclass
class LLMOutput:
    description: str
    code: str
    description_was_fallback: bool = False

    @property
    def has_code(self) -> bool:
        return bool(self.code.strip())

    @property
    def has_description(self) -> bool:
        return bool(self.description.strip())


class OutputParser:
    """Parses ``## Description`` + ``## Code`` blocks from raw LLM text."""

    def __init__(self, config: ParserConfig | None = None) -> None:
        self.config = config or ParserConfig()

    def parse(self, text: str) -> LLMOutput:
        """Best-effort extraction. Always returns an LLMOutput.

        Recognized output shapes, tried in order:

        1. ``## Description`` + ``## Code`` headers (the requested format).
        2. ``## Description`` header followed by a code fence (no ``## Code``).
        3. Header-less: any prose preceding the first fenced code block is
           treated as the description.
        4. Code fence only: description empty, caller may invoke the
           fallback summarizer.
        5. Raw Python (text begins with ``def``/``class``/``import``/
           ``from``): code only, no description.
        """
        text = text or ""
        code = self._extract_code(text)
        description = self._extract_description(text, code=code)
        return LLMOutput(description=description, code=code)

    def _extract_description(self, text: str, *, code: str) -> str:
        # Case 1+2: explicit "## Description" header wins.
        desc_m = _DESC_HEADER.search(text)
        if desc_m:
            start = desc_m.end()
            next_h = re.search(r"^\s*##\s+\S", text[start:], re.MULTILINE)
            end = start + next_h.start() if next_h else len(text)
            block = text[start:end]
            # Strip out the fenced code block if it sits inside the slice
            # (happens when ## Code is missing and the fence comes right
            # after the description prose).
            block = _FENCED_PY.sub("", block)
            return self._clip(block.strip())

        # Case 3: header-less, but there's a fenced code block — grab the
        # prose that comes before it.
        fenced = _FENCED_PY.search(text)
        if fenced and code:
            head = text[: fenced.start()].strip()
            # Drop a leading "## Code" line (if present, prose ends there).
            head = _CODE_HEADER.sub("", head).strip()
            if len(head) >= self.config.min_description_chars:
                return self._clip(head)
        return ""

    def _clip(self, desc: str) -> str:
        if len(desc) > self.config.max_description_chars:
            desc = desc[: self.config.max_description_chars].rstrip() + "…"
        return desc

    def _extract_code(self, text: str) -> str:
        # Prefer code under "## Code" header if present.
        code_m = _CODE_HEADER.search(text)
        haystack = text[code_m.end() :] if code_m else text
        fenced = _FENCED_PY.search(haystack)
        if fenced:
            return fenced.group(1).strip()
        # Fallback: try the very first fenced block anywhere.
        fenced = _FENCED_PY.search(text)
        if fenced:
            return fenced.group(1).strip()
        # Last resort: if text *looks* like raw python (def/class/import on
        # first non-blank line), accept as-is.
        stripped = text.strip()
        if stripped.startswith(("def ", "class ", "import ", "from ")):
            return stripped
        return ""

    def needs_fallback_summary(self, parsed: LLMOutput) -> bool:
        """True iff code is present but description is missing/too short."""
        return parsed.has_code and len(parsed.description) < self.config.min_description_chars


FALLBACK_SUMMARY_PROMPT = (
    "Write a 2-4 sentence paragraph summarizing this Python solution.\n\n"
    "The paragraph MUST cover, in any natural order:\n"
    "  - what algorithmic paradigm/approach this is (be specific: e.g. "
    "bottom-up dynamic programming, BFS over partial states, branch-and-bound "
    "with admissible lower bound, simulated annealing with linear cooling, ...)\n"
    "  - the key data structures used (name them concretely: dp array, "
    "min-heap, visited set of partial sums, ...)\n"
    "  - the distinguishing trick or heuristic that makes this solution "
    "different from a textbook version of the paradigm\n\n"
    "Do NOT restate the problem, inputs, outputs, or function signature. "
    "Write a flowing paragraph — no bullet points, no headings, no tag-style "
    "openers. Keep it under 80 words.\n\n"
    "Code:\n```python\n{code}\n```\n\n"
    "Paragraph:"
)


async def fallback_summarize(
    code: str,
    *,
    completion_fn: Callable[[str], Awaitable[str]],
) -> str:
    """Ask the mutation model to write a description for *code*.

    ``completion_fn`` is an injected async callable so this module stays
    free of litellm coupling and is trivially testable. The convention is
    ``completion_fn(prompt) -> str``.
    """
    prompt = FALLBACK_SUMMARY_PROMPT.format(code=code[:4000])
    return await completion_fn(prompt)


OUTPUT_FORMAT_INSTRUCTION = """\
Your output MUST follow this exact format, with both sections in order.

## Description
Write a single flowing paragraph (2-4 sentences, ≤ 80 words) that covers,
in any natural order:

- **What paradigm or approach** this is. Be specific — say "bottom-up
  dynamic programming over remaining target" rather than just "DP", say
  "branch-and-bound with admissible lower bound" rather than just "search".
- **Which key data structures** it uses (name them: e.g. "a dp array
  indexed by target", "a min-heap of (cost, state) pairs", "a visited
  set of partial sums").
- **The distinguishing trick or heuristic** that makes this attempt
  different from a textbook version of the paradigm (e.g. "prunes
  branches whose lower bound exceeds the incumbent", "restarts from a
  perturbed solution every 50 iterations").

Plain prose paragraph — no bullet points, no headings, no tag-style
openers, no boilerplate like "This solution..." / "The function...".
Do NOT restate the problem or the function signature.

## Code
```python
<complete, runnable Python code>
```
"""
