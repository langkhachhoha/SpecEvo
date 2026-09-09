"""Tests for the LLM output parser."""

from __future__ import annotations

from specevo.simple.parser import LLMOutput, OutputParser, ParserConfig


def test_parses_well_formed_output() -> None:
    text = """\
## Description
Branch-and-bound with priority queue keyed on a partial-sum lower bound.
Uses a numpy ndarray for the frontier and prunes nodes whose bound exceeds
the incumbent.

## Code
```python
def solve(x):
    return x * 2
```
"""
    parsed = OutputParser().parse(text)
    assert parsed.has_description
    assert parsed.has_code
    assert "branch-and-bound" in parsed.description.lower()
    assert "def solve" in parsed.code


def test_handles_missing_description_header() -> None:
    text = """```python
def f(): return 1
```"""
    parsed = OutputParser().parse(text)
    assert not parsed.has_description
    assert "def f" in parsed.code


def test_extracts_prose_before_fence_when_no_header() -> None:
    """When LLM forgets the `## Description` header, treat the prose that
    precedes the first code fence as the description."""
    text = """\
This is a bottom-up dynamic programming solution that uses a dp array
indexed by target. The distinguishing trick is initializing every cell
to infinity so missing solutions are detected with a single comparison.

```python
def solve(n, target):
    return -1
```
"""
    parsed = OutputParser().parse(text)
    assert parsed.has_description
    assert "dynamic programming" in parsed.description
    assert "def solve" in parsed.code


def test_needs_fallback_when_description_short() -> None:
    text = """## Description
foo

## Code
```python
def g(): return 1
```"""
    parser = OutputParser(ParserConfig(min_description_chars=20))
    parsed = parser.parse(text)
    assert parser.needs_fallback_summary(parsed)


def test_no_fallback_when_description_long_enough() -> None:
    text = """## Description
A reasonably long paradigm description that contains enough detail.

## Code
```python
def h(): return 1
```"""
    parser = OutputParser(ParserConfig(min_description_chars=20))
    parsed = parser.parse(text)
    assert not parser.needs_fallback_summary(parsed)


def test_truncates_overly_long_description() -> None:
    text = "## Description\n" + ("x" * 5000) + "\n## Code\n```python\ndef a(): pass\n```"
    parser = OutputParser(ParserConfig(max_description_chars=200))
    parsed = parser.parse(text)
    assert len(parsed.description) <= 201  # +1 for ellipsis char


def test_falls_back_to_raw_python() -> None:
    text = "def quick():\n    return 'no fences here'"
    parsed = OutputParser().parse(text)
    assert parsed.has_code
    assert "def quick" in parsed.code


def test_empty_output() -> None:
    parsed = OutputParser().parse("")
    assert not parsed.has_code
    assert not parsed.has_description
    assert isinstance(parsed, LLMOutput)
