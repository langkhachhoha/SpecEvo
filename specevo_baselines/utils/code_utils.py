"""
Utilities for code parsing, diffing, and manipulation
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple


def apply_diff(original_solution: str, diff_text: str) -> str:
    """
    Apply a diff to the original code

    Args:
        original_solution: Original source solution
        diff_text: Diff in the SEARCH/REPLACE format

    Returns:
        Modified solution
    """
    # Split into lines for easier processing
    original_lines = original_solution.split("\n")
    result_lines = original_lines.copy()

    # Extract diff blocks
    diff_blocks = extract_diffs(diff_text)

    # Apply each diff block
    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")
        replace_lines = replace_text.split("\n")

        # Find where the search pattern starts in the original solution
        for i in range(len(result_lines) - len(search_lines) + 1):
            if result_lines[i : i + len(search_lines)] == search_lines:
                # Replace the matched section
                result_lines[i : i + len(search_lines)] = replace_lines
                break

    return "\n".join(result_lines)


def extract_diffs(diff_text: str) -> List[Tuple[str, str]]:
    """
    Extract diff blocks from the diff text

    Args:
        diff_text: Diff in the SEARCH/REPLACE format

    Returns:
        List of tuples (search_text, replace_text)
    """
    diff_pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
    diff_blocks = re.findall(diff_pattern, diff_text, re.DOTALL)
    return [(match[0].rstrip(), match[1].rstrip()) for match in diff_blocks]


def parse_full_rewrite(llm_response: str, language: str = "python") -> Optional[str]:
    """
    Extract a full rewrite from an LLM response

    Args:
        llm_response: Response from the LLM
        language: Programming language

    Returns:
        Extracted code or None if not found
    """
    solution_block_pattern = r"```" + language + r"\n(.*?)```"
    matches = re.findall(solution_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to any solution block
    solution_block_pattern = r"```(.*?)```"
    matches = re.findall(solution_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to plain text
    return llm_response


def _extract_def_info(solution: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """
    Extract function/class name and docstring (or first comment as fallback) from solution block.

    Returns:
        Tuple of (kind, name, docstring_first_line) or None if not found
    """
    # Look for function definition
    func_match = re.search(r"^\s*def\s+(\w+)\s*\(", solution, re.MULTILINE)
    if func_match:
        name = func_match.group(1)
        # Try to extract docstring, fallback to first comment
        docstring = _extract_docstring(solution, func_match.end())
        if not docstring:
            docstring = _extract_first_comment(solution, func_match.start())
        return ("function", name, docstring)

    # Look for class definition
    class_match = re.search(r"^\s*class\s+(\w+)", solution, re.MULTILINE)
    if class_match:
        name = class_match.group(1)
        docstring = _extract_docstring(solution, class_match.end())
        if not docstring:
            docstring = _extract_first_comment(solution, class_match.start())
        return ("class", name, docstring)

    return None


def _extract_first_comment(solution: str, func_start: int) -> Optional[str]:
    """
    Extract consecutive comment lines inside a function/class body.
    Used as fallback when no docstring is available.
    Returns up to 5 lines of comments joined together.
    """
    remaining = solution[func_start:]
    colon_match = re.search(r"(?:\)|[^:]+):\s*\n", remaining)
    if not colon_match:
        return None

    # Get the body after the colon
    body_start = colon_match.end()
    body = remaining[body_start:]

    # Collect consecutive comment lines
    comment_lines = []
    lines = body.split("\n")
    for line in lines[:10]:  # Check first 10 lines for comments
        stripped = line.strip()
        if stripped.startswith("#"):
            # Remove the # and leading space
            comment_text = stripped[1:].strip()
            if comment_text:
                comment_lines.append(comment_text)
            if len(comment_lines) >= 5:  # Max 5 lines
                break
        elif stripped and not stripped.startswith("#"):
            # Hit actual code, stop collecting
            break

    return "\n".join(comment_lines) if comment_lines else None


def _extract_docstring(solution: str, start_pos: int) -> Optional[str]:
    """
    Extract first line of docstring after a given position.

    Args:
        solution: Source code
        start_pos: Position to start searching from
    """
    remaining = solution[start_pos:]
    docstring_match = re.search(r':\s*\n\s*("""|\'\'\')(.*?)("""|\'\'\')', remaining, re.DOTALL)

    if docstring_match:
        docstring_content = docstring_match.group(2).strip()
        return docstring_content.split("\n")[0].strip()

    return None


def format_diff_summary(diff_blocks: List[Tuple[str, str]]) -> str:
    """
    Create a human-readable summary of the diff.

    If docstrings are identical between old and new code, uses simpler format.
    If docstrings differ or function is renamed, shows the meaningful change.

    Args:
        diff_blocks: List of (search_text, replace_text) tuples

    Returns:
        Summary string
    """
    summary = []

    for i, (search_text, replace_text) in enumerate(diff_blocks):
        search_lines = search_text.strip().split("\n")
        replace_lines = replace_text.strip().split("\n")

        # Try to extract meaningful info from the solution
        old_info = _extract_def_info(search_text)
        new_info = _extract_def_info(replace_text)

        # Build a meaningful summary
        if old_info or new_info:
            info = new_info or old_info
            kind, name, docstring = info

            # Get docstrings from both to compare
            old_docstring = old_info[2] if old_info else None
            new_docstring = new_info[2] if new_info else None

            if old_info and new_info and old_info[1] != new_info[1]:
                # Renamed function/class - always show this
                desc = f"Renamed {old_info[0]} `{old_info[1]}` → `{new_info[1]}`"
            elif old_docstring and new_docstring and old_docstring != new_docstring:
                # Docstrings are DIFFERENT - show the new docstring
                desc = f"Modified {kind} `{name}`: {new_docstring}"
            elif old_docstring == new_docstring:
                # Docstrings are IDENTICAL - use simple format (just line counts)
                desc = f"Modified {kind} `{name}` ({len(search_lines)}→{len(replace_lines)} lines)"
            elif docstring:
                # Only one has docstring
                desc = f"Modified {kind} `{name}`: {docstring}"
            else:
                desc = f"Modified {kind} `{name}` ({len(search_lines)}→{len(replace_lines)} lines)"

            summary.append(f"Change {i+1}: {desc}")
        elif len(search_lines) == 1 and len(replace_lines) == 1:
            # Single line change - show the actual change
            summary.append(
                f"Change {i+1}: '{search_lines[0].strip()}' → '{replace_lines[0].strip()}'"
            )
        else:
            # Fallback: show first non-empty line as context
            first_old = next((l.strip() for l in search_lines if l.strip()), "")
            first_new = next((l.strip() for l in replace_lines if l.strip()), "")

            if first_old and first_new:
                summary.append(
                    f"Change {i+1}: Near `{first_old[:50]}...` ({len(search_lines)}→{len(replace_lines)} lines)"
                )
            else:
                summary.append(
                    f"Change {i+1}: Replace {len(search_lines)} lines with {len(replace_lines)} lines"
                )

    return "\n".join(summary)


def extract_solution_language(solution: str) -> str:
    """
    Try to determine the language of a solution snippet in string format

    Args:
        solution: Solution snippet

    Returns:
        Detected language or "text" by default if no language is detected
    """
    # Look for common language signatures
    if re.search(r"^(import|from|def|class)\s", solution, re.MULTILINE):
        return "python"
    elif re.search(r"^(package|import java|public class)", solution, re.MULTILINE):
        return "java"
    elif re.search(r"^(#include|int main|void main)", solution, re.MULTILINE):
        return "cpp"
    elif re.search(r"^(function|var|let|const|console\.log)", solution, re.MULTILINE):
        return "javascript"
    elif re.search(r"^(module|fn|let mut|impl)", solution, re.MULTILINE):
        return "rust"
    elif re.search(r"^(SELECT|CREATE TABLE|INSERT INTO)", solution, re.MULTILINE):
        return "sql"

    return "text"


def build_repo_map(
    root: str,
    *,
    max_depth: int = 4,
    allowed_extensions: Tuple[str, ...] = (".py",),
    excluded_dirs: Tuple[str, ...] = (".git", "__pycache__"),
) -> str:
    """Return a depth-limited directory tree of *root* as a string.

    Only files whose extension is in *allowed_extensions* are included.
    Directories in *excluded_dirs* (and hidden directories) are skipped.
    Returns an empty string if *root* does not exist or is not a directory.
    """
    if not root or not os.path.isdir(root):
        return ""

    root_path = Path(root).resolve()
    excluded: Set[str] = set(excluded_dirs)
    allowed: Set[str] = set(allowed_extensions)
    lines: List[str] = []

    def walk(directory: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return
        for entry in entries:
            if entry.name.startswith(".") or entry.name in excluded:
                continue
            if entry.is_dir():
                lines.append(f"{prefix}{entry.name}/")
                walk(entry, prefix + "  ", depth + 1)
            elif entry.suffix in allowed:
                lines.append(f"{prefix}{entry.name}")

    walk(root_path, "  ", 0)
    return "\n".join(lines)
