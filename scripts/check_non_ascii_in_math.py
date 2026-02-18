#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ARROW_COMMANDS = ("\\xrightarrow", "\\xleftarrow")


def is_escaped(text: str, idx: int) -> bool:
    backslashes = 0
    k = idx - 1
    while k >= 0 and text[k] == "\\":
        backslashes += 1
        k -= 1
    return (backslashes % 2) == 1


def iter_markdown_files(paths: list[str]) -> list[Path]:
    files: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.update(path.rglob("*.md"))
        elif path.is_file() and path.suffix == ".md":
            files.add(path)
    return sorted(files)


def is_fence_line(stripped: str) -> tuple[bool, str]:
    if stripped.startswith("```"):
        return True, "```"
    if stripped.startswith("~~~"):
        return True, "~~~"
    return False, ""


def is_non_ascii_char(ch: str) -> bool:
    return ord(ch) > 0x7F


def find_matching(text: str, start: int, opener: str, closer: str) -> int | None:
    if start >= len(text) or text[start] != opener:
        return None

    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue

        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i
        i += 1

    return None


def extract_math_segments(path: Path) -> list[tuple[str, list[tuple[int, int]]]]:
    segments: list[tuple[str, list[tuple[int, int]]]] = []

    in_fence = False
    fence_marker = ""
    in_code_span = False
    code_span_ticks = 0
    in_math = False
    math_delim = ""

    math_chars: list[str] = []
    math_positions: list[tuple[int, int]] = []

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(keepends=True), start=1):
        stripped = line.lstrip()
        is_fence, marker = is_fence_line(stripped)
        if is_fence and not in_math:
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            continue

        if in_fence:
            continue

        i = 0
        while i < len(line):
            ch = line[i]

            if not in_math and ch == "`":
                n = 1
                while i + n < len(line) and line[i + n] == "`":
                    n += 1
                if not in_code_span:
                    in_code_span = True
                    code_span_ticks = n
                elif n == code_span_ticks:
                    in_code_span = False
                    code_span_ticks = 0
                i += n
                continue

            if in_code_span and not in_math:
                i += 1
                continue

            if not in_math and ch == "$" and not is_escaped(line, i):
                if i + 1 < len(line) and line[i + 1] == "$":
                    in_math = True
                    math_delim = "$$"
                    i += 2
                else:
                    in_math = True
                    math_delim = "$"
                    i += 1
                continue

            if in_math:
                if ch == "$" and not is_escaped(line, i):
                    if math_delim == "$$":
                        if i + 1 < len(line) and line[i + 1] == "$":
                            segments.append(("".join(math_chars), math_positions.copy()))
                            math_chars.clear()
                            math_positions.clear()
                            in_math = False
                            math_delim = ""
                            i += 2
                            continue
                    else:
                        if not (i + 1 < len(line) and line[i + 1] == "$"):
                            segments.append(("".join(math_chars), math_positions.copy()))
                            math_chars.clear()
                            math_positions.clear()
                            in_math = False
                            math_delim = ""
                            i += 1
                            continue

                if ch not in "\r\n":
                    math_chars.append(ch)
                    math_positions.append((line_no, i + 1))

            i += 1

    return segments


def collect_arrow_issues(math_text: str, positions: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
    issues: list[tuple[int, int, str]] = []
    i = 0

    while i < len(math_text):
        matched = None
        for command in ARROW_COMMANDS:
            if math_text.startswith(command, i):
                matched = command
                break

        if matched is None:
            i += 1
            continue

        j = i + len(matched)

        while j < len(math_text) and math_text[j].isspace():
            j += 1

        label_spans: list[tuple[int, int]] = []

        if j < len(math_text) and math_text[j] == "[":
            end = find_matching(math_text, j, "[", "]")
            if end is None:
                i += len(matched)
                continue
            label_spans.append((j + 1, end))
            j = end + 1

        while j < len(math_text) and math_text[j].isspace():
            j += 1

        if j < len(math_text) and math_text[j] == "{":
            end = find_matching(math_text, j, "{", "}")
            if end is None:
                i += len(matched)
                continue
            label_spans.append((j + 1, end))
            j = end + 1

        for start, end in label_spans:
            label = math_text[start:end]

            for rel, ch in enumerate(label):
                if is_non_ascii_char(ch):
                    abs_idx = start + rel
                    line_no, col = positions[abs_idx]
                    issues.append((line_no, col, f"Non-ASCII char {ch!r} in {matched} label"))

        i = j if j > i else i + len(matched)

    return issues


def check_file(path: Path) -> list[str]:
    errors_by_line: dict[int, list[tuple[int, str]]] = defaultdict(list)

    for math_text, positions in extract_math_segments(path):
        issues = collect_arrow_issues(math_text, positions)
        for line_no, col, message in issues:
            errors_by_line[line_no].append((col, message))

    errors: list[str] = []
    for line_no in sorted(errors_by_line):
        items = sorted(errors_by_line[line_no], key=lambda x: x[0])
        details = "; ".join(f"{msg}({col})" for col, msg in items)
        errors.append(f"{path}:{line_no}: Problematic arrow label in math ({len(items)}): {details}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Detect problematic labels in \\xrightarrow/\\xleftarrow inside markdown math blocks ($...$, $$...$$)."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["series", "exercise"],
        help="Files or directories to scan (default: series exercise)",
    )
    args = parser.parse_args()

    files = iter_markdown_files(args.paths)
    if not files:
        print("No markdown files found.")
        return 0

    errors: list[str] = []
    for path in files:
        errors.extend(check_file(path))

    if errors:
        for err in errors:
            print(err)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
