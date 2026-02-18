#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path


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


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    in_fence = False
    fence_marker = ""
    in_code_span = False
    code_span_ticks = 0
    in_math = False
    math_delim = ""

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
                            in_math = False
                            math_delim = ""
                            i += 2
                            continue
                    else:
                        if not (i + 1 < len(line) and line[i + 1] == "$"):
                            in_math = False
                            math_delim = ""
                            i += 1
                            continue

                if ord(ch) > 0x7F and ch not in "\r\n":
                    codepoint = f"U+{ord(ch):04X}"
                    name = unicodedata.name(ch, "UNKNOWN")
                    errors.append(
                        f"{path}:{line_no}:{i + 1}: Non-ASCII in math ({codepoint} {name}): {ch!r}"
                    )

            i += 1

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect non-ASCII characters inside markdown math blocks ($...$, $$...$$)."
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
