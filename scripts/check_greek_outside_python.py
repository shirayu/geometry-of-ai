#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

GREEK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
HTML_TAG_RE = re.compile(r"</?([A-Za-z][A-Za-z0-9:-]*)\b[^>]*?>")
VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


def iter_markdown_files(series_dir: Path) -> list[Path]:
    return sorted(series_dir.rglob("*.md"))


def is_fence_line(stripped: str) -> tuple[bool, str]:
    if stripped.startswith("```"):
        return True, "```"
    if stripped.startswith("~~~"):
        return True, "~~~"
    return False, ""


def fence_info_name(stripped: str, marker: str) -> str:
    rest = stripped[len(marker) :].strip()
    if not rest:
        return ""
    return rest.split()[0]


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    in_fence = False
    fence_marker = ""
    fence_is_python = False
    html_stack: list[str] = []

    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        is_fence, marker = is_fence_line(stripped)
        if is_fence:
            if not in_fence:
                in_fence = True
                fence_marker = marker
                fence_name = fence_info_name(stripped, marker)
                fence_is_python = fence_name.endswith(".py")
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
                fence_is_python = False
            continue

        has_html_tag = False
        for match in HTML_TAG_RE.finditer(line):
            has_html_tag = True
            tag = match.group(1).lower()
            full = match.group(0)
            is_closing = full.startswith("</")
            is_self_closing = full.endswith("/>") or tag in VOID_TAGS
            if is_closing:
                if html_stack and html_stack[-1] == tag:
                    html_stack.pop()
                else:
                    # best-effort: remove first matching tag if nesting is odd
                    if tag in html_stack:
                        html_stack.remove(tag)
            elif not is_self_closing:
                html_stack.append(tag)

        if fence_is_python:
            continue

        if GREEK_RE.search(line):
            if html_stack or has_html_tag:
                continue
            errors.append(f"{path}:{idx}: Greek letter outside python fence")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail when markdown under series/ contains Greek letters outside python fences."
    )
    parser.add_argument(
        "series_dir",
        nargs="?",
        default="series",
        help="Root directory for series markdown files (default: series)",
    )
    args = parser.parse_args()

    series_dir = Path(args.series_dir)
    if not series_dir.exists():
        print(f"series dir not found: {series_dir}")
        return 2

    errors: list[str] = []
    for path in iter_markdown_files(series_dir):
        errors.extend(check_file(path))

    if errors:
        for err in errors:
            print(err)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
