#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SKIP_DIRS = {"node_modules", ".git", "tmp", "snippets"}
PLAIN_PYTHON_FENCE_RE = re.compile(r"^\s*```python(?:\s+.*)?\s*$")


def iter_markdown_files(paths: list[str]) -> list[Path]:
    files: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix == ".md":
            files.add(path)
            continue
        if not path.is_dir():
            continue
        for md in path.rglob("*.md"):
            if any(part in SKIP_DIRS for part in md.parts):
                continue
            if md.is_file():
                files.add(md)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail when markdown contains plain ```python fences.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to scan",
    )
    args = parser.parse_args()

    files = iter_markdown_files(args.paths)
    if not files:
        print("No markdown files found.")
        return 0

    has_error = False
    for path in files:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if PLAIN_PYTHON_FENCE_RE.match(line):
                print(f"{path}:{line_no}: plain ```python fence is not allowed")
                has_error = True

    return 1 if has_error else 0


if __name__ == "__main__":
    sys.exit(main())
