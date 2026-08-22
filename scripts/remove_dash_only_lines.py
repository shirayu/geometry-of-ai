#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

SKIP_DIRS = {"node_modules", ".git", "tmp"}
DASH_ONLY_RE = re.compile(r"^-{3,}\s*$")


def iter_markdown_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.md"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            files.append(path)
    return files


def remove_dash_only_lines(text: str) -> str:
    has_trailing_newline = text.endswith("\n")
    lines = text.split("\n")

    frontmatter_end = -1
    if lines and lines[0] == "---":
        for i in range(1, len(lines)):
            if lines[i] == "---":
                frontmatter_end = i
                break

    filtered = [line for i, line in enumerate(lines) if i <= frontmatter_end or not DASH_ONLY_RE.match(line)]
    updated = "\n".join(filtered)
    if has_trailing_newline and not updated.endswith("\n"):
        updated += "\n"
    return updated


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    files = iter_markdown_files(root)
    for path in files:
        original = path.read_text(encoding="utf-8")
        updated = remove_dash_only_lines(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
