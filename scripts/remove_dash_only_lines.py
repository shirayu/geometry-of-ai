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
    filtered = [line for line in lines if not DASH_ONLY_RE.match(line)]
    updated = "\n".join(filtered)
    if has_trailing_newline and not updated.endswith("\n"):
        updated += "\n"
    return updated


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    files = iter_markdown_files(root)
    changed = False
    for path in files:
        original = path.read_text(encoding="utf-8")
        updated = remove_dash_only_lines(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
