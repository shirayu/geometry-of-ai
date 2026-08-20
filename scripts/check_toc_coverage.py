#!/usr/bin/env python3
"""Check that every top-level series markdown page is registered in toc.data.ts's PAGES."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Pages intentionally excluded from the PAGES-based TOC listing.
EXCLUDED = {"toc.md", "quizzes.md"}

RE_FILE_ENTRY = re.compile(r"file:\s*'([^']+)'")


def registered_files(toc_data_path: Path) -> set[str]:
    text = toc_data_path.read_text(encoding="utf-8")
    return set(RE_FILE_ENTRY.findall(text))


def actual_files(series_dir: Path) -> set[str]:
    return {p.name for p in series_dir.glob("*.md")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series_dir", type=Path, help="Directory containing .md files (e.g. site/series)")
    args = parser.parse_args()

    series_dir: Path = args.series_dir
    toc_data_path = series_dir / "toc.data.ts"

    registered = registered_files(toc_data_path)
    actual = actual_files(series_dir) - EXCLUDED

    missing = sorted(actual - registered)
    stale = sorted(registered - actual)

    errors: list[str] = []
    for name in missing:
        errors.append(f"{series_dir / name}: not registered in {toc_data_path} PAGES")
    for name in stale:
        errors.append(f"{toc_data_path}: PAGES references missing file '{name}'")

    for msg in errors:
        print(msg, file=sys.stderr)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
