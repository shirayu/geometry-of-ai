#!/usr/bin/env python3
"""Check SVG file naming convention: NN_name.svg (no 'fig_' prefix after chapter number)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Valid: NN_name.svg  (two digits, underscore, then name without 'fig_')
RE_VALID = re.compile(r"^\d{2}_(?!fig_).+\.svg$")
# Non-conforming: NN_fig_name.svg
RE_FIG_PREFIX = re.compile(r"^\d{2}_fig_.+\.svg$")

# Detect <img src="..."> and ![...](...)  references in markdown
RE_IMG_TAG_SRC = re.compile(r"""<img\b[^>]*\bsrc\s*=\s*(?P<q>['"]?)(?P<url>[^'">\s]+)(?P=q)""", re.IGNORECASE)
RE_MD_IMG = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def check_svg_files(public_dir: Path) -> list[str]:
    errors: list[str] = []
    for svg in sorted(public_dir.rglob("*.svg")):
        name = svg.name
        if RE_FIG_PREFIX.match(name):
            errors.append(f"{svg}: non-conforming name (use '{name.replace('_fig_', '_', 1)}' instead of '{name}')")
        elif not RE_VALID.match(name):
            errors.append(f"{svg}: name does not match NN_name.svg convention")
    return errors


def _extract_svg_refs(text: str) -> list[str]:
    refs: list[str] = []
    for m in RE_IMG_TAG_SRC.finditer(text):
        url = m.group("url")
        if url.endswith(".svg"):
            refs.append(url)
    for m in RE_MD_IMG.finditer(text):
        url = m.group(1).split()[0]  # strip title
        if url.endswith(".svg"):
            refs.append(url)
    return refs


def check_markdown_refs(series_dir: Path) -> list[str]:
    errors: list[str] = []
    for md in sorted(series_dir.rglob("*.md")):
        text = md.read_text(encoding="utf-8")
        for url in _extract_svg_refs(text):
            basename = Path(url).name
            if RE_FIG_PREFIX.match(basename):
                errors.append(
                    f"{md}: references non-conforming SVG name '{basename}'"
                    f" (expected '{basename.replace('_fig_', '_', 1)}')"
                )
            elif re.match(r"^\d{2}_", basename) and not RE_VALID.match(basename):
                errors.append(f"{md}: references SVG with non-conforming name '{basename}'")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series_dir", type=Path, help="Directory containing .md files (e.g. site/series)")
    parser.add_argument(
        "--public-dir",
        type=Path,
        default=None,
        help="Directory containing SVG files (default: <series_dir>/../public/series)",
    )
    args = parser.parse_args()

    series_dir: Path = args.series_dir
    public_dir: Path = args.public_dir or (series_dir.parent / "public" / "series")

    all_errors: list[str] = []
    if public_dir.exists():
        all_errors.extend(check_svg_files(public_dir))
    else:
        print(f"warning: public dir not found: {public_dir}", file=sys.stderr)

    all_errors.extend(check_markdown_refs(series_dir))

    for msg in all_errors:
        print(msg, file=sys.stderr)

    return 1 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
