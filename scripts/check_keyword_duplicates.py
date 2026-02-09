#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

TABLE_ALIGN_RE = re.compile(r"^\|?\s*-{3,}\s*\|")


PARENS_RE = re.compile(r"[（(]([^）)]+)[）)]")
PUNCT_RE = re.compile(r"[\\s\\-_/,:;・.·]+")


def normalize_keyword(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("**", "").replace("*", "").replace("`", "")
    text = text.strip().lower()
    text = PUNCT_RE.sub("", text)
    return text


def split_keywords(cell: str) -> list[str]:
    cleaned = unicodedata.normalize("NFKC", cell)
    cleaned = cleaned.replace("**", "").replace("*", "").replace("`", "")

    extras: list[str] = []
    for match in PARENS_RE.finditer(cleaned):
        extras.append(match.group(1))

    base = PARENS_RE.sub("", cleaned)
    parts = re.split(r"\s*(?:,|、|，)\s*", base)
    parts.extend(extras)

    normalized = [normalize_keyword(p) for p in parts if p.strip()]
    return [n for n in normalized if n]


def extract_keywords(path: Path) -> list[tuple[int, str, str]]:
    results: list[tuple[int, str, str]] = []
    for idx, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        if TABLE_ALIGN_RE.match(line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        if cells[0] in {"キーワード", "---", ""}:
            continue
        # Reprint rows are references to the first definition; exclude them.
        if "［再掲" in cells[1]:
            continue
        for kw in split_keywords(cells[0]):
            results.append((idx, cells[0], kw))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect duplicate keywords in series/keywords.md")
    parser.add_argument(
        "--file",
        default="series/keywords.md",
        help="Path to keywords markdown file",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    entries = extract_keywords(path)
    by_key: dict[str, list[tuple[int, str, str]]] = {}
    for line_no, raw_cell, norm_kw in entries:
        if not norm_kw:
            continue
        by_key.setdefault(norm_kw, []).append((line_no, raw_cell, norm_kw))

    duplicates = {k: v for k, v in by_key.items() if len(v) > 1}

    if not duplicates:
        print("No duplicate keywords found.")
        return 0

    print(f"Duplicate keywords found in {path}:")
    for keyword in sorted(duplicates.keys()):
        print(f"- {keyword}")
        for line_no, raw_cell, _ in duplicates[keyword]:
            print(f"  line {line_no}: {raw_cell}")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
