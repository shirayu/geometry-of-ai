#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

NEXT_RE = re.compile(r"^#{2,3}\\s+次回予告")
IMPL_RE = re.compile(r"^##\\s+実装ノート")
REF_RE = re.compile(r"^##\\s+参考文献")


def find_index(lines, regex):
    for i, line in enumerate(lines):
        if regex.search(line):
            return i
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check section order in markdown files.")
    parser.add_argument("series_dir", help="Directory containing markdown files")
    args = parser.parse_args()

    series_dir = Path(args.series_dir)
    if not series_dir.exists():
        print(f"{series_dir} が見つかりません。", file=sys.stderr)
        return 1

    errors = []
    for path in sorted(series_dir.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        next_i = find_index(lines, NEXT_RE)
        impl_i = find_index(lines, IMPL_RE)
        ref_i = find_index(lines, REF_RE)

        missing_targets = [
            ("次回予告", next_i),
            ("実装ノート", impl_i),
            ("参考文献", ref_i),
        ]
        missing = [name for name, idx in missing_targets if idx is None]
        if missing:
            if len(missing) == 3:
                continue
            errors.append(f"{path}: セクション未検出: {', '.join(missing)}")
            continue

        if not (next_i < impl_i < ref_i):
            errors.append(
                f"{path}: セクション順が不正 (次回予告={next_i + 1}, 実装ノート={impl_i + 1}, 参考文献={ref_i + 1})"
            )

    if errors:
        print("セクション順チェックでエラーが見つかりました:", file=sys.stderr)
        for e in errors:
            print(f"- {e}", file=sys.stderr)
        return 1

    print("セクション順チェック: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
