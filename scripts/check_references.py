#!/usr/bin/env python3
"""参考文献セクションの書式チェック。

正規形（00.md の水準）:
  - 姓, 名イニシャル., & 姓, 名イニシャル. (年). タイトル. *誌名*, 巻(号). DOI/arXiv リンク付き
      - 注釈（1〜2行、インデント 4 スペース + "- "）

著者省略ルール: 3名以上の場合は「筆頭著者 et al.」形式を許容。

違反検出:
  V2: arXiv 裸表記（`*arXiv:XXXXXXX*` でリンクなし）
  V3: タイトルにリンクを張る形式（`[タイトル](URL)` が文献エントリ内に）
  V4: 注釈なし（文献エントリの次の行がインデント注釈でない）
  V5: appendix.5.md の参考文献セクションが空（エントリがない）
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 参考文献ブロックの開始
REF_SECTION_RE = re.compile(r"^## 参考文献")

# 文献エントリ行（トップレベルの "- " で始まる）
ENTRY_RE = re.compile(r"^- ")

# 注釈行（4 スペース + "- " または 2 スペース + "- "）
ANNOTATION_RE = re.compile(r"^ {2,}- ")

# カテゴリ見出し（### レベル）
CATEGORY_RE = re.compile(r"^### ")

# 空行
BLANK_RE = re.compile(r"^\s*$")

# 次セクション（##）
NEXT_SECTION_RE = re.compile(r"^## ")

# V2: arXiv 裸表記（*arXiv:数字* でリンクなし）
# リンク付きは arXiv: [arXiv:XXXX](https://...) なので、[ がない場合が違反
ARXIV_BARE_RE = re.compile(r"\*arXiv:\d{4}\.\d+\*")

# V3: タイトルにリンクを張る形式
# "- [タイトル](URL)" のパターン — 文献エントリ先頭行にタイトルリンク
TITLE_LINK_RE = re.compile(r"^- \[.+?\]\(https?://")

# 対象外ファイル
EXCLUDED_FILES = {"intro.md", "toc.md", "references.md", "keywords.md"}


def extract_ref_block(lines: list[str]) -> list[tuple[int, str]]:
    """参考文献セクション以降の行を (行番号1始まり, 行内容) で返す。"""
    in_ref = False
    result = []
    for i, line in enumerate(lines, 1):
        if REF_SECTION_RE.match(line):
            in_ref = True
            continue
        if in_ref:
            result.append((i, line.rstrip("\n")))
    return result


def check_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    errors: list[str] = []

    ref_block = extract_ref_block(lines)
    if not ref_block:
        return errors  # 参考文献セクションなし → 構造チェックで別途検出

    # V5: appendix.5.md の参考文献が空かチェック
    entry_lines = [line for _, line in ref_block if ENTRY_RE.match(line)]
    if path.name == "appendix.5.md" and not entry_lines:
        errors.append(f"{path}: 参考文献セクションにエントリがありません（appendix.5.md は新設対象）")
        return errors

    # 文献エントリを順に検査
    # ref_block は (lineno, content) のリスト
    i = 0
    while i < len(ref_block):
        lineno, line = ref_block[i]

        # エントリ行
        if ENTRY_RE.match(line):
            # V2: arXiv 裸表記
            if ARXIV_BARE_RE.search(line):
                errors.append(
                    f"{path}:{lineno}: arXiv 裸表記 (*arXiv:XXXX*) が含まれています"
                    "（arXiv: [arXiv:XXXX](https://arxiv.org/abs/XXXX) の形式にしてください）"
                )

            # V3: タイトルにリンクを張る形式
            if TITLE_LINK_RE.match(line):
                errors.append(
                    f"{path}:{lineno}: タイトルにリンクが張られています"
                    "（リンクは DOI/arXiv 番号に張り、タイトルは通常テキストにしてください）"
                )

            # V4: 注釈なし — 次の非空行を確認
            j = i + 1
            # 複数行エントリを読み飛ばす（行継続はない想定）
            next_annotation = False
            while j < len(ref_block):
                next_lineno, next_line = ref_block[j]
                if BLANK_RE.match(next_line):
                    j += 1
                    continue
                if ANNOTATION_RE.match(next_line):
                    next_annotation = True
                break

            if not next_annotation:
                msg = "参考文献エントリに注釈がありません（「なぜこの文献か」を 1 行で記載してください）"
                errors.append(f"{path}:{lineno}: {msg}")

        i += 1

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="参考文献書式チェック")
    parser.add_argument("series_dir", help="対象ディレクトリ（site/series/）")
    args = parser.parse_args()

    series_dir = Path(args.series_dir)
    if not series_dir.exists():
        print(f"{series_dir} が見つかりません。", file=sys.stderr)
        return 1

    all_errors: list[str] = []
    for path in sorted(series_dir.glob("*.md")):
        if path.name in EXCLUDED_FILES:
            continue
        all_errors.extend(check_file(path))

    if all_errors:
        print("参考文献チェックでエラーが見つかりました:", file=sys.stderr)
        for e in all_errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    print("参考文献チェック: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
