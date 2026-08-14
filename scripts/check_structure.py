#!/usr/bin/env python3
"""章構造の正規形チェック（check_section_order.py の後継）。

正規形（本章 00〜14.md）:
  # 第N回：タイトル ～サブタイトル～   ← タイトル行の波ダッシュは U+FF5E(～)
  ## 注意事項          ← 箇条書き + 末尾引用句
  ## 導入：…
  （本文セクション群）
  ## まとめ：…（副題付き）
  ## 次回予告          ← ## レベル必須（### は違反）
  ## 実装ノート：…（副題付き）
  ## 参考文献

例外:
  15.md       — 次回予告・実装ノートなし（最終回）。波ダッシュのみチェック。
  appendix.*  — 注意事項・次回予告なし。まとめ副題・実装ノート副題・参考文献のみチェック。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# タイトル行（# で始まる行）の波ダッシュ: U+301C(〜) は NG、U+FF5E(～) が正
WAVEDASH_BAD = re.compile(r"^#{1,6} .*〜")  # U+301C、見出し行限定

# 末尾セクション群
SUMMARY_RE = re.compile(r"^## まとめ")
NEXT_RE = re.compile(r"^## 次回予告")  # ## のみ（### は違反）
NEXT_ANY_RE = re.compile(r"^#{2,3} 次回予告")  # 違反検出用
IMPL_RE = re.compile(r"^## 実装ノート")
REF_RE = re.compile(r"^## 参考文献")

# 副題なし（コロンが続かない）
SUMMARY_NO_SUBTITLE = re.compile(r"^## まとめ$")
IMPL_NO_SUBTITLE = re.compile(r"^## 実装ノート$")

# 注意事項ブロック
CAUTION_HEADER = re.compile(r"^## 注意事項$")
NEXT_SECTION = re.compile(r"^## ")
BULLET = re.compile(r"^- ")
BLOCKQUOTE = re.compile(r"^> ")


def find_first(lines: list[str], regex: re.Pattern) -> int | None:
    for i, line in enumerate(lines):
        if regex.search(line):
            return i
    return None


def check_caution_block(lines: list[str], path: Path) -> list[str]:
    """注意事項ブロックに箇条書きと末尾引用句があるか確認する。"""
    errors = []
    start = find_first(lines, CAUTION_HEADER)
    if start is None:
        errors.append(f"{path}: ## 注意事項 セクションが見つかりません")
        return errors

    # 注意事項ブロック（次の ## まで）
    block = []
    for line in lines[start + 1 :]:
        if NEXT_SECTION.match(line):
            break
        block.append(line)

    has_bullet = any(BULLET.match(line) for line in block)
    has_quote = any(BLOCKQUOTE.match(line) for line in block)

    if not has_bullet:
        errors.append(f"{path}: 注意事項ブロックに箇条書き（- ）がありません")
    if not has_quote:
        errors.append(f"{path}: 注意事項ブロックに末尾引用句（> ）がありません")

    return errors


def check_file(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    errors = []
    name = path.name
    is_appendix = name.startswith("appendix.")

    # --- R6: 見出し行の波ダッシュ（全ファイル共通） ---
    for i, line in enumerate(lines, 1):
        if WAVEDASH_BAD.search(line):
            errors.append(f"{path}:{i}: 見出し行に U+301C(〜) が含まれています（U+FF5E(～) を使用してください）")

    # --- R1: まとめに副題がない（全ファイル共通） ---
    for i, line in enumerate(lines, 1):
        if SUMMARY_NO_SUBTITLE.match(line):
            errors.append(f"{path}:{i}: ## まとめ に副題がありません（例: ## まとめ：〇〇）")

    # --- R2: 実装ノートに副題がない（全ファイル共通） ---
    for i, line in enumerate(lines, 1):
        if IMPL_NO_SUBTITLE.match(line):
            errors.append(f"{path}:{i}: ## 実装ノート に副題がありません（例: ## 実装ノート：〇〇）")

    # appendix は注意事項・次回予告なしが正常。まとめ・参考文献の順序だけ確認。
    if is_appendix:
        summary_i = find_first(lines, SUMMARY_RE)
        ref_i = find_first(lines, REF_RE)
        if summary_i is None:
            errors.append(f"{path}: ## まとめ セクションが見つかりません")
        if ref_i is None:
            errors.append(f"{path}: ## 参考文献 セクションが見つかりません")
        if summary_i is not None and ref_i is not None and summary_i >= ref_i:
            errors.append(
                f"{path}: セクション順が不正 — まとめ({summary_i + 1}行目) が"
                f" 参考文献({ref_i + 1}行目) より後にあります"
            )
        return errors

    # --- R3: 次回予告が ### になっていないか（本章のみ） ---
    for i, line in enumerate(lines, 1):
        if NEXT_ANY_RE.match(line) and not NEXT_RE.match(line):
            errors.append(f"{path}:{i}: 次回予告の見出しレベルが ### です（## に昇格させてください）")

    # --- R4: 注意事項ブロック（本章のみ、15.md は除く） ---
    if name != "15.md":
        errors.extend(check_caution_block(lines, path))

    # --- R5: 章末セクションの順序と存在（本章のみ） ---
    summary_i = find_first(lines, SUMMARY_RE)
    next_i = find_first(lines, NEXT_RE)
    impl_i = find_first(lines, IMPL_RE)
    ref_i = find_first(lines, REF_RE)

    # 15.md: 注意事項・次回予告・実装ノートなしを許容（最終回）
    if name == "15.md":
        if summary_i is None:
            errors.append(f"{path}: ## まとめ セクションが見つかりません")
        if ref_i is None:
            errors.append(f"{path}: ## 参考文献 セクションが見つかりません")
        return errors

    present = {"まとめ": summary_i, "次回予告": next_i, "実装ノート": impl_i, "参考文献": ref_i}
    missing = [k for k, v in present.items() if v is None]
    if missing:
        errors.append(f"{path}: セクション未検出: {', '.join(missing)}")
        return errors

    # まとめ → 次回予告 → 実装ノート → 参考文献 の順
    order = [("まとめ", summary_i), ("次回予告", next_i), ("実装ノート", impl_i), ("参考文献", ref_i)]
    for (a_name, a_i), (b_name, b_i) in zip(order, order[1:], strict=False):
        if a_i >= b_i:
            errors.append(
                f"{path}: セクション順が不正 — {a_name}({a_i + 1}行目) が {b_name}({b_i + 1}行目) より後にあります"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="章構造の正規形チェック")
    parser.add_argument("series_dir", help="対象ディレクトリ（site/series/）")
    args = parser.parse_args()

    series_dir = Path(args.series_dir)
    if not series_dir.exists():
        print(f"{series_dir} が見つかりません。", file=sys.stderr)
        return 1

    all_errors: list[str] = []
    for path in sorted(series_dir.glob("*.md")):
        # 構成ファイルはスキップ
        if path.name in {"intro.md", "toc.md", "references.md", "keywords.md", "quizzes.md"}:
            continue
        all_errors.extend(check_file(path))

    if all_errors:
        print("構造チェックでエラーが見つかりました:", file=sys.stderr)
        for e in all_errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    print("構造チェック: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
