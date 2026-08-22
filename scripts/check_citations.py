#!/usr/bin/env python3
"""本文中の引用（（著者, 年））と章の参考文献リストの突合。

ルール:
  - 引用は全角括弧（著者, 年）。著者表記は bib から導出:
      1名 (A, 2016) / 2名 (A & B, 2015) / 3名以上 (A et al., 2017)。year_suffix 対応 (2016a 等)。
  - 本文中の引用は、その章の参考文献（site/series/refs/<chapter>.yml）に解決できなければならない。
  - 参考文献のエントリは本文中で1回以上引用されなければならない（nocite: true 指定は除外）。
  - 半角括弧の引用・会議名形式の引用（Puigcerver et al., ICLR 2024）は違反として報告。

対象外: references.md, intro.md, toc.md, quizzes, コードフェンス内。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sync_references import BibEntry, citation_label, load_yml, parse_bib  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIB = REPO_ROOT / "references.bib"
REF_SECTION_RE = re.compile(r"^## 参考文献[ \t]*$", re.MULTILINE)
FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
# sync_references.py がアンカーリンク化した [著者, 年](#ref-key) を 著者, 年 に戻す（検証前の前処理）。
CITE_LINK_RE = re.compile(r"\[([^\[\]]+?, \d{4}[a-z]?)\]\(#ref-[^()]+\)")

# （著者部, 年）: 著者部は大文字開始、括弧を含まない。セミコロン区切りの複数引用も扱う。
CITE_ITEM_RE = re.compile(r"^(?P<authors>.+), (?P<year>\d{4}[a-z]?)$")
FULLWIDTH_CITE_RE = re.compile(r"（([^（）]+?, \d{4}[a-z]?(?:; [^（）]+?, \d{4}[a-z]?)*)）")
HALFWIDTH_CITE_RE = re.compile(r"\(([^()]+?, \d{4}[a-z]?(?:; [^()]+?, \d{4}[a-z]?)*)\)")
# 会議名形式: （X et al., ICLR 2024）
VENUE_STYLE_RE = re.compile(r"[（(]([A-Z][^（）()]*? et al\., [A-Za-z]+ \d{4})[）)]")
# 叙述形式: X et al.（2024）/ 著者名（2024）
NARRATIVE_RE = re.compile(r"([A-Z][\w'-]*(?: et al\.)?) ?（(\d{4}[a-z]?)）")


def strip_fences_and_refs(text: str) -> str:
    """コードフェンス内と参考文献セクション以降を取り除く。

    フェンスは開いた記号（` か ~）と長さに対応する閉じフェンスでのみ閉じる
    （ブロック内の ````~~~~```` 行などで状態が反転しないようにする）。
    """
    m = REF_SECTION_RE.search(text)
    if m:
        text = text[: m.start()]
    out: list[str] = []
    in_fence = False
    marker_char = ""
    marker_len = 0
    for line in text.splitlines():
        fm = FENCE_RE.match(line)
        if fm:
            marker = fm.group(1)
            if not in_fence:
                in_fence = True
                marker_char = marker[0]
                marker_len = len(marker)
            elif marker[0] == marker_char and len(marker) >= marker_len and line.strip() == marker:
                in_fence = False
            out.append("")
            continue
        out.append("" if in_fence else line)
    return "\n".join(out)


def all_surnames(bib: dict[str, BibEntry]) -> set[str]:
    names: set[str] = set()
    for e in bib.values():
        if "author" not in e.fields:
            continue
        first = e.authors[0].split(",")[0].strip()
        names.add(first)
        names.add(first.split()[0])  # 複合姓の先頭語（検出用）
    return names


def check_chapter(
    yml_path: Path,
    series_dir: Path,
    bib: dict[str, BibEntry],
    labels: dict[str, list[str]],
    surnames: set[str],
) -> list[str]:
    errors: list[str] = []
    md_path = series_dir / f"{yml_path.stem}.md"
    if not md_path.exists():
        return [f"{yml_path}: 対応する md ファイルがありません"]
    data = load_yml(yml_path)
    chapter_keys: list[str] = []
    nocite: set[str] = set()
    for cat in data.get("categories", []):
        for ent in cat.get("entries", []):
            chapter_keys.append(ent["key"])
            if ent.get("nocite"):
                nocite.add(ent["key"])

    # 章内のラベル衝突（year_suffix で解決すべき状態）
    chapter_labels: dict[str, set[str]] = {}
    for key in chapter_keys:
        if key in bib:
            chapter_labels.setdefault(citation_label(bib[key]), set()).add(key)
    for label, keys in chapter_labels.items():
        if len(keys) > 1:
            errors.append(
                f"{md_path}: 引用ラベル「{label}」が章内で衝突: {', '.join(sorted(keys))}"
                "（year_suffix で区別してください）"
            )

    body = strip_fences_and_refs(md_path.read_text(encoding="utf-8"))
    body = CITE_LINK_RE.sub(r"\1", body)

    for m in VENUE_STYLE_RE.finditer(body):
        errors.append(f"{md_path}: 会議名形式の引用があります（年号で引用してください）: （{m.group(1)}）")

    cited_labels: set[str] = set()

    def handle_item(authors_part: str, year: str, halfwidth: bool, whole: str) -> None:
        label = f"{authors_part.strip()}, {year}"
        looks_author = "et al." in authors_part or authors_part.split()[0] in surnames
        if label not in labels and not looks_author:
            # venue 名（JMLR 等の総文字大文字）は除外しつつ、未知の著者名は警告
            word = authors_part.split()[0].strip(".,;")
            if word and not word.isupper():
                errors.append(f"{md_path}: 引用（{label}）に対応する bib エントリがありません")
            return
        if halfwidth:
            errors.append(f"{md_path}: 半角括弧の引用 → 全角にしてください: ({whole})")
        cited_labels.add(label)
        if label not in labels:
            errors.append(f"{md_path}: 引用（{label}）に対応する bib エントリがありません")
        elif not any(k in chapter_keys for k in labels[label]):
            errors.append(f"{md_path}: 引用（{label}）がこの章の参考文献にありません（refs/{yml_path.name} に追加を）")

    def handle(content: str, halfwidth: bool) -> None:
        for part in content.split("; "):
            m = CITE_ITEM_RE.match(part.strip())
            if m:
                handle_item(m.group("authors"), m.group("year"), halfwidth, content)

    for m in FULLWIDTH_CITE_RE.finditer(body):
        handle(m.group(1), halfwidth=False)
    for m in HALFWIDTH_CITE_RE.finditer(body):
        handle(m.group(1), halfwidth=True)

    # 叙述形式（X et al.（2024））は未対応の形式として検出
    for m in NARRATIVE_RE.finditer(body):
        head = m.group(1)
        if "et al." in head and head.split()[0] in surnames:
            errors.append(f"{md_path}: 叙述形式の引用（（著者, 年）形式にしてください）: {head}（{m.group(2)}）")

    # 参考文献エントリの本文引用カバー率
    seen: set[str] = set()
    for key in chapter_keys:
        if key in seen or key in nocite:
            continue
        seen.add(key)
        if key not in bib:
            errors.append(f"{md_path}: bib に存在しないキーが章に収録されています: {key}")
            continue
        label = citation_label(bib[key])
        if label not in cited_labels:
            errors.append(f"{md_path}: 参考文献 {key} が本文中で一度も引用されていません: （{label}）")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="本文引用と参考文献の突合チェック")
    parser.add_argument("series_dir", nargs="?", default=str(REPO_ROOT / "site" / "series"))
    parser.add_argument("--bib", default=str(DEFAULT_BIB))
    args = parser.parse_args()

    series_dir = Path(args.series_dir)
    refs_dir = series_dir / "refs"
    bib = parse_bib(Path(args.bib).read_text(encoding="utf-8"))
    labels: dict[str, list[str]] = {}  # citation label → keys
    for key, e in bib.items():
        if "author" in e.fields and "year" in e.fields:
            labels.setdefault(citation_label(e), []).append(key)
    surnames = all_surnames(bib)

    errors: list[str] = []
    for yml_path in sorted(refs_dir.glob("*.yml")):
        if yml_path.stem == "readinglist":
            continue
        errors.extend(check_chapter(yml_path, series_dir, bib, labels, surnames))

    if errors:
        print("引用チェックでエラーが見つかりました:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    print("引用チェック: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
