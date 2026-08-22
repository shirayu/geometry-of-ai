#!/usr/bin/env python3
"""参考文献の同期: references.bib + site/series/refs/*.yml → markdown。

正（source of truth）:
  - references.bib: 書誌情報（著者・タイトル・掲載誌・DOI/arXiv）
  - site/series/refs/<chapter>.yml: 章ごとのカテゴリ構成と注釈
  - site/series/refs/readinglist.yml: references.md（通読リスト）の構成と注釈

生成物:
  - 各章 md の `## 参考文献` セクション（ファイル末尾まで置換）
  - site/series/references.md 全体
  - 本文中の（著者, 年）引用を、章の参考文献セクション内アンカー（#ref-<key>）への
    Markdown リンクに変換（`（[著者, 年](#ref-key)）`）

正規形の書誌レンダリング（00.md の水準）:
  - 著者: 「A, X., B, Y., & C, Z.」／2名「A, X., & B, Y.」／7名以上「A, X., et al.」
  - 論文タイトルは通常テキスト、誌名・会議名は *イタリック*
  - DOI 優先、DOI なしなら arXiv リンク（arXiv: [id](https://arxiv.org/abs/id)）
  - セグメント末尾がリンクで終わる場合は末尾ピリオドを付けない

使い方:
  python3 scripts/sync_references.py apply   # markdown に書き込む
  python3 scripts/sync_references.py check   # 差分があれば exit 1（lint 用）
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIB_PATH = REPO_ROOT / "references.bib"
REF_SECTION_HEADER = "## 参考文献"
READING_LIST_HEADER = "# 参考文献"
ET_AL_AUTHOR_THRESHOLD = 7

DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
EPRINT_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
REQUIRED_FIELDS = ("author", "title", "year")
CITE_ITEM_RE = re.compile(r"^(?P<authors>.+), (?P<year>\d{4}[a-z]?)$")


@dataclass
class BibEntry:
    key: str
    entry_type: str
    fields: dict[str, str] = field(default_factory=dict)

    @property
    def authors(self) -> list[str]:
        return [a.strip() for a in self.fields["author"].split(" and ")]

    @property
    def year(self) -> str:
        return self.fields["year"]

    @property
    def title(self) -> str:
        return self.fields["title"]


# ---------------------------------------------------------------------------
# .bib パース（管理対象サブセット: @type{key, field = {value}, ...}）
# ---------------------------------------------------------------------------


def parse_bib(text: str) -> dict[str, BibEntry]:
    entries: dict[str, BibEntry] = {}
    i = 0
    n = len(text)
    while True:
        at = text.find("@", i)
        if at == -1:
            break
        m = re.match(r"@(\w+)\s*\{", text[at:])
        if not m:
            i = at + 1
            continue
        entry_type = m.group(1).lower()
        j = at + m.end()
        # key: 最初のコンマまで
        comma = text.find(",", j)
        if comma == -1:
            break
        key = text[j:comma].strip()
        # エントリ本体: 対応する閉じ brace まで
        depth = 1
        k = comma + 1
        while k < n and depth > 0:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
            k += 1
        body = text[comma + 1 : k - 1]
        entries[key] = BibEntry(key=key, entry_type=entry_type, fields=_parse_fields(body))
        i = k
    return entries


def _parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for m in re.finditer(r"(\w+)\s*=\s*\{", body):
        name = m.group(1).lower()
        start = m.end()
        depth = 1
        k = start
        while k < len(body) and depth > 0:
            if body[k] == "{":
                depth += 1
            elif body[k] == "}":
                depth -= 1
            k += 1
        fields[name] = body[start : k - 1].strip()
    return fields


# ---------------------------------------------------------------------------
# 書誌レンダリング
# ---------------------------------------------------------------------------


def _surname(author: str) -> str:
    return author.split(",")[0].strip()


def render_authors(authors: list[str]) -> str:
    if len(authors) >= ET_AL_AUTHOR_THRESHOLD:
        return f"{authors[0]}, et al."
    if len(authors) == 1:
        return authors[0]
    return ", ".join(authors[:-1]) + ", & " + authors[-1]


def year_label(entry: BibEntry) -> str:
    """表示・引用に使う年（同一ラベルの曖昧性回避用 year_suffix 対応: 2016a 等）。"""
    return entry.year + entry.fields.get("year_suffix", "")


def citation_label(entry: BibEntry) -> str:
    """本文中の引用表記（（…）の中身）: 1名「A」, 2名「A & B」, 3名以上「A et al.»+年。

    citename フィールドがあれば筆頭著者の表記として使う（日本語文献の「渡辺」等）。
    """
    surnames = [_surname(a) for a in entry.authors]
    if "citename" in entry.fields:
        surnames[0] = entry.fields["citename"]
    if len(surnames) == 1:
        base = surnames[0]
    elif len(surnames) == 2:
        base = f"{surnames[0]} & {surnames[1]}"
    else:
        base = f"{surnames[0]} et al."
    return f"{base}, {year_label(entry)}"


def _pages(pages: str) -> str:
    return pages.replace("--", "–")


def _finalize_segment(segment: str) -> str:
    """セグメント末尾を整える: Markdownリンク・?/! 終端以外にはピリオドを付与。

    「(2nd ed.)」のような括弧終端にはピリオドが必要で、
    「[x](https://...)」や「(2016)」のようなリンク/注記の括弧には不要。
    """
    if not segment or segment.endswith(("?", "!", ".")):
        return segment
    if segment.endswith(")") and ("](" in segment or re.search(r"\(\d{4}\)$", segment)):
        return segment
    return segment + "."


def _venue_segment(entry: BibEntry) -> str:
    """誌名・会議名・出版社セグメント（*イタリック* と巻号ページ）。"""
    f = entry.fields
    if entry.entry_type == "article":
        parts = [f"*{f['journal']}*"]
        if "volume" in f:
            vol = f["volume"]
            if "number" in f:
                vol += f"({f['number']})"
            parts.append(vol)
        if "pages" in f:
            parts.append(_pages(f["pages"]))
        return ", ".join(parts)
    if entry.entry_type == "inproceedings":
        parts = [f"*{f['booktitle']}*"]
        if "pages" in f:
            parts.append(_pages(f["pages"]))
        if "publisher" in f:
            parts.append(f["publisher"])
        return ", ".join(parts)
    if entry.entry_type == "book":
        bits = []
        if "series" in f:
            series = f["series"]
            if "volume" in f:
                series += f", Vol. {f['volume']}"
            bits.append(series)
        bits.append(f["publisher"])
        return ". ".join(bits)
    return ""  # misc（arXiv のみ）など


def render_entry(entry: BibEntry) -> str:
    """正規形の書誌文字列（先頭の '- ' を除く1行）を返す。"""
    f = entry.fields
    segments = [f"{render_authors(entry.authors)} ({year_label(entry)})."]

    title = entry.title
    if entry.entry_type == "book":
        title = f"*{title}*"
        if "edition" in f:
            title += f" ({f['edition']} ed.)"
    segments.append(_finalize_segment(title))

    venue = _venue_segment(entry)
    if venue:
        segments.append(_finalize_segment(venue))

    if "doi" in f:
        segments.append(f"DOI: [{f['doi']}](https://doi.org/{f['doi']})")
    elif "eprint" in f:
        arxiv = f"arXiv: [{f['eprint']}](https://arxiv.org/abs/{f['eprint']})"
        if "eprintyear" in f:
            arxiv += f" ({f['eprintyear']})"
        segments.append(arxiv)

    return " ".join(segments)


def render_short(entry: BibEntry) -> str:
    """references.md 用の短い表記: 著者, "タイトル" (掲載誌 年)。"""
    f = entry.fields
    surnames = [_surname(a) for a in entry.authors]
    if len(surnames) == 1:
        label = surnames[0]  # 日本語単独著者（渡辺澄夫 等）はフルネーム表示
    elif len(surnames) == 2:
        label = f"{surnames[0]} & {surnames[1]}"
    else:
        label = f"{surnames[0]} et al."
    if entry.entry_type == "inproceedings":
        venue = f["booktitle"]
    elif entry.entry_type == "article":
        venue = f.get("venueshort") or f["journal"]
        venue = f"{venue} {entry.year}"
    elif entry.entry_type == "book":
        venue = f"{f['publisher']} {entry.year}"
    elif "eprint" in f:
        venue = f"arXiv:{f['eprint']}, {entry.year}"
    else:
        venue = entry.year
    title = entry.title
    if title.startswith("『"):
        return f"{label}, {title} ({venue})"
    return f'{label}, "{title}" ({venue})'


# ---------------------------------------------------------------------------
# 章 yml → 参考文献セクション
# ---------------------------------------------------------------------------


def load_yml(path: Path) -> dict:
    import yaml  # pyyaml（pyproject.toml 依存）

    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def ref_anchor_id(key: str) -> str:
    return f"ref-{key}"


def render_ref_section(data: dict, bib: dict[str, BibEntry]) -> str:
    """章の `## 参考文献` セクション（見出し〜ファイル末尾）を生成する。

    各項目の先頭に本文からのアンカーリンク用 id を埋め込む（同一キーが章内で
    複数回掲載される場合、id の重複を避けるため2回目以降には付与しない）。
    """
    lines = [REF_SECTION_HEADER, ""]
    seen_keys: set[str] = set()
    for cat in data.get("categories", []):
        if cat.get("title"):
            lines.append(f"### {cat['title']}")
            lines.append("")
        for ent in cat.get("entries", []):
            key = ent["key"]
            anchor = f'<a id="{ref_anchor_id(key)}"></a>' if key not in seen_keys else ""
            seen_keys.add(key)
            lines.append(f"- {anchor}{render_entry(bib[key])}")
            for note in ent.get("note", []):
                lines.append(f"    - {note}")
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def render_reading_list(data: dict, bib: dict[str, BibEntry]) -> str:
    lines = [READING_LIST_HEADER, ""]
    for cat in data.get("categories", []):
        lines.append(f"## {cat['title']}")
        lines.append("")
        for ent in cat.get("entries", []):
            lines.append(f"- {render_short(bib[ent['key']])}")
            for note in ent.get("note", []):
                lines.append(f"    - {note}")
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    text = "\n".join(lines) + "\n"
    if data.get("footer"):
        text += "\n" + data["footer"].rstrip("\n") + "\n"
    return text


# ---------------------------------------------------------------------------
# バリデーション
# ---------------------------------------------------------------------------


def validate_bib(bib: dict[str, BibEntry]) -> list[str]:
    errors: list[str] = []
    for key, e in bib.items():
        for req in REQUIRED_FIELDS:
            if req not in e.fields:
                errors.append(f"references.bib: {key}: フィールド {req} がありません")
        doi = e.fields.get("doi")
        if doi and not DOI_RE.match(doi):
            errors.append(f"references.bib: {key}: DOI の形式が不正です: {doi}")
        eprint = e.fields.get("eprint")
        if eprint and not EPRINT_RE.match(eprint):
            errors.append(f"references.bib: {key}: arXiv ID の形式が不正です: {eprint}")
        if e.fields.get("eprintyear") and not eprint:
            errors.append(f"references.bib: {key}: eprintyear がありますが eprint がありません")
        if e.entry_type == "article" and "journal" not in e.fields:
            errors.append(f"references.bib: {key}: @article には journal が必要です")
        if e.entry_type == "inproceedings" and "booktitle" not in e.fields:
            errors.append(f"references.bib: {key}: @inproceedings には booktitle が必要です")
        if e.entry_type == "book" and "publisher" not in e.fields:
            errors.append(f"references.bib: {key}: @book には publisher が必要です")
    # 注: 引用ラベルの章内重複は check_citations.py が検出する
    # （章をまたぐ同名ラベルは sidecar で解決できるため bib 全体では禁止しない）
    return errors


def collect_used_keys(refs_dir: Path) -> set[str]:
    used: set[str] = set()
    for path in sorted(refs_dir.glob("*.yml")):
        data = load_yml(path)
        for cat in data.get("categories", []):
            for ent in cat.get("entries", []):
                used.add(ent["key"])
    return used


def validate_refs(refs_dir: Path, bib: dict[str, BibEntry]) -> list[str]:
    errors: list[str] = []
    used = collect_used_keys(refs_dir)
    for key in sorted(used - set(bib)):
        errors.append(f"refs/*.yml: bib に存在しないキーが参照されています: {key}")
    for key in sorted(set(bib) - used):
        errors.append(f"references.bib: どこからも参照されていないエントリです: {key}")
    return errors


def validate_yml(refs_dir: Path) -> list[str]:
    """章ファイルの注釈欠落をチェック（同一キーの複数掲載は意図的な重複として許容）。"""
    errors: list[str] = []
    for path in sorted(refs_dir.glob("*.yml")):
        if path.stem == "readinglist":
            continue  # 通読リストは注釈任意
        data = load_yml(path)
        for cat in data.get("categories", []):
            for ent in cat.get("entries", []):
                if not ent.get("note"):
                    errors.append(f"{path}: {ent['key']}: 注釈がありません")
    return errors


# ---------------------------------------------------------------------------
# apply / check
# ---------------------------------------------------------------------------


REF_SECTION_LINE_RE = re.compile(rf"^({re.escape(REF_SECTION_HEADER)})[ \t]*$", re.MULTILINE)
FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
# （著者部, 年）: 既にリンク済み（（[…](#…)）等）は対象外。check_citations.py の
# FULLWIDTH_CITE_RE と同じ著者部パターンだが、直前に "[" が続く場合は除外する。
FULLWIDTH_CITE_RE = re.compile(r"（([^（）\[\]]+?, \d{4}[a-z]?(?:; [^（）\[\]]+?, \d{4}[a-z]?)*)）")


def md_path_for(yml_path: Path) -> Path:
    if yml_path.stem == "readinglist":
        return yml_path.parent.parent / "references.md"
    return yml_path.parent.parent / f"{yml_path.stem}.md"


def linkify_citations(body: str, chapter_keys: list[str], labels: dict[str, list[str]]) -> str:
    """本文中の（著者, 年）を、章の参考文献に解決できる場合に限りアンカーリンク化する。

    - 対象はコードフェンス外かつ `## 参考文献` セクションより前の部分のみ。
    - セミコロン区切りの複数引用（（A, 2020; B, 2021））は要素ごとに個別リンク化する。
    - ラベルが章内の複数キーに衝突する場合はリンク化をスキップする（year_suffix 等で
      本来は一意になるはずだが、安全側に倒す）。
    """
    chapter_key_set = set(chapter_keys)

    def resolve(label: str) -> str | None:
        keys = [k for k in labels.get(label, []) if k in chapter_key_set]
        if len(keys) != 1:
            return None
        return keys[0]

    def replace_item(part: str) -> str:
        m = CITE_ITEM_RE.match(part.strip())
        if not m:
            return part
        label = f"{m.group('authors').strip()}, {m.group('year')}"
        key = resolve(label)
        if key is None:
            return part
        return f"[{label}]({'#' + ref_anchor_id(key)})"

    def replace_match(m: re.Match) -> str:
        parts = m.group(1).split("; ")
        return "（" + "; ".join(replace_item(p) for p in parts) + "）"

    out_lines: list[str] = []
    in_fence = False
    marker_char = ""
    marker_len = 0
    for line in body.splitlines():
        fm = FENCE_RE.match(line)
        if fm:
            marker = fm.group(1)
            if not in_fence:
                in_fence = True
                marker_char = marker[0]
                marker_len = len(marker)
            elif marker[0] == marker_char and len(marker) >= marker_len and line.strip() == marker:
                in_fence = False
            out_lines.append(line)
            continue
        out_lines.append(line if in_fence else FULLWIDTH_CITE_RE.sub(replace_match, line))
    result = "\n".join(out_lines)
    if body.endswith("\n"):
        result += "\n"
    return result


def apply_chapter(yml_path: Path, bib: dict[str, BibEntry], labels: dict[str, list[str]], dry_run: bool) -> str | None:
    """章 md の参考文献セクションを生成物で置き換える。変更があれば報告文を返す。"""
    md_path = md_path_for(yml_path)
    data = load_yml(yml_path)
    if yml_path.stem == "readinglist":
        new_text = render_reading_list(data, bib)
        old_text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
        if new_text == old_text:
            return None
        if not dry_run:
            md_path.write_text(new_text, encoding="utf-8")
            return None
        return f"{md_path}: 内容が乖離しています"

    text = md_path.read_text(encoding="utf-8")
    m = REF_SECTION_LINE_RE.search(text)
    if not m:
        return f"{md_path}: {REF_SECTION_HEADER} セクションが見つかりません"

    chapter_keys = [ent["key"] for cat in data.get("categories", []) for ent in cat.get("entries", [])]
    body = linkify_citations(text[: m.start()], chapter_keys, labels)
    new_text = body + render_ref_section(data, bib)
    if new_text == text:
        return None
    if not dry_run:
        md_path.write_text(new_text, encoding="utf-8")
        return None
    return f"{md_path}: 参考文献セクションが乖離しています"


def build_labels(bib: dict[str, BibEntry]) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    for key, e in bib.items():
        if "author" in e.fields and "year" in e.fields:
            labels.setdefault(citation_label(e), []).append(key)
    return labels


def run(check: bool, series_dir: Path, bib_path: Path) -> int:
    refs_dir = series_dir / "refs"
    errors: list[str] = []

    if not bib_path.exists():
        print(f"{bib_path} が見つかりません。", file=sys.stderr)
        return 1
    bib = parse_bib(bib_path.read_text(encoding="utf-8"))
    errors += validate_bib(bib)
    errors += validate_refs(refs_dir, bib)
    errors += validate_yml(refs_dir)
    labels = build_labels(bib)

    for yml_path in sorted(refs_dir.glob("*.yml")):
        result = apply_chapter(yml_path, bib, labels, dry_run=check)
        if result:
            errors.append(result)

    if errors:
        label = "同期チェック" if check else "apply"
        print(f"参考文献{label}でエラーが見つかりました:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    print("参考文献同期: OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="参考文献の同期（bib + yml → markdown）")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("apply", "check"):
        p = sub.add_parser(name)
        p.add_argument("--series-dir", default=str(REPO_ROOT / "site" / "series"))
        p.add_argument("--bib", default=str(DEFAULT_BIB_PATH))
    args = parser.parse_args()
    return run(check=args.command == "check", series_dir=Path(args.series_dir), bib_path=Path(args.bib))


if __name__ == "__main__":
    raise SystemExit(main())
