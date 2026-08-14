#!/usr/bin/env python3
"""「第N回」「Appendix N」参照のリンク化チェック。

本文中で他章を指す「第N回」「Appendix N」という表現は、対象ファイルへの
Markdownリンク（例: [第4回](04.md) / [Appendix 2](appendix.2.md)）で
書かれていなければならない。

また、既にリンク化されている「第N回」「Appendix N」についても、
リンクテキストの番号とリンク先ファイル名の番号が一致しているか、
リンク先ファイルが実在するかを検証する。

除外対象:
  - 見出し行（# 第N回：... / ## Appendix N:... など。keywords.md の章区切り見出しを含む）
  - 自己参照（自ファイルを指す「第N回」「Appendix N」。例: 00.md 内の「第0回（本回）」）
  - quiz.md（STYLE_GUIDE.md により講義本体への参照語は禁止）
  - コードフェンス内
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
TITLE_LECTURE_RE = re.compile(r"^#+\s*第\d+回[：:]")
TITLE_APPENDIX_RE = re.compile(r"^#+\s*Appendix\s*\d+[：:]")

# 「第N回」の直前が "[" でなければ未リンク（既存リンクテキスト内は許容）
LECTURE_RE = re.compile(r"第(\d+)回(?!補論)")
APPENDIX_RE = re.compile(r"Appendix\s*(\d+)")

# 直後に補足として "補論" が続く形式は、appendix.M.md を機械的に特定できないため
# 個別に手当てされている前提で対象外とする（第N回補論 は別ルールで検出）
LECTURE_HOJIRON_RE = re.compile(r"第(\d+)回補論")

LECTURE_FILE_RE = re.compile(r"^(\d{2})\.md$")
APPENDIX_FILE_RE = re.compile(r"^appendix\.(\d+)\.md$")


def _self_numbers(path: Path) -> tuple[int | None, int | None]:
    """(自ファイルの講義回番号, 自ファイルのAppendix番号) を返す。該当しなければ None。"""
    m = LECTURE_FILE_RE.match(path.name)
    if m:
        return int(m.group(1)), None
    m = APPENDIX_FILE_RE.match(path.name)
    if m:
        return None, int(m.group(1))
    return None, None


def _fence_marker(line: str) -> str | None:
    m = FENCE_RE.match(line.lstrip())
    return m.group(1)[:3] if m else None


def _links(line: str) -> list[tuple[int, int, str, str]]:
    """[text](url) の (text_start, text_end, text, url) のリストを返す。"""
    result = []
    for m in re.finditer(r"\[([^\]]*)\]\(([^)]*)\)", line):
        result.append((m.start(1), m.end(1), m.group(1), m.group(2)))
    return result


def _linked_spans(links: list[tuple[int, int, str, str]]) -> list[tuple[int, int]]:
    return [(s, e) for s, e, _, _ in links]


def _in_span(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in spans)


LINK_TARGET_LECTURE_RE = re.compile(r"^(?:\./)?(\d{2})\.md")
LINK_TARGET_APPENDIX_RE = re.compile(r"^(?:\./)?appendix\.(\d+)\.md")


REKAI_RE = re.compile(r"^［再掲・.*］$")


def _check_existing_links(
    path: Path, line_no: int, links: list[tuple[int, int, str, str]], series_dir: Path
) -> list[str]:
    errors: list[str] = []
    for _, _, text, url in links:
        # keywords.md の「[［再掲・第N回］](#anchor)」は同ファイル内アンカーへの
        # ショートカットであり、第N回はあくまで初出回の注記なので対象外。
        if REKAI_RE.match(text):
            continue

        text_lecture_nums = {int(n) for n in re.findall(r"第(\d+)回(?!補論)", text)}
        text_appendix_nums = {int(n) for n in re.findall(r"Appendix\s*(\d+)", text)}
        if not text_lecture_nums and not text_appendix_nums:
            continue

        m_lecture = LINK_TARGET_LECTURE_RE.match(url)
        m_appendix = LINK_TARGET_APPENDIX_RE.match(url)

        if text_lecture_nums:
            if not m_lecture:
                errors.append(
                    f"{path}:{line_no}: リンクテキスト「{text}」は第N回を指しますが、"
                    f"リンク先「{url}」がNN.md形式ではありません"
                )
            else:
                target_n = int(m_lecture.group(1))
                if target_n not in text_lecture_nums:
                    errors.append(f"{path}:{line_no}: リンクテキスト「{text}」とリンク先「{url}」の回数が一致しません")

        if text_appendix_nums:
            if not m_appendix:
                errors.append(
                    f"{path}:{line_no}: リンクテキスト「{text}」はAppendix Nを指しますが、"
                    f"リンク先「{url}」がappendix.N.md形式ではありません"
                )
            else:
                target_n = int(m_appendix.group(1))
                if target_n not in text_appendix_nums:
                    errors.append(
                        f"{path}:{line_no}: リンクテキスト「{text}」とリンク先「{url}」のAppendix番号が一致しません"
                    )

        # リンク先ファイルの実在確認（NN.md / appendix.N.md 形式のみ）
        if m_lecture or m_appendix:
            target_path = series_dir / url.split("#")[0]
            if not target_path.exists():
                errors.append(f"{path}:{line_no}: リンク先ファイルが存在しません「{url}」")

    return errors


def check_file(path: Path, series_dir: Path) -> list[str]:
    errors: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    self_lecture, self_appendix = _self_numbers(path)

    in_fence = False
    fence_marker = ""

    for line_no, line in enumerate(lines, start=1):
        marker = _fence_marker(line)
        if marker:
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            continue

        if in_fence:
            continue

        if TITLE_LECTURE_RE.match(line) or TITLE_APPENDIX_RE.match(line):
            continue

        links = _links(line)
        spans = _linked_spans(links)
        errors.extend(_check_existing_links(path, line_no, links, series_dir))

        for m in LECTURE_HOJIRON_RE.finditer(line):
            if _in_span(m.start(), spans):
                continue
            # 講義回番号とAppendix番号は機械的に対応しないため、
            # どのappendix.N.mdを指すかはこのスクリプトでは判定しない。
            # 自ファイルが該当appendixの場合のみ自己参照として除外する。
            if self_appendix is not None:
                continue
            errors.append(
                f"{path}:{line_no}: 「{m.group(0)}」が未リンクです"
                "（appendix.N.md への Markdown リンクにしてください。"
                "講義回番号とAppendix番号は一致しないため対応表を確認）"
            )

        for m in LECTURE_RE.finditer(line):
            if _in_span(m.start(), spans):
                continue
            n = int(m.group(1))
            if n == self_lecture:
                continue
            errors.append(
                f"{path}:{line_no}: 「{m.group(0)}」が未リンクです（[{m.group(0)}]({n:02d}.md) の形式にしてください）"
            )

        for m in APPENDIX_RE.finditer(line):
            if _in_span(m.start(), spans):
                continue
            n = int(m.group(1))
            if n == self_appendix:
                continue
            errors.append(
                f"{path}:{line_no}: 「{m.group(0)}」が未リンクです"
                f"（[{m.group(0)}](appendix.{n}.md) の形式にしてください）"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="第N回・Appendix N 参照のリンク化チェック")
    parser.add_argument("series_dir", help="対象ディレクトリ（site/series/）")
    args = parser.parse_args()

    series_dir = Path(args.series_dir)
    if not series_dir.exists():
        print(f"{series_dir} が見つかりません。", file=sys.stderr)
        return 1

    all_errors: list[str] = []
    for path in sorted(series_dir.glob("*.md")):
        all_errors.extend(check_file(path, series_dir))

    if all_errors:
        print("第N回・Appendix N リンク化チェックでエラーが見つかりました:", file=sys.stderr)
        for e in all_errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    print("第N回・Appendix N リンク化チェック: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
