#!/usr/bin/env python3
"""用語・記法 lint: style/terms.yaml のルールに従い site/series/*.md を検証する。"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

RULES_FILE = Path(__file__).parent.parent / "style" / "terms.yaml"

# 対象外ファイル（内容チェック不要）
EXCLUDED_FILES = {"intro.md", "toc.md", "references.md", "keywords.md"}


@dataclass
class Rule:
    id: str
    description: str
    pattern: re.Pattern[str]
    scope: str  # "text" | "math" | "any"
    require_absent: re.Pattern[str] | None = None
    allow_line_pattern: re.Pattern[str] | None = None
    allow_section: str | None = None


def load_rules(rules_file: Path) -> list[Rule]:
    data = yaml.safe_load(rules_file.read_text(encoding="utf-8"))
    rules = []
    for r in data["rules"]:
        require_absent = r.get("require_absent")
        allow_line = r.get("allow_line_pattern")
        rules.append(
            Rule(
                id=r["id"],
                description=r["description"],
                pattern=re.compile(r["pattern"]),
                scope=r.get("scope", "any"),
                require_absent=re.compile(require_absent) if require_absent else None,
                allow_line_pattern=re.compile(allow_line) if allow_line else None,
                allow_section=r.get("allow_section"),
            )
        )
    return rules


# ---------------------------------------------------------------------------
# Markdown パーサーユーティリティ
# ---------------------------------------------------------------------------

FENCE_RE = re.compile(r"^(`{3,}|~{3,})")


def _fence_marker(line: str) -> str | None:
    """コードフェンス行のマーカー（``` or ~~~）を返す。非フェンス行は None。"""
    m = FENCE_RE.match(line.lstrip())
    return m.group(1)[:3] if m else None


def _heading_text(line: str) -> str | None:
    """## 見出し行のテキスト部分を返す。非見出し行は None。"""
    m = re.match(r"^#{1,6}\s+(.*)", line)
    return m.group(1).strip() if m else None


def _math_ranges(line: str) -> list[tuple[int, int]]:
    """行内の数式範囲 $...$ / $$...$$ の (start, end) リストを返す（インデックスは行内）。
    コードスパン内は除外する。"""
    ranges: list[tuple[int, int]] = []
    i = 0
    in_code = False
    code_ticks = 0

    while i < len(line):
        ch = line[i]

        # コードスパンの開閉
        if ch == "`":
            n = 0
            while i + n < len(line) and line[i + n] == "`":
                n += 1
            if not in_code:
                in_code = True
                code_ticks = n
            elif n == code_ticks:
                in_code = False
                code_ticks = 0
            i += n
            continue

        if in_code:
            i += 1
            continue

        # エスケープ
        if ch == "\\" and i + 1 < len(line):
            i += 2
            continue

        if ch == "$":
            # $$...$$
            if i + 1 < len(line) and line[i + 1] == "$":
                end = line.find("$$", i + 2)
                if end != -1:
                    ranges.append((i, end + 2))
                    i = end + 2
                else:
                    i += 2
                continue
            # $...$
            j = i + 1
            while j < len(line):
                if line[j] == "\\" and j + 1 < len(line):
                    j += 2
                    continue
                if line[j] == "$":
                    if j + 1 < len(line) and line[j + 1] == "$":
                        j += 1
                        continue
                    ranges.append((i, j + 1))
                    i = j + 1
                    break
                j += 1
            else:
                i += 1
            continue

        i += 1

    return ranges


def _text_outside_math(line: str) -> str:
    """数式範囲を空白に置換した行テキストを返す（scope="text" 用）。"""
    math_rs = _math_ranges(line)
    if not math_rs:
        return line
    result = list(line)
    for start, end in math_rs:
        for k in range(start, min(end, len(result))):
            result[k] = " "
    return "".join(result)


def _text_only_math(line: str) -> str:
    """数式範囲のみを連結した文字列を返す（scope="math" 用）。"""
    math_rs = _math_ranges(line)
    return "".join(line[s:e] for s, e in math_rs)


# ---------------------------------------------------------------------------
# ファイルチェック
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    path: Path
    line_no: int
    rule_id: str
    description: str
    line_text: str


def check_file(path: Path, rules: list[Rule]) -> list[Violation]:
    violations: list[Violation] = []
    lines = path.read_text(encoding="utf-8").splitlines()

    in_fence = False
    fence_marker = ""
    current_section: str = ""

    for line_no, line in enumerate(lines, start=1):
        # --- コードフェンスの追跡 ---
        marker = _fence_marker(line)
        if marker:
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            continue  # フェンス行自体はチェック対象外

        # --- セクション見出しの追跡 ---
        heading = _heading_text(line)
        if heading is not None:
            current_section = heading

        # コードフェンス内はすべてスキップ
        if in_fence:
            continue

        # --- ルールごとにチェック ---
        for rule in rules:
            # scope に応じてチェック対象テキストを決定
            if rule.scope == "text":
                target = _text_outside_math(line)
            elif rule.scope == "math":
                target = _text_only_math(line)
            else:  # "any"
                target = line

            if not rule.pattern.search(target):
                continue

            # require_absent: 同一行にパターンがあれば除外
            if rule.require_absent and rule.require_absent.search(line):
                continue

            # allow_line_pattern: 同一行にパターンがあれば除外
            if rule.allow_line_pattern and rule.allow_line_pattern.search(line):
                continue

            # allow_section: 現在セクションが対象なら除外
            if rule.allow_section and rule.allow_section in current_section:
                continue

            violations.append(
                Violation(
                    path=path,
                    line_no=line_no,
                    rule_id=rule.id,
                    description=rule.description,
                    line_text=line.rstrip(),
                )
            )

    return violations


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------


def iter_markdown_files(paths: list[str]) -> list[Path]:
    files: set[Path] = set()
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for f in p.rglob("*.md"):
                if f.name not in EXCLUDED_FILES:
                    files.add(f)
        elif p.is_file() and p.suffix == ".md":
            if p.name not in EXCLUDED_FILES:
                files.add(p)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="用語・記法 lint（style/terms.yaml のルールを適用）")
    parser.add_argument("paths", nargs="+", help="チェック対象のファイルまたはディレクトリ")
    parser.add_argument("--rules", default=str(RULES_FILE), help="ルール定義 YAML ファイル")
    args = parser.parse_args()

    rules_path = Path(args.rules)
    if not rules_path.exists():
        print(f"ルールファイルが見つかりません: {rules_path}", file=sys.stderr)
        return 2

    rules = load_rules(rules_path)
    files = iter_markdown_files(args.paths)

    if not files:
        print("Markdown ファイルが見つかりません。")
        return 0

    all_violations: list[Violation] = []
    for path in files:
        all_violations.extend(check_file(path, rules))

    if all_violations:
        for v in all_violations:
            print(f"{v.path}:{v.line_no}: [{v.rule_id}] {v.description}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
