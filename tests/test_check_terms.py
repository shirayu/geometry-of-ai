"""check_terms.py のユニットテスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from check_terms import Rule, Violation, check_file, load_rules  # noqa: E402

RULES_FILE = Path(__file__).parent.parent / "style" / "terms.yaml"


@pytest.fixture
def rules() -> list[Rule]:
    return load_rules(RULES_FILE)


def _md(tmp_path: Path, content: str) -> Path:
    """一時 .md ファイルを作成して返す。"""
    p = tmp_path / "test.md"
    p.write_text(content, encoding="utf-8")
    return p


def violation_ids(vs: list[Violation]) -> list[str]:
    return [v.rule_id for v in vs]


# ---------------------------------------------------------------------------
# sphere-notation
# ---------------------------------------------------------------------------


class TestSphereNotation:
    def test_ok_mathbb(self, tmp_path, rules):
        p = _md(tmp_path, "球面 $\\mathbb{S}^{d-1}$ 上の点。\n")
        assert "sphere-notation" not in violation_ids(check_file(p, rules))

    def test_violation_plain(self, tmp_path, rules):
        p = _md(tmp_path, "球面 $S^{d-1}$ 上の点。\n")
        assert "sphere-notation" in violation_ids(check_file(p, rules))

    def test_violation_S1(self, tmp_path, rules):
        p = _md(tmp_path, "円周 $S^{1}$ はループ。\n")
        assert "sphere-notation" in violation_ids(check_file(p, rules))

    def test_ok_in_code_fence(self, tmp_path, rules):
        p = _md(tmp_path, "```python\nS^{d-1}\n```\n")
        assert "sphere-notation" not in violation_ids(check_file(p, rules))


# ---------------------------------------------------------------------------
# no-vmf-lowercase
# ---------------------------------------------------------------------------


class TestNoVmfLowercase:
    def test_ok_uppercase(self, tmp_path, rules):
        p = _md(tmp_path, "vMF分布を使う。\n")
        assert "no-vmf-lowercase" not in violation_ids(check_file(p, rules))

    def test_violation(self, tmp_path, rules):
        p = _md(tmp_path, "vmf分布を使う。\n")
        assert "no-vmf-lowercase" in violation_ids(check_file(p, rules))

    def test_ok_in_code_fence(self, tmp_path, rules):
        p = _md(tmp_path, "```python\nvmf_layer = ...\n```\n")
        assert "no-vmf-lowercase" not in violation_ids(check_file(p, rules))

    def test_ok_partial_word(self, tmp_path, rules):
        # "vmf" が単語境界を持たない場合はマッチしない（\b のテスト）
        p = _md(tmp_path, "pvmfq という文字列。\n")
        assert "no-vmf-lowercase" not in violation_ids(check_file(p, rules))


# ---------------------------------------------------------------------------
# no-boldsymbol-latin
# ---------------------------------------------------------------------------


class TestNoBoldsymbolLatin:
    def test_ok_greek(self, tmp_path, rules):
        p = _md(tmp_path, "$\\boldsymbol{\\mu}$ は平均方向。\n")
        assert "no-boldsymbol-latin" not in violation_ids(check_file(p, rules))

    def test_violation_lowercase(self, tmp_path, rules):
        p = _md(tmp_path, "$\\boldsymbol{h} \\in \\mathbb{S}^{d-1}$\n")
        assert "no-boldsymbol-latin" in violation_ids(check_file(p, rules))

    def test_violation_x(self, tmp_path, rules):
        p = _md(tmp_path, "$\\boldsymbol{x}$ はベクトル。\n")
        assert "no-boldsymbol-latin" in violation_ids(check_file(p, rules))

    def test_ok_in_code_fence(self, tmp_path, rules):
        p = _md(tmp_path, "```python\n# \\boldsymbol{x}\n```\n")
        assert "no-boldsymbol-latin" not in violation_ids(check_file(p, rules))


# ---------------------------------------------------------------------------
# no-chokyu-men
# ---------------------------------------------------------------------------


class TestNoChokyuMen:
    def test_ok_normal(self, tmp_path, rules):
        p = _md(tmp_path, "高次元の球面上で学習する。\n")
        assert "no-chokyu-men" not in violation_ids(check_file(p, rules))

    def test_violation(self, tmp_path, rules):
        p = _md(tmp_path, "高次元の超球面（hypersphere）に射影する。\n")
        assert "no-chokyu-men" in violation_ids(check_file(p, rules))

    def test_ok_in_allowed_section(self, tmp_path, rules):
        content = "## 球面と超球面の混同\n\n超球面という用語について解説する。\n"
        p = _md(tmp_path, content)
        assert "no-chokyu-men" not in violation_ids(check_file(p, rules))

    def test_violation_outside_allowed_section(self, tmp_path, rules):
        # 許可セクションの後に別セクションが始まったら再び違反
        content = "## 球面と超球面の混同\n\n超球面の説明。\n\n## 別のセクション\n\n超球面とも呼ばれる。\n"
        p = _md(tmp_path, content)
        vs = check_file(p, rules)
        chokyu = [v for v in vs if v.rule_id == "no-chokyu-men"]
        assert len(chokyu) == 1
        assert chokyu[0].line_no == 7

    def test_ok_in_code_fence(self, tmp_path, rules):
        p = _md(tmp_path, "```\n超球面\n```\n")
        assert "no-chokyu-men" not in violation_ids(check_file(p, rules))


# ---------------------------------------------------------------------------
# no-simplex-katakana
# ---------------------------------------------------------------------------


class TestNoSimplexKatakana:
    def test_ok_first_mention(self, tmp_path, rules):
        p = _md(tmp_path, "確率単体（シンプレックス、probability simplex）を定義する。\n")
        assert "no-simplex-katakana" not in violation_ids(check_file(p, rules))

    def test_violation_standalone(self, tmp_path, rules):
        p = _md(tmp_path, "シンプレックス内部の点。\n")
        assert "no-simplex-katakana" in violation_ids(check_file(p, rules))

    def test_violation_in_heading(self, tmp_path, rules):
        p = _md(tmp_path, "### シンプレックス：確率分布の空間\n")
        assert "no-simplex-katakana" in violation_ids(check_file(p, rules))

    def test_ok_in_code_fence(self, tmp_path, rules):
        p = _md(tmp_path, "```\nシンプレックス\n```\n")
        assert "no-simplex-katakana" not in violation_ids(check_file(p, rules))


# ---------------------------------------------------------------------------
# vmf-parenthesis
# ---------------------------------------------------------------------------


class TestVmfParenthesis:
    def test_ok_fullwidth(self, tmp_path, rules):
        p = _md(tmp_path, "von Mises-Fisher（vMF）分布を使う。\n")
        assert "vmf-parenthesis" not in violation_ids(check_file(p, rules))

    def test_violation_halfwidth(self, tmp_path, rules):
        p = _md(tmp_path, "von Mises-Fisher (vMF) 分布を使う。\n")
        assert "vmf-parenthesis" in violation_ids(check_file(p, rules))

    def test_ok_in_code_fence(self, tmp_path, rules):
        # Pythonコード内の ax.set_title("von Mises-Fisher Distribution (2D)") 等
        p = _md(tmp_path, '```python\nax.set_title("von Mises-Fisher Distribution (2D)")\n```\n')
        assert "vmf-parenthesis" not in violation_ids(check_file(p, rules))

    def test_ok_in_reference(self, tmp_path, rules):
        # 参考文献中の著者名括弧は "von Mises-Fisher Distributions" 等で ( が続かない
        p = _md(tmp_path, "- von Mises-Fisher Distributions. *JMLR*, 2005.\n")
        assert "vmf-parenthesis" not in violation_ids(check_file(p, rules))


# ---------------------------------------------------------------------------
# 複数ルール・複合ケース
# ---------------------------------------------------------------------------


class TestMultipleRules:
    def test_clean_file(self, tmp_path, rules):
        content = (
            "# 第3回\n\n"
            "von Mises-Fisher（vMF）分布は $\\mathbb{S}^{d-1}$ 上の分布。\n"
            "集中度 $\\boldsymbol{\\mu}$ は平均方向。\n"
            "確率単体（シンプレックス、probability simplex）を定義する。\n"
        )
        p = _md(tmp_path, content)
        assert check_file(p, rules) == []

    def test_multiple_violations_in_one_file(self, tmp_path, rules):
        content = (
            "球面 $S^{d-1}$ 上の点。\n"  # sphere-notation
            "vmf分布を使う。\n"  # no-vmf-lowercase
            "$\\boldsymbol{x}$ はベクトル。\n"  # no-boldsymbol-latin
        )
        p = _md(tmp_path, content)
        ids = violation_ids(check_file(p, rules))
        assert "sphere-notation" in ids
        assert "no-vmf-lowercase" in ids
        assert "no-boldsymbol-latin" in ids

    def test_excluded_filenames(self, tmp_path, rules):
        """intro.md 等は対象外だが、check_file は直接呼ばれると普通にチェックされる。
        除外はイテレータ側の責務なので、ここでは check_file が結果を返すことを確認。"""
        p = tmp_path / "intro.md"
        p.write_text("球面 $S^{d-1}$ 上の点。\n", encoding="utf-8")
        # check_file 自体は除外しない
        assert "sphere-notation" in violation_ids(check_file(p, rules))
