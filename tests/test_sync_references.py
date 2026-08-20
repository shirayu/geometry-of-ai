"""sync_references.py のユニットテスト。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from sync_references import (  # noqa: E402
    BibEntry,
    citation_label,
    parse_bib,
    render_authors,
    render_entry,
    render_ref_section,
    render_short,
    year_label,
)


def _entry(entry_type: str = "article", **fields: str) -> BibEntry:
    return BibEntry(key="k", entry_type=entry_type, fields=fields)


def test_parse_bib_basic() -> None:
    text = """
@article{amari1998,
  author = {Amari, S.},
  title = {Natural Gradient Works Efficiently in Learning},
  year = {1998},
  journal = {Neural Computation},
}

@book{amari2016,
  author = {Amari, S.},
  title = {Information Geometry and Its Applications},
  year = {2016},
  publisher = {Springer Japan},
  doi = {10.1007/978-4-431-55978-8},
}
"""
    bib = parse_bib(text)
    assert set(bib) == {"amari1998", "amari2016"}
    assert bib["amari1998"].entry_type == "article"
    assert bib["amari2016"].fields["doi"] == "10.1007/978-4-431-55978-8"


def test_render_authors_counts() -> None:
    assert render_authors(["Amari, S."]) == "Amari, S."
    assert render_authors(["Kingma, D. P.", "Ba, J."]) == "Kingma, D. P., & Ba, J."
    assert (
        render_authors(["Fefferman, C.", "Mitter, S.", "Narayanan, H."]) == "Fefferman, C., Mitter, S., & Narayanan, H."
    )
    seven = [f"A{i}, X." for i in range(7)]
    assert render_authors(seven) == "A0, X., et al."


def test_citation_label_and_suffix() -> None:
    assert citation_label(_entry(author="Amari, S.", year="2016")) == "Amari, 2016"
    assert citation_label(_entry(author="Kingma, D. P. and Ba, J.", year="2015")) == "Kingma & Ba, 2015"
    assert citation_label(_entry(author="A, X. and B, Y. and C, Z.", year="2017")) == "A et al., 2017"
    assert citation_label(_entry(author="He, K. and Z, X.", year="2016", year_suffix="a")) == "He & Z, 2016a"
    # 日本語文献は citename を使う
    assert citation_label(_entry(author="渡辺澄夫", year="2012", citename="渡辺")) == "渡辺, 2012"


def test_year_label() -> None:
    assert year_label(_entry(author="A, X.", year="2016")) == "2016"
    assert year_label(_entry(author="A, X.", year="2016", year_suffix="b")) == "2016b"


def test_render_entry_article_doi() -> None:
    e = _entry(
        author="Amari, S.",
        title="Natural Gradient Works Efficiently in Learning",
        year="1998",
        journal="Neural Computation",
        volume="10",
        number="2",
        pages="251--276",
        doi="10.1162/089976698300017746",
    )
    assert render_entry(e) == (
        "Amari, S. (1998). Natural Gradient Works Efficiently in Learning. "
        "*Neural Computation*, 10(2), 251–276. "
        "DOI: [10.1162/089976698300017746](https://doi.org/10.1162/089976698300017746)"
    )


def test_render_entry_inproceedings_arxiv_eprintyear() -> None:
    e = _entry(
        entry_type="inproceedings",
        author="Kingma, D. P. and Ba, J.",
        title="Adam: A Method for Stochastic Optimization",
        year="2015",
        booktitle="ICLR 2015",
        eprint="1412.6980",
        eprintyear="2014",
    )
    assert render_entry(e) == (
        "Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. "
        "*ICLR 2015*. arXiv: [1412.6980](https://arxiv.org/abs/1412.6980) (2014)"
    )


def test_render_entry_book_series_edition() -> None:
    e = _entry(
        entry_type="book",
        author="Lee, J. M.",
        title="Introduction to Riemannian Manifolds",
        year="2018",
        edition="2nd",
        series="Graduate Texts in Mathematics",
        volume="176",
        publisher="Springer",
        doi="10.1007/978-3-319-91755-9",
    )
    assert render_entry(e) == (
        "Lee, J. M. (2018). *Introduction to Riemannian Manifolds* (2nd ed.). "
        "Graduate Texts in Mathematics, Vol. 176. Springer. "
        "DOI: [10.1007/978-3-319-91755-9](https://doi.org/10.1007/978-3-319-91755-9)"
    )


def test_render_entry_title_with_question_mark() -> None:
    e = _entry(
        entry_type="inproceedings",
        author="Beyer, K. and Goldstein, J.",
        title='When Is "Nearest Neighbor" Meaningful?',
        year="1999",
        booktitle="ICDT 1999",
        pages="217--235",
    )
    assert render_entry(e).startswith(
        'Beyer, K., & Goldstein, J. (1999). When Is "Nearest Neighbor" Meaningful? *ICDT 1999*'
    )


def test_render_short() -> None:
    e = _entry(
        entry_type="inproceedings",
        author="Vaswani, A. and Shazeer, N. and Parmar, N.",
        title="Attention Is All You Need",
        year="2017",
        booktitle="NeurIPS 2017",
    )
    assert render_short(e) == 'Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)'
    two = _entry(
        entry_type="inproceedings", author="Nickel, M. and Kiela, D.", title="T", year="2017", booktitle="NeurIPS 2017"
    )
    assert render_short(two).startswith("Nickel & Kiela, ")
    ja = _entry(
        entry_type="book", author="甘利俊一", title="『新版 情報幾何学の新展開』", year="2019", publisher="サイエンス社"
    )
    assert render_short(ja) == "甘利俊一, 『新版 情報幾何学の新展開』 (サイエンス社 2019)"
    arxiv = _entry(entry_type="misc", author="A, X. and B, Y. and C, Z.", title="T", year="2024", eprint="2401.00000")
    assert render_short(arxiv) == 'A et al., "T" (arXiv:2401.00000, 2024)'
    jmlr = _entry(
        author="F, W. and Z, B. and S, N.",
        title="T",
        year="2022",
        journal="Journal of Machine Learning Research",
        venueshort="JMLR",
    )
    assert render_short(jmlr) == 'F et al., "T" (JMLR 2022)'


def test_render_ref_section() -> None:
    bib = {
        "bellman1957": _entry(
            entry_type="book",
            author="Bellman, R.",
            title="Dynamic Programming",
            year="1957",
            publisher="Princeton University Press",
        )
    }
    data = {
        "categories": [
            {"title": "次元の呪い", "entries": [{"key": "bellman1957", "note": ["用語の起源。"]}]},
        ]
    }
    assert render_ref_section(data, bib) == (
        "## 参考文献\n\n### 次元の呪い\n\n"
        "- Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.\n"
        "    - 用語の起源。\n"
    )
