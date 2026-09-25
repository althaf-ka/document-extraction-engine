from pathlib import Path
from types import SimpleNamespace

import pytest

from document_extractor.engines.paddle_vl import adapt_results
from document_extractor.exporters.document import DocumentExporter
from document_extractor.models import FormulaPlacement


def block(label, content, bbox=(0, 0, 10, 10)):
    return SimpleNamespace(label=label, content=content, bbox=bbox, image=None)


def export(tmp_path, blocks):
    document = adapt_results([{"parsing_res_list": blocks}], Path("sample.pdf"))
    bundle = DocumentExporter().export(document, Path("sample.pdf"), tmp_path)
    return document, bundle, bundle.document_path.read_text()


def test_rectangular_table_and_exact_artifact(tmp_path):
    html = " <table><tr><th>Name</th><th>Score</th></tr><tr><td>Alice</td><td>92</td></tr><tr><td>Bob</td><td>87</td></tr></table>\n"
    document, bundle, markdown = export(tmp_path, [block("table", html)])
    assert markdown == "| Name | Score |\n| --- | --- |\n| Alice | 92 |\n| Bob | 87 |\n"
    assert (bundle.tables_dir / "tbl-0.html").read_text() == html
    table = document.pages[0].elements[0].table
    assert table is not None
    assert table.has_merged_cells is False


@pytest.mark.parametrize("span", ['rowspan="2"', 'colspan="2"'])
def test_spanning_table_embedded_with_artifact(tmp_path, span):
    html = f"<table><tr><th {span}>Subject</th><th>Marks</th></tr><tr><td>92</td></tr></table>"
    document, bundle, markdown = export(
        tmp_path,
        [block("text", "Before"), block("table", html), block("text", "After")],
    )
    assert markdown == f"Before\n\n{html}\n\nAfter\n"
    assert (bundle.tables_dir / "tbl-0.html").read_text() == html
    table = document.pages[0].elements[1].table
    assert table is not None
    assert table.has_merged_cells is True


def test_headerless_table_keeps_all_data_and_escapes_cells(tmp_path):
    html = "<table><tr><td>A|B</td><td>&lt;x&gt; &amp; _y_</td></tr><tr><td>C</td><td>2</td></tr></table>"
    _, _, markdown = export(tmp_path, [block("table", html)])
    assert (
        markdown
        == "|  |  |\n| --- | --- |\n| A\\|B | &lt;x&gt; &amp; \\_y\\_ |\n| C | 2 |\n"
    )


@pytest.mark.parametrize(
    "html",
    [
        "<table><tr><td><b>Bold</b></td></tr></table>",
        "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>",
        "<table><caption>Results</caption><tr><td>92</td></tr></table>",
    ],
)
def test_complex_content_preserved(tmp_path, html):
    _, _, markdown = export(tmp_path, [block("table", html)])
    assert markdown == html + "\n"


@pytest.mark.parametrize("latex", [r"\frac{x}{2}", r"\frac{\pi}{2}"])
@pytest.mark.parametrize("delimiters", [("", ""), ("$", "$"), (r"\(", r"\)")])
def test_inline_formula_normalized_once_and_joined(tmp_path, latex, delimiters):
    opening, closing = delimiters
    document, _, markdown = export(
        tmp_path,
        [
            block("text", "The interval is"),
            block("inline_formula", opening + latex + closing),
            block("text", ". Next sentence."),
        ],
    )
    formula = document.pages[0].elements[1]
    assert formula.content == latex
    assert formula.formula_placement == FormulaPlacement.INLINE
    assert markdown == f"The interval is ${latex}$. Next sentence.\n"


@pytest.mark.parametrize(
    "latex",
    [
        r"\frac{x}{2}",
        r"\begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix}",
        r"\begin{vmatrix}1 & 2 \\ 3 & 4\end{vmatrix}",
    ],
)
@pytest.mark.parametrize(
    "delimiters", [("", ""), ("$", "$"), ("$$", "$$"), (r"\[", r"\]")]
)
def test_display_formula_normalized_once(tmp_path, latex, delimiters):
    opening, closing = delimiters
    document, _, markdown = export(
        tmp_path,
        [
            block("text", "Equation:"),
            block("display_formula", opening + latex + closing),
        ],
    )
    assert document.pages[0].elements[1].content == latex
    assert markdown == f"Equation:\n\n$$\n{latex}\n$$\n"


@pytest.mark.parametrize(
    "bbox,placement",
    [
        ((25, 0, 40, 10), FormulaPlacement.INLINE),
        ((0, 20, 40, 40), FormulaPlacement.DISPLAY),
        ((100, 0, 120, 10), FormulaPlacement.DISPLAY),
    ],
)
def test_generic_formula_uses_layout_not_delimiters(tmp_path, bbox, placement):
    document, _, _ = export(
        tmp_path,
        [block("text", "Value", (0, 0, 20, 10)), block("formula", "$x$", bbox)],
    )
    assert document.pages[0].elements[1].formula_placement == placement


def test_existing_math_in_text_is_unchanged(tmp_path):
    content = r"The interval is $\left[0,\frac{\pi}{2}\right)$."
    _, _, markdown = export(tmp_path, [block("text", content)])
    assert markdown == content + "\n"
