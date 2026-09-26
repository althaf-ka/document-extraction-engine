from pathlib import Path

import pytest

from document_extractor.engines.normalization import normalize_table
from document_extractor.exporters.document import DocumentExporter, render_table
from document_extractor.exporters.math import normalize_inline_math, render_math_cell
from document_extractor.models import (
    AnswerOption,
    DocumentElement,
    NormalizedDocument,
    OptionGroup,
    Page,
)

Q49 = r"Q49. A container of fixed volume contains a gas at  $ 27^{\circ} $C. To double the pressure of the gas, the temperature of gas should be raised to ___  $ ^{\circ} $C."


def test_plain_question_math_gets_valid_delimiters_without_exam_table(tmp_path):
    document = NormalizedDocument([Page(1, [DocumentElement("text", Q49)])], "jee.pdf")
    bundle = DocumentExporter(question_layout=False).export(
        document, Path("jee.pdf"), tmp_path
    )
    markdown = bundle.document_path.read_text()
    assert r"$27^{\circ}$ C." in markdown
    assert r"___  $^{\circ}$ C." in markdown
    assert document.pages[0].elements[0].content == Q49
    assert normalize_inline_math(markdown) == markdown


@pytest.mark.parametrize(
    "text",
    [
        r"`$ x $C`",
        r"```latex" + "\n$ x $C\n```",
        r'<img alt="$ x $C" src="crop.png">',
        r"Prices are $5 and $10.",
        r"Pay \$5, then \$10.",
        r"Unclosed $x and text.",
        "$$\nx^2\n$$",
        r"\(x\) and \[y\]",
        r"The interval is $\left[0,\frac{\pi}{2}\right)$.",
    ],
)
def test_nonmath_and_protected_content_is_preserved(text):
    assert normalize_inline_math(text) == text


def test_spacing_at_both_boundaries_and_multiple_formulas():
    assert (
        normalize_inline_math(r"Temperature$ 27^{\circ} $C; $ x $and$ y $.")
        == r"Temperature $27^{\circ}$ C; $x$ and $y$."
    )


def test_matching_table_keeps_latex_and_remains_a_table(tmp_path):
    html = (
        "<table><tr><td>Q51.</td><td>List - I</td><td>List - II</td></tr>"
        r"<tr><td>(A)</td><td>$ [\mathrm{MnBr}_{4}]^{2-} $</td><td>$ d^{2}s^{3} $ &amp; diamagnetic</td></tr>"
        r"<tr><td>(B)</td><td>$ [\mathrm{FeF}_{6}]^{3-} $</td><td>$ sp^{3} $ &amp; paramagnetic</td></tr></table>"
    )
    document = NormalizedDocument(
        [Page(1, [DocumentElement("table", html, table=normalize_table(html))])],
        "jee.pdf",
    )
    bundle = DocumentExporter().export(document, Path("jee.pdf"), tmp_path)
    markdown = bundle.document_path.read_text()
    assert (
        r"| (A) | $[\mathrm{MnBr}_{4}]^{2-}$ | $d^{2}s^{3}$ &amp; diamagnetic |"
        in markdown
    )
    assert (
        r"| (B) | $[\mathrm{FeF}_{6}]^{3-}$ | $sp^{3}$ &amp; paramagnetic |" in markdown
    )
    assert "[Table 1](tables/tbl-0.html)" in markdown
    assert (bundle.tables_dir / "tbl-0.html").read_text() == html
    # Rendering must not invent missing list labels or missing orbital terms.
    assert "(II)" not in markdown
    assert "$sp^{3}d^{2}$" not in markdown


def test_table_prose_is_escaped_but_chemical_subscripts_are_not():
    assert (
        render_math_cell(
            r"_literal_ $ [\mathrm{Co}(\mathrm{C}_{2}\mathrm{O}_{4})_{3}]^{3-} $ A|B"
        )
        == r"\_literal\_ $[\mathrm{Co}(\mathrm{C}_{2}\mathrm{O}_{4})_{3}]^{3-}$ A\|B"
    )
    assert render_math_cell("Price $5") == r"Price \$5"


def test_formula_with_pipe_preserves_original_table_html():
    html = r"<table><tr><td>$|x|$</td><td>Absolute value</td></tr></table>"
    assert render_table(normalize_table(html)) == html


def test_recognized_option_groups_also_normalize_math(tmp_path):
    group = OptionGroup(
        "Q1. Choose:",
        (
            AnswerOption("(1)", r"$ 27^{\circ} $C"),
            AnswerOption("(2)", r"$ 10^{\circ} $C"),
        ),
    )
    document = NormalizedDocument(
        [Page(1, [DocumentElement("text", "original", option_group=group)])], "jee.pdf"
    )
    bundle = DocumentExporter().export(document, Path("jee.pdf"), tmp_path)
    assert r"(1) $27^{\circ}$ C" in bundle.document_path.read_text()
    assert document.pages[0].elements[0].option_group == group
