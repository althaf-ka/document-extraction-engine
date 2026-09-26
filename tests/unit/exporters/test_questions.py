import json
from pathlib import Path

import pytest

from document_extractor.engines.normalization import normalize_table
from document_extractor.exporters.document import DocumentExporter
from document_extractor.exporters.questions import (
    format_question_text,
    render_question_table,
)
from document_extractor.models import DocumentElement, NormalizedDocument, Page


def exam(body=None):
    return (
        "<table><tr><th>Q.No.</th><th>Questions</th><th>Marks</th></tr>"
        "<tr><td>1.</td><td>"
        + (
            body
            or r"Which of these is the correct value? (A) $ x^{2} $ (B) $ \frac{1}{x} $ (C) 2 (D) 3"
        )
        + "</td><td>1</td></tr>"
        "<tr><td>2.</td><td>Find the distance between the two given points.</td><td>2</td></tr></table>"
    )


def test_exam_renders_math_and_choices_without_changing_expressions():
    result = render_question_table(exam())
    assert result is not None
    assert "**1.** Which" in result.markdown
    assert "\n\n(A) $x^{2}$" in result.markdown
    assert "\n\n(B) $\\frac{1}{x}$" in result.markdown
    assert "Marks: 2" in result.markdown
    assert "<table>" not in result.markdown


def test_swallowed_label_repair_preserves_equations_and_reports_change():
    source = r"Which equation? (A) $(x+2)^{2}=2(x+3)(C)$ $x^{3}+1=(x+1)^{3}$ (B) $x^2=1$ (D) $x=2$"
    result = format_question_text(source)
    assert r"$(x+2)^{2}=2(x+3)$" in result.markdown
    assert "\n\n(C) $x^{3}+1=(x+1)^{3}$" in result.markdown
    assert result.repairs
    # Preserve extracted order rather than guessing spatial reading order.
    assert result.markdown.index("(C)") < result.markdown.index("(B)")
    second = format_question_text(result.markdown)
    assert second.markdown == result.markdown
    assert not second.repairs


@pytest.mark.parametrize(
    "text",
    [
        r"Value $f(A)$ with no choices.",
        r"(A) $(x+2)(C)$ $x+1$ (B) 2",
        r"(A) $f(C)$ $x+1$ (B) 2 (D) 3",
        r"(A) $f(x)(C)$ $x+1$ (B) 2 (D) 3",
        r"(A) $x$ (B) $y$ (C) $z$ (D) $w$",
    ],
)
def test_no_boundary_repair_without_strong_evidence(text):
    assert not format_question_text(text).repairs


def test_unbalanced_math_fails_closed():
    source = r"Choose: (A) $x (B) 2 (C) 3 (D) 4"
    result = format_question_text(source)
    assert result.markdown == source
    assert result.warnings


def test_literal_newlines_do_not_corrupt_latex_commands():
    result = format_question_text(
        r"Find $ \nu \neq \nabla f $.\n(A) 1\n(B) 2\n(C) 3\n(D) 4"
    )
    assert r"$\nu \neq \nabla f$" in result.markdown
    assert "\n\n(A)" in result.markdown


def test_assertion_mentions_are_not_choices():
    result = format_question_text(
        "Choose:\n(A) Assertion (A) and reason (R) are true.\n"
        "(B) Assertion (A) is false.\n(C) Both are false.\n(D) Neither."
    )
    assert "Assertion (A)" in result.markdown
    assert "\n\n(B)" in result.markdown
    assert not result.warnings


def test_multiple_accessible_choice_groups():
    result = format_question_text(
        "Choose (A) 1 (B) 2 (C) 3 (D) 4\nAlternative: (A) 5 (B) 6 (C) 7 (D) 8"
    )
    assert result.markdown.count("\n\n(A)") == 2


def test_code_and_image_attributes_are_not_reformatted():
    image = '<img src="figure.png" alt="(A) $ x $ (B) $ y $ (C) z (D) w">'
    text = image + "\n`$ x $`\n(A) 1 (B) 2 (C) 3 (D) 4"
    result = format_question_text(text)
    assert image in result.markdown
    assert "`$ x $`" in result.markdown
    assert "\n\n(A) 1" in result.markdown


def test_repeated_internal_alternatives_are_not_reported_as_mcqs():
    text = "(iii) (A) Find the sum. OR (B) Find the mean. Alternative: (A) Find the sum. OR (B) Find the mean."
    result = format_question_text(text)
    assert result.markdown == text
    assert not result.warnings


@pytest.mark.parametrize(
    "html",
    [
        "<table><tr><th>Item</th><th>Description</th><th>Marks</th></tr><tr><td>1.</td><td>A sufficiently long description of an ordinary item</td><td>4</td></tr></table>",
        "<table><tr><td>0-15</td><td>15-30</td></tr><tr><td>10</td><td>7</td></tr></table>",
        exam().replace("2.</td>", "1.</td>"),
        exam().replace("</td><td>1</td>", "<td>1</td>"),
        exam("<table><tr><td>Frequency</td><td>10</td></tr></table>"),
        exam().replace("</table>", ""),
    ],
)
def test_ordinary_nested_and_ambiguous_tables_are_preserved(html):
    assert render_question_table(html) is None


def test_rowspan_continuation_retains_question_and_marks():
    html = (
        exam()
        .replace("<td>1.</td>", '<td rowspan="2">1.</td>')
        .replace(
            "</td><td>1</td></tr>",
            '</td><td rowspan="2">1</td></tr><tr><td>Accessible alternative text.</td></tr>',
        )
    )
    result = render_question_table(html)
    assert result is not None
    assert "Accessible alternative text." in result.markdown
    assert result.markdown.count("Marks: 1") == 1


def test_automatic_export_keeps_original_and_images_with_opt_out(tmp_path):
    html = exam(
        'Which of these describes the figure shown? <img src="figure.png"> (A) 1 (B) 2 (C) 3 (D) 4'
    )
    document = NormalizedDocument(
        [Page(1, [DocumentElement("table", html, table=normalize_table(html))])],
        "sample.pdf",
        image_assets={"figure.png": b"pixels"},
    )
    default = DocumentExporter(question_layout=False).export(
        document, Path("sample.pdf"), tmp_path / "default"
    )
    formatted = DocumentExporter(export_layout_report=True).export(
        document, Path("sample.pdf"), tmp_path / "formatted"
    )
    assert "<table>" in default.document_path.read_text()
    assert not (default.root_dir / "question-layout.json").exists()
    markdown = formatted.document_path.read_text()
    assert "<table>" not in markdown
    assert '<img src="images/img-0.png">' in markdown
    assert (formatted.images_dir / "img-0.png").read_bytes() == b"pixels"
    assert (formatted.tables_dir / "tbl-0.html").read_text() == html.replace(
        "figure.png", "../images/img-0.png"
    )
    assert document.pages[0].elements[0].content == html
    report = json.loads(
        (formatted.root_dir / "debug" / "question-layout.json").read_text()
    )
    assert report["tables"][0]["converted"] is True


@pytest.mark.parametrize("unit", ["cm", "m", "mm", "kg", "cm²"])
def test_fraction_followed_by_unit_has_markdown_boundary(unit):
    result = format_question_text(r"(A) $\frac{5}{6}$" + unit)
    assert result.markdown == r"(A) $\frac{5}{6}$ " + unit
    assert format_question_text(result.markdown).markdown == result.markdown


@pytest.mark.parametrize(
    "text",
    [
        r"(A) $\frac{5}{36}$",
        r"(A) $\frac{5}{6}$ cm",
        r"`$\frac{5}{6}$cm`",
        r"$\frac{5}{6}\mathrm{cm}$",
        '<img src="x" alt="$x$cm">',
    ],
)
def test_unit_spacing_preserves_already_valid_math_and_protected_text(text):
    assert format_question_text(text).markdown == text


def test_math_comparison_cannot_hide_later_image_during_export(tmp_path):
    image_source = "paddle-page-3/imgs/crop.jpg"
    html = exam(r"Find the measure of the angles with $0&lt;A,B&lt;90$.").replace(
        "Find the distance between the two given points.",
        f'Find the distance between the two given points. <img src="{image_source}">',
    )
    data = "<table><tr><th>Interval</th><th>Frequency</th></tr><tr><td>0-15</td><td>10</td></tr></table>"
    document = NormalizedDocument(
        [
            Page(
                1,
                [
                    DocumentElement("table", html, table=normalize_table(html)),
                    DocumentElement("table", data, table=normalize_table(data)),
                ],
            )
        ],
        "sample.pdf",
        image_assets={image_source: b"pixels"},
    )
    bundle = DocumentExporter(export_layout_report=True).export(
        document, Path("sample.pdf"), tmp_path
    )
    markdown = bundle.document_path.read_text()
    assert "$0<A,B<90$" in markdown
    assert '<img src="images/img-0.png">' in markdown
    assert "paddle-page-" not in markdown
    assert (bundle.images_dir / "img-0.png").read_bytes() == b"pixels"
    assert (
        '<img src="../images/img-0.png">'
        in (bundle.tables_dir / "tbl-0.html").read_text()
    )
    assert "| Interval | Frequency |" in markdown
    assert "| 0-15 | 10 |" in markdown
    assert "[Table 1](tables/tbl-0.html)" in markdown
    assert "[Table 2](tables/tbl-1.html)" in markdown
    report = json.loads(
        (bundle.root_dir / "debug" / "question-layout.json").read_text()
    )
    assert [table["converted"] for table in report["tables"]] == [True, False]
