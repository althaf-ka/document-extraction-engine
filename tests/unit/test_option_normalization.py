from unittest.mock import Mock

import pytest

from document_extractor.models import DocumentElement, NormalizedDocument, Page
from document_extractor.option_normalization import normalize_options, recognize_options
from document_extractor.pipeline import ExtractionPipeline


@pytest.mark.parametrize(
    "answers",
    [
        "(1) 90 (2) 84\n(3) 122 (4) 108",
        "(1) 90\n(2) 84\n(3) 122\n(4) 108",
        "(1) 90\n\n(2) 84\n\n(3) 122\n\n(4) 108",
    ],
)
def test_pipeline_renders_vertical_options(tmp_path, answers):
    source = tmp_path / "sample.pdf"
    source.touch()
    content = "Q2. Its $11^{th}$ term is:\n" + answers
    document = NormalizedDocument(
        [Page(1, [DocumentElement("text", content, order=7)])], str(source)
    )
    engine = Mock()
    engine.extract.return_value = document
    bundle = ExtractionPipeline(engine=engine).run(source, tmp_path / "output")
    assert (
        bundle.document_path.read_text()
        == "Q2. Its $11^{th}$ term is:\n\n(1) 90\n\n(2) 84\n\n(3) 122\n\n(4) 108\n"
    )
    assert document.pages[0].elements[0].content == content
    assert document.pages[0].elements[0].order == 7


@pytest.mark.parametrize(
    "math",
    [
        r"$f(2) + (3) x$",
        r"$$ (2) x $$",
        r"\( (2) x \)",
        r"\[ (2) x \]",
        r"`(2) example`",
    ],
)
def test_math_and_code_are_not_option_labels(math):
    group = recognize_options(f"Q1. Choose:\n(1) {math} (2) Other")
    assert group is not None
    assert group.options[0].content == math
    assert len(group.options) == 2


def test_incomplete_group_is_not_filled_in():
    group = recognize_options("Q1. Value: (1) 100 (2) 120")
    assert group is not None
    assert [option.label for option in group.options] == ["(1)", "(2)"]


@pytest.mark.parametrize(
    "text",
    [
        "Ordinary prose (1) one (2) two",
        "Q1. Conditions (1) first and (2) second imply what?",
        "Q1. Value: (1) one (1) repeated",
        "Q1. Value: (1) one (3) uncertain",
        "Q1. Value: (1) $unclosed (2) two",
        "Q1. Value: (1) one",
        "Q1. Value: (1) one (2) ",
        "Q1. Value: (1) one (2) two\nQ2. Next question",
        "Q1. <table><tr><td>(1) one (2) two</td></tr></table>",
    ],
)
def test_ambiguous_content_is_unchanged(text):
    assert recognize_options(text) is None


def test_separate_answer_block_and_idempotence():
    document = NormalizedDocument(
        [
            Page(
                1,
                [
                    DocumentElement("text", "Q2. Choose:"),
                    DocumentElement("text", "(1) A (2) B"),
                ],
            )
        ],
        "sample.pdf",
    )
    normalize_options(document)
    original = list(document.pages[0].elements)
    assert original[1].option_group is not None
    normalize_options(document)
    assert document.pages[0].elements == original


def test_preserves_extracted_option_order():
    group = recognize_options("Q1. Choose: (1) A (3) C (2) B (4) D")
    assert group is not None
    assert [option.label for option in group.options] == ["(1)", "(3)", "(2)", "(4)"]
