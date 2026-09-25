from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from document_extractor.config import PaddleVLConfig
from document_extractor.engines.paddle_vl import PaddleVLEngine
from document_extractor.processing.mcq import merge_missing, retry_missing_options

PRIMARY = "Q1. Find $x^2$:\n(1) 10\n(2) 20"


def block(text=PRIMARY, bbox=(10, 10, 100, 50)):
    return SimpleNamespace(label="text", content=text, bbox=bbox, image=None)


def page(*blocks):
    pixels = np.zeros((200, 200, 3), dtype=np.uint8)
    pixels[:, :] = [11, 22, 33]  # BGR
    return {
        "page_index": 0,
        "parsing_res_list": list(blocks),
        "doc_preprocessor_res": {"output_img": pixels},
    }


def test_crop_uses_unmasked_pixels_and_nearest_same_column_boundary():
    question = block()
    result = page(
        question,
        block("Other column", (120, 55, 180, 80)),
        block("Q2. Next", (10, 100, 100, 120)),
    )
    paths = []

    def predict(**kwargs):
        paths.append(Path(kwargs.pop("input")))
        assert kwargs == {
            "use_layout_detection": False,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "prompt_label": "ocr",
        }
        with Image.open(paths[0]) as crop:
            assert crop.size == (90, 82)
            assert crop.getpixel((0, 70)) == (33, 22, 11)
        return [
            {
                "parsing_res_list": [
                    block("Q1. Changed stem:\n(1) 999\n(2) 888\n(3)110\n(4)90")
                ]
            }
        ]

    pipeline = Mock(predict=Mock(side_effect=predict))
    retry_missing_options(result, pipeline, 8)
    assert question.content == PRIMARY + "\n\n(3) 110\n\n(4) 90"
    pipeline.predict.assert_called_once()
    assert not paths[0].exists()


@pytest.mark.parametrize(
    "text,boundary",
    [
        (PRIMARY + "\n(3) 30\n(4) 40", 100),
        ("Ordinary text\n(1) one\n(2) two", 100),
        ("Q1. Math $ (1) 1 (2) 2 $", 100),
        ("Q1. Code `(1) 1 (2) 2`", 100),
        (PRIMARY + "\nQ2. Another question", 100),
        (PRIMARY, 55),
        (PRIMARY, None),
    ],
)
def test_safe_skips_make_no_extra_calls(text, boundary):
    blocks = [block(text)]
    if boundary is not None:
        blocks.append(block("Next", (10, boundary, 100, boundary + 20)))
    pipeline = Mock()
    retry_missing_options(page(*blocks), pipeline, 8)
    pipeline.predict.assert_not_called()
    assert blocks[0].content == text


@pytest.mark.parametrize(
    "retry",
    [
        "Q2. Wrong question\n(3) 30\n(4) 40",
        "(3) 30\n(3) 40",
        "(3) ",
        "No options found",
        "$(3) 30$",
    ],
)
def test_ambiguous_retry_preserves_primary(retry):
    assert merge_missing(PRIMARY, retry) == PRIMARY


def test_partial_retry_only_fills_holes():
    primary = PRIMARY + "\n(3) $x+1$"
    assert merge_missing(primary, "(3) wrong\n(4) $y+1$") == (primary + "\n\n(4) $y+1$")
    numbered = PRIMARY.replace("Q1.", "1.")
    assert merge_missing(numbered, "(3) 110") == numbered + "\n\n(3) 110"


def test_retry_failure_keeps_primary_and_logs(caplog):
    question = block()
    pipeline = Mock()
    pipeline.predict.side_effect = RuntimeError("server unavailable")
    retry_missing_options(
        page(question, block("Next", (10, 100, 100, 120))), pipeline, 8
    )
    assert question.content == PRIMARY
    assert "keeping primary text" in caplog.text


def test_retry_can_be_disabled():
    engine = PaddleVLEngine(replace(PaddleVLConfig(), mcq_retry=False))
    pipeline = Mock()
    pipeline.predict.return_value = [page(block(), block("Next", (10, 100, 100, 120)))]
    engine._pipeline = pipeline
    document = engine.extract(Path("sample.png"))
    assert document.pages[0].elements[0].content == PRIMARY
    pipeline.predict.assert_called_once()


def test_engine_debug_keeps_primary_snapshot_before_retry_and_reconstruction():
    question = block()

    class Result(dict):
        @property
        def json(self):
            return {"content": question.content}

    result = Result(page(question, block("Next", (10, 100, 100, 120))))
    engine = PaddleVLEngine(PaddleVLConfig(), debug=True)
    pipeline = Mock()
    pipeline.predict.side_effect = [
        [result],
        [{"parsing_res_list": [block("(3) 110\n(4) 90")]}],
    ]
    pipeline.restructure_pages.side_effect = lambda results, **kwargs: results
    engine._pipeline = pipeline
    document = engine.extract(Path("sample.pdf"))
    assert document.debug_results == [{"content": PRIMARY}]
    assert document.pages[0].elements[0].content == PRIMARY + "\n\n(3) 110\n\n(4) 90"
    assert pipeline.predict.call_count == 2
