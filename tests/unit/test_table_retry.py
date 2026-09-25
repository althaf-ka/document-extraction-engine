from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image

from document_extractor.processing.tables import (
    complete_match_list,
    incomplete_match_list,
    retry_incomplete_tables,
)

PRIMARY = """<table><tr><td>Q51.</td><td>List - I</td><td>List - II</td></tr><tr><td>(A)</td><td>A</td><td>first</td></tr><tr><td>(B)</td><td>B</td><td>second</td></tr><tr><td>(C)</td><td>C</td><td>(III) third</td></tr><tr><td>(D)</td><td>D</td><td>(IV) fourth</td></tr></table>"""
RETRY = """<table><tr><td>Q51.</td><td>List - I</td><td>List - II</td></tr><tr><td>(A)</td><td>A</td><td>(I) first</td></tr><tr><td>(B)</td><td>B</td><td>(II) second</td></tr><tr><td>(C)</td><td>C</td><td>(III) third</td></tr><tr><td>(D)</td><td>D</td><td>(IV) fourth</td></tr></table>"""


def block(content=PRIMARY, bbox=(10, 20, 190, 150), label="table"):
    return SimpleNamespace(label=label, content=content, bbox=bbox, image=None)


def page(*blocks):
    pixels = np.zeros((200, 200, 3), dtype=np.uint8)
    pixels[:, :] = [11, 22, 33]
    return {
        "page_index": 7,
        "parsing_res_list": list(blocks),
        "doc_preprocessor_res": {"output_img": pixels},
    }


def test_detects_only_incomplete_match_list_tables():
    assert incomplete_match_list(PRIMARY) == ("A", "B", "C", "D")
    assert incomplete_match_list(RETRY) is None
    assert incomplete_match_list("<table><tr><td>(A)</td></tr></table>") is None
    assert complete_match_list(RETRY, ("A", "B", "C", "D"))


def test_retry_uses_table_prompt_and_replaces_complete_result():
    table = block()
    paths = []

    def predict(**kwargs):
        paths.append(Path(kwargs.pop("input")))
        assert kwargs == {
            "use_layout_detection": False,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "prompt_label": "table",
        }
        with Image.open(paths[0]) as crop:
            assert crop.size == (196, 146)
            assert crop.getpixel((0, 0)) == (33, 22, 11)
        return [{"parsing_res_list": [block(RETRY)]}]

    pipeline = Mock(predict=Mock(side_effect=predict))
    retry_incomplete_tables(page(table), pipeline, 8)
    assert table.content == RETRY
    assert not paths[0].exists()


def test_incomplete_retry_preserves_primary(caplog):
    table = block()
    pipeline = Mock()
    pipeline.predict.return_value = [{"parsing_res_list": [block(PRIMARY)]}]
    retry_incomplete_tables(page(table), pipeline, 8)
    assert table.content == PRIMARY
    assert "remained incomplete" in caplog.text


def test_complete_and_non_match_tables_skip_retry():
    complete = block(RETRY)
    ordinary = block("<table><tr><td>Value</td></tr></table>")
    pipeline = Mock()
    retry_incomplete_tables(page(complete, ordinary), pipeline, 8)
    pipeline.predict.assert_not_called()
