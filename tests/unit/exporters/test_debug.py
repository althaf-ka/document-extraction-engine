import json
from pathlib import Path

import pytest

from document_extractor.config import ExtractionConfig
from document_extractor.exporters.debug import write_json
from document_extractor.models import DocumentElement, NormalizedDocument, Page
from document_extractor.pipeline import ExtractionPipeline


@pytest.mark.parametrize("debug", [False, True])
def test_optional_debug_preserves_omitted_blocks(tmp_path, monkeypatch, debug):
    monkeypatch.setattr("document_extractor.pipeline.validate_input", lambda path: None)
    document = NormalizedDocument(
        [
            Page(
                1,
                [
                    DocumentElement("header", "Running header"),
                    DocumentElement("text", "Body"),
                ],
            )
        ],
        "sample.pdf",
        "paddle-vl",
        [{"res": {"parsing_res_list": []}}],
    )

    class Engine:
        def extract(self, path: Path) -> NormalizedDocument:
            return document

    engine = Engine()
    bundle = ExtractionPipeline(
        engine=engine, config=ExtractionConfig(debug=debug)
    ).run(Path("sample.pdf"), tmp_path)
    assert bundle.document_path.read_text() == "Body\n"
    assert document.pages[0].elements[0].content == "Running header"
    directory = bundle.root_dir / "debug"
    assert directory.exists() is debug
    if debug:
        payload = json.loads((directory / "elements.json").read_text())
        assert payload["elements"][0]["included_in_markdown"] is False
        assert payload["elements"][0]["content"] == "Running header"
        assert payload["elements"][1]["source"] == "paddle-vl"
        assert (
            json.loads((directory / "paddle-result.json").read_text())
            == document.debug_results
        )


def test_invalid_json_does_not_create_artifact(tmp_path):
    destination = tmp_path / "debug" / "elements.json"
    with pytest.raises(TypeError):
        write_json(destination, {"pixels": b"not JSON"})
    assert not destination.exists()


def test_json_replacement_leaves_no_temporary_file(tmp_path):
    destination = tmp_path / "elements.json"
    write_json(destination, {"old": 1})
    write_json(destination, {"new": 2})
    assert json.loads(destination.read_text()) == {"new": 2}
    assert list(tmp_path.iterdir()) == [destination]
