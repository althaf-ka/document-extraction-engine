from pathlib import Path
from unittest.mock import Mock

import pytest

from document_extractor.exceptions import DocumentExtractorError, InputFileNotFoundError
from document_extractor.models import DocumentElement, NormalizedDocument, Page
from document_extractor.pipeline import ExtractionPipeline, page_count


def test_run_exports_document(tmp_path: Path) -> None:
    source = tmp_path / "input.pdf"
    source.touch()
    engine = Mock()
    engine.extract.return_value = NormalizedDocument(
        [Page(1, [DocumentElement("text", "Hello")])], str(source)
    )
    pipeline = ExtractionPipeline(engine=engine)
    bundle = pipeline.run(source, tmp_path / "output")
    assert bundle.document_path.read_text() == "Hello\n"
    assert bundle.tables_dir.is_dir()
    assert bundle.images_dir.is_dir()


def test_validation_precedes_inference(tmp_path: Path) -> None:
    engine = Mock()
    with pytest.raises(InputFileNotFoundError):
        ExtractionPipeline(engine=engine).run(tmp_path / "missing.pdf", tmp_path)
    engine.extract.assert_not_called()


def test_engine_failure_does_not_create_bundle(tmp_path: Path) -> None:
    source = tmp_path / "input.pdf"
    source.touch()
    engine = Mock()
    engine.extract.side_effect = RuntimeError("Server unavailable")
    with pytest.raises(DocumentExtractorError, match="Server unavailable"):
        ExtractionPipeline(engine=engine).run(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_pipeline_reports_completed_export_only_after_bundle_exists(tmp_path):
    source = tmp_path / "input.png"
    source.touch()
    engine = Mock()
    engine.extract.return_value = NormalizedDocument([Page(1)], str(source))
    events = []

    def on_event(event):
        if event.stage == "export" and event.completed:
            assert (tmp_path / "output" / "input" / "document.md").is_file()
        events.append(event)

    ExtractionPipeline(engine=engine, on_progress=on_event).run(
        source, tmp_path / "output"
    )
    assert [event.stage for event in events if event.completed] == [
        "validation",
        "normalization",
        "export",
    ]
    assert events[1].total == 1


def test_page_count_reads_metadata_without_rendering(tmp_path):
    import pypdfium2

    source = tmp_path / "two.pdf"
    with pypdfium2.PdfDocument.new() as pdf:
        for _ in range(2):
            pdf.new_page(100, 100).close()
        pdf.save(source)
    assert page_count(source) == 2
    source.write_bytes(b"not a PDF")
    assert page_count(source) is None
