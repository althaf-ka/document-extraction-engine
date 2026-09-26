import logging
from pathlib import Path

from document_extractor.config import ExtractionConfig, WatermarkMode
from document_extractor.engines import create_engine
from document_extractor.engines.base import DocumentEngine
from document_extractor.events import ProgressCallback, ProgressEvent, ignore_progress
from document_extractor.exceptions import DocumentExtractorError
from document_extractor.exporters.bundle import OutputBundle
from document_extractor.exporters.debug import export_debug, write_json
from document_extractor.exporters.document import DocumentExporter
from document_extractor.option_normalization import normalize_options
from document_extractor.validation import validate_input


def page_count(path: Path) -> int | None:
    """Read PDF metadata without rendering. Unknown counts must not block extraction."""
    if path.suffix.lower() != ".pdf":
        return 1
    try:
        import pypdfium2

        with pypdfium2.PdfDocument(path) as pdf:
            return len(pdf)
    except Exception:
        logging.getLogger(__name__).debug("Could not read page count", exc_info=True)
        return None


class ExtractionPipeline:
    """Validate, extract, and export. Reuse one instance for multiple inputs."""

    def __init__(
        self,
        engine: DocumentEngine | None = None,
        config: ExtractionConfig | None = None,
        on_progress: ProgressCallback = ignore_progress,
    ) -> None:
        self.config = config or ExtractionConfig()
        self.on_progress = on_progress
        self.engine = (
            engine
            if engine is not None
            else create_engine(self.config, on_progress=on_progress)
        )
        self.exporter = DocumentExporter(
            question_layout=self.config.question_layout,
            export_layout_report=self.config.debug,
        )

    def run(self, input_path: Path, output_root: Path) -> OutputBundle:
        logging.getLogger(__name__).info("Validating %s", input_path)
        self.on_progress(ProgressEvent("validation", "Validating input"))
        validate_input(input_path)
        self.on_progress(
            ProgressEvent(
                "validation",
                "Input validated",
                total=page_count(input_path),
                completed=True,
            )
        )
        try:
            logging.getLogger(__name__).info("Extracting document")
            document = self.engine.extract(input_path)
            if not document.pages:
                raise DocumentExtractorError("The engine returned no pages")
            self.on_progress(ProgressEvent("normalization", "Normalizing document"))
            normalize_options(document)
            self.on_progress(
                ProgressEvent("normalization", "Document normalized", completed=True)
            )
            logging.getLogger(__name__).info("Writing Markdown, tables, and images")
            self.on_progress(ProgressEvent("export", "Exporting results"))
            bundle = self.exporter.export(document, input_path, output_root)
            if self.config.watermark_mode == WatermarkMode.REPORT or self.config.debug:
                watermark_path = (
                    bundle.root_dir / "watermarks.json"
                    if self.config.watermark_mode == WatermarkMode.REPORT
                    else bundle.root_dir / "debug" / "watermarks.json"
                )
                write_json(
                    watermark_path,
                    {
                        "schema_version": 1,
                        "mode": self.config.watermark_mode,
                        "bbox_coordinates": "original Paddle page pixels, top-left origin",
                        "candidates": document.watermark_report,
                    },
                )
            if self.config.debug:
                export_debug(document, bundle.root_dir / "debug", self.config)
            self.on_progress(
                ProgressEvent("export", "Results exported", completed=True)
            )
            return bundle
        except DocumentExtractorError:
            raise
        except Exception as error:
            raise DocumentExtractorError(f"Extraction failed: {error}") from error
