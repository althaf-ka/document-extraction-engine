# pyright: reportMissingImports=false
# Docling is an optional extra; the default environment does not install it.
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

from document_extractor.engines.normalization import (
    formula_placement,
    normalize_formula,
    normalize_table,
)
from document_extractor.events import ProgressCallback, ProgressEvent, ignore_progress
from document_extractor.models import DocumentElement, NormalizedDocument, Page


class DoclingEngine:
    def __init__(self, *, on_progress: ProgressCallback = ignore_progress) -> None:
        self._converter: Any = None
        self.on_progress = on_progress

    def extract(self, path: Path) -> NormalizedDocument:
        self.on_progress(ProgressEvent("loading", "Preparing Docling"))
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import (
                DocumentConverter,
                ImageFormatOption,
                PdfFormatOption,
            )
            from docling_core.types.doc import PictureItem, TableItem
        except ImportError as error:
            raise RuntimeError(
                "Install Docling with: uv sync --extra docling"
            ) from error

        if self._converter is None:
            options = PdfPipelineOptions()
            options.generate_picture_images = True
            options.generate_page_images = True
            options.do_formula_enrichment = True
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=options),
                    InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
                }
            )
        self.on_progress(ProgressEvent("loading", "Converter ready", completed=True))
        self.on_progress(ProgressEvent("inference", "Extracting with Docling"))
        logging.getLogger(__name__).info(
            "Running Docling conversion; waiting for document result"
        )
        document = self._converter.convert(path).document
        self.on_progress(
            ProgressEvent(
                "inference",
                "Extracted",
                current=len(document.pages),
                total=len(document.pages),
                completed=True,
            )
        )
        self.on_progress(ProgressEvent("normalization", "Normalizing document"))
        pages = {number: Page(number) for number in sorted(document.pages)}
        for order, (item, _) in enumerate(document.iterate_items()):
            provenance = item.prov[0] if item.prov else None
            number = provenance.page_no if provenance else 1
            page = pages.setdefault(number, Page(number))
            bbox = None
            if provenance:
                box = provenance.bbox.to_top_left_origin(
                    page_height=document.pages[number].size.height
                )
                bbox = (box.l, box.t, box.r, box.b)
            kind = str(item.label.value)
            content = getattr(item, "text", "")
            image_png = None
            if isinstance(item, TableItem):
                kind = "table"
                content = item.export_to_html(doc=document)
            elif isinstance(item, PictureItem):
                kind = "image"
                image = item.get_image(document)
                if image is None:
                    raise RuntimeError(
                        f"Docling returned a picture without pixels on page {number}"
                    )
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                image_png = buffer.getvalue()
            elif kind in {"title", "section_header"}:
                content = f"{'#' * getattr(item, 'level', 1)} {content}"
            elif kind == "list_item":
                content = f"{getattr(item, 'marker', '') or '-'} {content}"
            placement = None
            table = normalize_table(content) if kind == "table" else None
            if kind in {"formula", "display_formula", "inline_formula"}:
                content = normalize_formula(content)
                placement = formula_placement(kind, bbox)
            page.elements.append(
                DocumentElement(kind, content, bbox, order, image_png, placement, table)
            )
        return NormalizedDocument(list(pages.values()), str(path), engine="docling")
