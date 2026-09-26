import logging
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

from document_extractor.config import (
    ExtractionConfig,
    InferenceBackend,
    PaddleVLConfig,
)
from document_extractor.engines.normalization import (
    formula_placement,
    normalize_formula,
    normalize_table,
)
from document_extractor.events import ProgressCallback, ProgressEvent, ignore_progress
from document_extractor.html_images import rewrite_image_sources
from document_extractor.models import DocumentElement, NormalizedDocument, Page
from document_extractor.processing.mcq import retry_missing_options
from document_extractor.processing.tables import retry_incomplete_tables
from document_extractor.processing.watermarks import filter_watermarks


class PaddleVLEngine:
    """Keep Paddle objects and runtime configuration inside this adapter."""

    def __init__(
        self,
        config: ExtractionConfig | PaddleVLConfig,
        *,
        debug: bool = False,
        on_progress: ProgressCallback = ignore_progress,
    ):
        self.config = (
            config.to_paddle_config()
            if isinstance(config, ExtractionConfig)
            else config
        )
        self.debug = config.debug if isinstance(config, ExtractionConfig) else debug
        self._pipeline: Any = None
        self.on_progress = on_progress

    def extract(self, path: Path) -> NormalizedDocument:
        logger = logging.getLogger(__name__)
        self.on_progress(ProgressEvent("loading", "Preparing models"))
        if self._pipeline is None:
            logger.info(
                "Loading Paddle runtime and models; first use may download weights"
            )
            from paddleocr import PaddleOCRVL

            vlm = self.config.vlm
            options: dict[str, Any] = {
                "pipeline_version": self.config.pipeline_version,
                "use_doc_orientation_classify": self.config.use_doc_orientation_classify,
                "use_doc_unwarping": self.config.use_doc_unwarping,
                "use_chart_recognition": self.config.use_chart_recognition,
                "use_layout_detection": self.config.use_layout_detection,
                "vl_rec_max_concurrency": vlm.max_concurrency,
                "layout_threshold": self.config.layout_threshold,
                "layout_unclip_ratio": list(self.config.layout_unclip_ratio),
                "layout_merge_bboxes_mode": self.config.layout_merge_bboxes_mode,
                "format_block_content": self.config.format_block_content,
                "merge_layout_blocks": self.config.merge_layout_blocks,
                "use_queues": self.config.use_queues,
                "markdown_ignore_labels": list(self.config.markdown_ignore_labels),
            }
            if vlm.backend != InferenceBackend.LOCAL:
                options.update(
                    vl_rec_backend={
                        InferenceBackend.LLAMA_CPP: "llama-cpp-server",
                        InferenceBackend.MLX: "mlx-vlm-server",
                    }[vlm.backend],
                    vl_rec_server_url=vlm.server_url,
                )
                if vlm.backend == InferenceBackend.MLX:
                    options["vl_rec_api_model_name"] = vlm.mlx_model_name
            self._pipeline = PaddleOCRVL(**options)

        self.on_progress(ProgressEvent("loading", "Models ready", completed=True))

        logger.info("Running Paddle inference; waiting for the first page result")
        results = []
        self.on_progress(ProgressEvent("inference", "Extracting", current=0))
        total_pages = None
        for result in self._pipeline.predict(
            input=str(path), layout_shape_mode=self.config.layout_shape_mode
        ):
            results.append(result)
            total_pages = result.get("page_count") or total_pages
            self.on_progress(
                ProgressEvent(
                    "inference", "Extracting", current=len(results), total=total_pages
                )
            )
            logger.info(
                "Received %d page result(s); waiting for remaining inference to finish",
                len(results),
            )
        logger.info("Inference finished: %d page result(s)", len(results))
        if not results:
            raise RuntimeError("The engine returned no pages")
        if results:
            self.on_progress(
                ProgressEvent(
                    "inference",
                    "Extracted",
                    current=len(results),
                    total=total_pages,
                    completed=True,
                )
            )
        # Capture before reconstruction so debug output can distinguish model
        # output from page merging and adapter normalization. No image pixels.
        debug_results = (
            [deepcopy(result.json) for result in results] if self.debug else None
        )
        watermark_report = filter_watermarks(
            results,
            path,
            self.config.watermark_mode,
            coordinates_transformed=(
                self.config.use_doc_orientation_classify
                or self.config.use_doc_unwarping
            ),
        )
        recovery_enabled = self.config.mcq_retry or self.config.table_retry
        if recovery_enabled and self.config.use_layout_detection:
            self.on_progress(ProgressEvent("recovery", "Checking incomplete questions"))
            for result in results:
                if self.config.mcq_retry:
                    retry_missing_options(
                        result,
                        self._pipeline,
                        self.config.mcq_crop_margin,
                        on_progress=self.on_progress,
                    )
                if self.config.table_retry:
                    retry_incomplete_tables(
                        result,
                        self._pipeline,
                        self.config.table_crop_margin,
                        on_progress=self.on_progress,
                    )
            self.on_progress(
                ProgressEvent("recovery", "Question checks complete", completed=True)
            )
        if path.suffix.lower() == ".pdf":
            # Table reconstruction can move HTML to another page. Qualify crop
            # paths first so equal coordinates on different pages stay distinct.
            qualify_image_paths(results)
            self.on_progress(ProgressEvent("reconstruction", "Reconstructing document"))
            logger.info("Reconstructing PDF headings and cross-page tables")
            results = self._pipeline.restructure_pages(
                results,
                merge_tables=True,
                relevel_titles=True,
                concatenate_pages=False,
            )
        logger.info("Converting Paddle results into document elements")
        if path.suffix.lower() == ".pdf":
            self.on_progress(
                ProgressEvent(
                    "reconstruction", "Document reconstructed", completed=True
                )
            )
        self.on_progress(ProgressEvent("normalization", "Normalizing document"))
        document = adapt_results(results, path)
        document.debug_results = debug_results
        document.watermark_report = watermark_report
        return document


def qualify_image_paths(results: Any) -> None:
    for index, result in enumerate(results):
        mapping = {}
        for asset in result.get("imgs_in_doc", []):
            original = asset["path"]
            qualified = (
                original
                if original.startswith("paddle-page-")
                else f"paddle-page-{index}/{original}"
            )
            mapping[original] = qualified
            asset["path"] = qualified
        for block in result["parsing_res_list"]:
            if "<img" in block.content.lower():
                block.content = rewrite_image_sources(
                    block.content,
                    lambda source, mapping=mapping: mapping.get(source, source),
                )


def adapt_results(results: Any, path: Path) -> NormalizedDocument:
    results = list(results)
    qualify_image_paths(results)
    pages = []
    image_assets = {}
    for index, result in enumerate(results):
        for asset in result.get("imgs_in_doc", []):
            if asset.get("img") is not None:
                buffer = BytesIO()
                asset["img"].save(buffer, format="PNG")
                image_assets[asset["path"]] = buffer.getvalue()
        page_index = result.get("page_index")
        page = Page(number=(page_index if page_index is not None else index) + 1)
        # Paddle's list already follows reading order, including unnumbered blocks.
        blocks = result["parsing_res_list"]
        for order, block in enumerate(blocks):
            image_png = None
            content = block.content
            placement = None
            table = None
            if block.label in {"formula", "display_formula", "inline_formula"}:
                content = normalize_formula(content)
                neighbors = (
                    blocks[max(0, order - 1) : order] + blocks[order + 1 : order + 2]
                )
                placement = formula_placement(block.label, block.bbox, neighbors)
            elif block.label == "table":
                table = normalize_table(content)
            if block.label in {
                "doc_title",
                "paragraph_title",
            } and not content.startswith("#"):
                level = (
                    1
                    if block.label == "doc_title"
                    else getattr(block, "title_level", 1) + 1
                )
                content = f"{'#' * min(6, max(1, level))} {content}"
            if block.image is not None and block.image.get("img") is not None:
                buffer = BytesIO()
                block.image["img"].save(buffer, format="PNG")
                image_png = buffer.getvalue()
            elif block.label in {"image", "chart", "seal"}:
                raise RuntimeError(
                    f"Paddle returned an image without pixels on page {page.number}"
                )
            page.elements.append(
                DocumentElement(
                    kind=block.label,
                    content=content,
                    bbox=tuple(float(value) for value in block.bbox),
                    order=order,
                    image_png=image_png,
                    formula_placement=placement,
                    table=table,
                )
            )
        pages.append(page)
    return NormalizedDocument(
        pages=pages, source=str(path), engine="paddle-vl", image_assets=image_assets
    )
