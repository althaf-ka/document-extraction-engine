"""Optional diagnostics, separate from the deliverable document."""

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from document_extractor.config import (
    MARKDOWN_IGNORE_LABELS,
    ExtractionConfig,
    PaddleVLConfig,
)
from document_extractor.models import NormalizedDocument


def write_json(path: Path, value: object) -> None:
    # Serialize first: invalid data must never leave an empty JSON artifact.
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_debug(
    document: NormalizedDocument,
    directory: Path,
    config: ExtractionConfig,
    *,
    paddle_config: PaddleVLConfig | None = None,
) -> None:
    elements = []
    for page in document.pages:
        for element in page.elements:
            elements.append(
                {
                    "page": page.number,
                    "order": element.order,
                    "kind": element.kind,
                    "label": element.kind,
                    "bbox": element.bbox,
                    "content": element.content,
                    "source": document.engine,
                    "formula_placement": element.formula_placement,
                    "option_group": asdict(element.option_group)
                    if element.option_group
                    else None,
                    "has_merged_cells": element.table.has_merged_cells
                    if element.table
                    else None,
                    "has_image": element.image_png is not None,
                    "included_in_markdown": element.kind not in MARKDOWN_IGNORE_LABELS,
                }
            )
    settings = asdict(config)
    # Server URLs may contain credentials or query tokens; they are not needed
    # to explain normalized content.
    settings.pop("server_url")
    if document.engine == "paddle-vl":
        paddle_settings = asdict(paddle_config or config.to_paddle_config())
        paddle_settings["vlm"].pop("server_url")
        settings["paddle"] = paddle_settings
    write_json(
        directory / "elements.json",
        {
            "schema_version": 1,
            "source_file": document.source,
            "bbox_coordinates": "source page coordinates, top-left origin; not normalized",
            "config": settings,
            "elements": elements,
        },
    )
    if document.debug_results is not None:
        write_json(directory / "paddle-result.json", document.debug_results)
