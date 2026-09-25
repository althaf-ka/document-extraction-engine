"""Retry incomplete MCQs using the unmasked page pixels, once per block."""

import logging
import math
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

from document_extractor.events import ProgressCallback, ProgressEvent, ignore_progress
from document_extractor.option_normalization import _outside_math

_QUESTION = re.compile(r"(?im)^\s*(?:Q(?:uestion)?\s*)?(\d+)\s*[.:]\s*")
_OPTION = re.compile(r"(?<!\S)\(([1-9])\)")


def parse_options(text: str) -> dict[int, str]:
    """Reject duplicate/ambiguous labels and ignore labels inside math or code."""
    visible = _outside_math(text)
    if visible is None or re.search(r"</?[A-Za-z][^>]*>", text):
        return {}
    matches = list(_OPTION.finditer(visible))
    labels = [int(match[1]) for match in matches]
    if labels != sorted(set(labels)) or any(label > 4 for label in labels):
        return {}
    options = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[match.end() : end].strip()
        if not value:
            return {}
        options[int(match[1])] = value
    return options


def merge_missing(primary: str, retry: str) -> str:
    """Keep primary text byte-for-byte; append only recovered missing options."""
    question = _QUESTION.match(primary)
    retry_questions = list(_QUESTION.finditer(retry))
    if question is None or len(retry_questions) > 1:
        return primary
    if retry_questions and retry_questions[0][1] != question[1]:
        return primary
    existing = parse_options(primary)
    recovered = parse_options(retry)
    additions = [
        f"({label}) {recovered[label]}"
        for label in (3, 4)
        if label not in existing and label in recovered
    ]
    return primary + ("\n\n" + "\n\n".join(additions) if additions else "")


def retry_missing_options(
    result: Any,
    pipeline: Any,
    margin: int,
    *,
    on_progress: ProgressCallback = ignore_progress,
) -> None:
    blocks = result["parsing_res_list"]
    pixels = result.get("doc_preprocessor_res", {}).get("output_img")
    if pixels is None:
        return
    page_image = None
    logger = logging.getLogger(__name__)
    for block in blocks:
        text = block.content
        if block.label != "text" or not _QUESTION.match(text):
            continue
        if len(list(_QUESTION.finditer(text))) != 1:
            continue
        options = parse_options(text)
        if not {1, 2}.issubset(options) or {3, 4}.issubset(options):
            continue
        left, top, right, bottom = map(float, block.bbox)
        if not all(math.isfinite(v) for v in (left, top, right, bottom)):
            continue
        # Any horizontally overlapping block is a boundary, including a figure
        # or footer. Do not extend into another column or a separate option block.
        next_tops = [
            float(other.bbox[1])
            for other in blocks
            if other is not block
            and other.bbox[1] > top
            and min(right, other.bbox[2]) > max(left, other.bbox[0])
        ]
        if not next_tops or min(next_tops) - margin <= bottom:
            continue
        if page_image is None:
            # Paddle's retained render uses BGR; PIL expects RGB. These pixels
            # are already in the same coordinate frame as the layout boxes.
            page_image = (
                pixels.convert("RGB")
                if isinstance(pixels, Image.Image)
                else Image.fromarray(pixels[:, :, ::-1]).convert("RGB")
            )
        bounds = (
            max(0, math.floor(left)),
            max(0, math.floor(top)),
            min(page_image.width, math.ceil(right)),
            min(page_image.height, math.floor(min(next_tops) - margin)),
        )
        if bounds[2] <= bounds[0] or bounds[3] <= max(bounds[1], bottom):
            continue
        logger.info("Retrying incomplete MCQ on page %s", result.get("page_index"))
        on_progress(ProgressEvent("recovery", "Retrying missing question options"))
        try:
            with TemporaryDirectory(prefix="mcq-retry-") as directory:
                crop_path = Path(directory) / "crop.png"
                page_image.crop(bounds).save(crop_path)
                retry_results = pipeline.predict(
                    input=str(crop_path),
                    use_layout_detection=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    prompt_label="ocr",
                )
                retry_text = "\n".join(
                    item.content
                    for page in retry_results
                    for item in page["parsing_res_list"]
                )
            block.content = merge_missing(text, retry_text)
        except Exception:
            logger.warning("MCQ retry failed; keeping primary text", exc_info=True)
