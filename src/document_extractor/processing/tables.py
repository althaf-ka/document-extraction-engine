"""Retry match-list tables when the first transcription drops list labels."""

import logging
import math
import re
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

from document_extractor.events import ProgressCallback, ProgressEvent, ignore_progress

_LETTER_LABEL = re.compile(r"\(([A-Z])\)")
_ROMAN_LABEL = re.compile(r"\((VIII|VII|VI|IV|V|III|II|I)\)")
_ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII")


class _TableHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.rows.append([])
        elif tag in {"td", "th"} and self.rows:
            self.cell = []

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.cell is not None and self.rows:
            self.rows[-1].append(" ".join("".join(self.cell).split()))
            self.cell = None


def _match_list_labels(html: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    parser = _TableHTML()
    parser.feed(html)
    parser.close()
    list_two_column = None
    for row in parser.rows:
        list_one = next(
            (
                index
                for index, cell in enumerate(row)
                if re.search(r"List\s*[-–—]?\s*I\b(?!I)", cell, re.IGNORECASE)
            ),
            None,
        )
        list_two = next(
            (
                index
                for index, cell in enumerate(row)
                if re.search(r"List\s*[-–—]?\s*II\b", cell, re.IGNORECASE)
            ),
            None,
        )
        if list_one is not None and list_two is not None and list_one < list_two:
            list_two_column = list_two
            break
    if list_two_column is None:
        return None
    letters = tuple(
        dict.fromkeys(
            match
            for row in parser.rows
            for cell in row[:list_two_column]
            for match in _LETTER_LABEL.findall(cell)
        )
    )
    romans = tuple(
        dict.fromkeys(
            match
            for row in parser.rows
            for cell in row[list_two_column:]
            for match in _ROMAN_LABEL.findall(cell)
        )
    )
    if len(letters) < 2 or letters != tuple(chr(65 + i) for i in range(len(letters))):
        return None
    return letters, romans


def incomplete_match_list(html: str) -> tuple[str, ...] | None:
    """Return the expected row labels when List II has missing labels."""
    labels = _match_list_labels(html)
    if labels is None:
        return None
    letters, romans = labels
    expected = _ROMAN[: len(letters)]
    return letters if romans != expected else None


def complete_match_list(html: str, expected_letters: tuple[str, ...]) -> bool:
    labels = _match_list_labels(html)
    if labels is None:
        return False
    letters, romans = labels
    return letters == expected_letters and romans == _ROMAN[: len(letters)]


def retry_incomplete_tables(
    result: Any,
    pipeline: Any,
    margin: int,
    *,
    on_progress: ProgressCallback = ignore_progress,
) -> None:
    """Re-read incomplete match-list tables from their unmasked page crop."""
    pixels = result.get("doc_preprocessor_res", {}).get("output_img")
    if pixels is None:
        return
    page_image = None
    logger = logging.getLogger(__name__)
    for block in result["parsing_res_list"]:
        if block.label != "table":
            continue
        expected_letters = incomplete_match_list(block.content)
        if expected_letters is None or block.bbox is None or len(block.bbox) != 4:
            continue
        left, top, right, bottom = map(float, block.bbox)
        if not all(math.isfinite(value) for value in (left, top, right, bottom)):
            continue
        if page_image is None:
            page_image = (
                pixels.convert("RGB")
                if isinstance(pixels, Image.Image)
                else Image.fromarray(pixels[:, :, ::-1]).convert("RGB")
            )
        bounds = (
            max(0, math.floor(left - margin)),
            max(0, math.floor(top - margin)),
            min(page_image.width, math.ceil(right + margin)),
            min(page_image.height, math.ceil(bottom + margin)),
        )
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            continue
        logger.info(
            "Retrying incomplete match-list table on page %s", result.get("page_index")
        )
        on_progress(ProgressEvent("recovery", "Retrying incomplete table"))
        try:
            with TemporaryDirectory(prefix="table-retry-") as directory:
                crop_path = Path(directory) / "crop.png"
                page_image.crop(bounds).save(crop_path)
                retry_results = pipeline.predict(
                    input=str(crop_path),
                    use_layout_detection=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    prompt_label="table",
                )
                candidates = [
                    item.content
                    for page in retry_results
                    for item in page["parsing_res_list"]
                    if item.label == "table"
                ]
            replacement = next(
                (
                    candidate
                    for candidate in candidates
                    if complete_match_list(candidate, expected_letters)
                ),
                None,
            )
            if replacement is not None:
                block.content = replacement
            else:
                logger.warning("Table retry remained incomplete; keeping primary table")
        except Exception:
            logger.warning("Table retry failed; keeping primary table", exc_info=True)
