"""Optional filtering of repeated text backed by a tiled PDF graphics group."""

import logging
import math
from collections import Counter, defaultdict
from ctypes import c_float
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from PIL import Image

from document_extractor.config import WatermarkMode
from document_extractor.engines.normalization import normalize_table

_MIN_REPEATS = 12
_MAX_RENDER_PIXELS = 4_000_000
_RENDER_SCALE = 1.5
_LOGGER = logging.getLogger(__name__)


def _phrase(value: str) -> str | None:
    words = value.casefold().split()
    # Numeric answers, formulas and punctuation-bearing content stay intact.
    if not 1 <= len(words) <= 4 or not all(word.isalpha() for word in words):
        return None
    phrase = " ".join(words)
    return phrase if 4 <= len(phrase) <= 80 else None


def repeated_phrase(label: str, content: str) -> tuple[str, int] | None:
    """Find entirely repetitive plain content, never a majority-text match."""
    if label == "table":
        table = normalize_table(content)
        if table.rows is None or table.has_header:
            return None
        if len(table.rows) < 3 or len(table.rows[0]) < 3:
            return None
        cells = [
            " ".join(cell.casefold().split()) for row in table.rows for cell in row
        ]
        counts = Counter(cell for cell in cells if cell)
        if len(counts) != 1:
            return None
        phrase, count = counts.most_common(1)[0]
        if count >= _MIN_REPEATS and _phrase(phrase) is not None:
            return phrase, count
    elif label == "text":
        words = content.casefold().split()
        for length in range(1, 5):
            count, remainder = divmod(len(words), length)
            if remainder or count < _MIN_REPEATS:
                continue
            phrase = _phrase(" ".join(words[:length]))
            if phrase and words == words[:length] * count:
                return phrase, count
    return None


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    try:
        left, top, right, bottom = map(float, value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (left, top, right, bottom)):
        return None
    if not 0 <= left < right or not 0 <= top < bottom:
        return None
    return left, top, right, bottom


@dataclass
class _Candidate:
    index: int
    bbox: tuple[float, float, float, float] | None
    record: dict[str, Any]


def _candidates(result: Any, page_number: int) -> list[_Candidate]:
    candidates = []
    for index, block in enumerate(result["parsing_res_list"]):
        # Never suppress a block that carries an image, including rich tables.
        if getattr(block, "image", None) is not None:
            continue
        repeated = repeated_phrase(block.label, block.content)
        if repeated is None:
            continue
        phrase, count = repeated
        bbox = _bbox(block.bbox)
        candidates.append(
            _Candidate(
                index,
                bbox,
                {
                    "page": page_number,
                    "block_index": index,
                    "kind": block.label,
                    "bbox": bbox,
                    "phrase": phrase,
                    "repetitions": count,
                    "content": block.content,
                    "decision": "kept",
                    "reason": "unverified",
                },
            )
        )
    return candidates


def _regular_grid(points: list[tuple[float, float]]) -> bool:
    xs = sorted({round(x, 1) for x, _ in points})
    ys = sorted({round(y, 1) for _, y in points})
    locations = {(round(x, 1), round(y, 1)) for x, y in points}
    if min(len(xs), len(ys)) < 3 or len(locations) != len(xs) * len(ys):
        return False
    if len(locations) != len(points):
        return False
    for axis in (xs, ys):
        gaps = [b - a for a, b in pairwise(axis)]
        # PDF generators can introduce small row-spacing changes. The JEE
        # background varies by about 2.2%; still require a complete grid and
        # matching vector outlines, rather than accepting repetition alone.
        if min(gaps) <= 0 or max(gaps) - min(gaps) > max(0.2, min(gaps) * 0.03):
            return False
    return True


def _path_signature(obj: Any, raw: Any) -> tuple | None:
    """Compare vector outlines, not just similarly sized bounding boxes."""
    count = raw.FPDFPath_CountSegments(obj)
    if not 1 <= count <= 4096:
        return None
    points = []
    origin = None
    for index in range(count):
        segment = raw.FPDFPath_GetPathSegment(obj, index)
        x, y = c_float(), c_float()
        if not segment or not raw.FPDFPathSegment_GetPoint(segment, x, y):
            return None
        origin = origin or (x.value, y.value)
        points.append(
            (
                raw.FPDFPathSegment_GetType(segment),
                x.value - origin[0],
                y.value - origin[1],
                bool(raw.FPDFPathSegment_GetClose(segment)),
            )
        )
    return tuple(points)


def _tiled_form(page: Any, raw: Any) -> Any | None:
    forms = [
        obj
        for obj in page.get_objects(max_depth=1)
        if obj.type == raw.FPDF_PAGEOBJ_FORM
    ]
    # Multiple or nested groups are ambiguous. Do not combine their evidence.
    if len(forms) != 1:
        return None
    form = forms[0]
    left, bottom, right, top = form.get_bounds()
    width, height = page.get_size()
    if (right - left) * (top - bottom) < width * height * 0.25:
        return None
    paths = []
    for obj in page.get_objects(form=form, max_depth=1):
        if obj.type != raw.FPDF_PAGEOBJ_PATH:
            return None
        paths.append(obj)
        if len(paths) > 1024:
            return None
    if len(paths) < _MIN_REPEATS:
        return None
    shapes = defaultdict(list)
    outlines = {}
    total_segments = 0
    for obj in paths:
        total_segments += raw.FPDFPath_CountSegments(obj)
        if total_segments > 250_000:
            return None
        signature = _path_signature(obj, raw)
        if signature is None:
            return None
        left, bottom, right, top = obj.get_bounds()
        key = (
            round(right - left),
            round(top - bottom),
            tuple((point[0], point[3]) for point in signature),
        )
        if key in outlines:
            # Translation of large coordinates can introduce float32 noise.
            if any(
                abs(a[1] - b[1]) > 0.05 or abs(a[2] - b[2]) > 0.05
                for a, b in zip(signature, outlines[key])
            ):
                return None
        else:
            outlines[key] = signature
        shapes[key].append((left, bottom))
    if not all(
        len(points) >= _MIN_REPEATS and _regular_grid(points)
        for points in shapes.values()
    ):
        return None
    return form


def _render(page: Any) -> Image.Image:
    bitmap = page.render(scale=_RENDER_SCALE)
    try:
        image = bitmap.to_pil()
        try:
            return image.convert("RGB")
        finally:
            image.close()
    finally:
        bitmap.close()


def _dimensions(result: Any) -> tuple[int, int] | None:
    pixels = result.get("doc_preprocessor_res", {}).get("output_img")
    if isinstance(pixels, Image.Image):
        return pixels.size
    if pixels is not None and hasattr(pixels, "shape") and len(pixels.shape) >= 2:
        return int(pixels.shape[1]), int(pixels.shape[0])
    width, height = result.get("width"), result.get("height")
    if isinstance(width, int) and isinstance(height, int) and min(width, height) > 0:
        return width, height
    return None


def _darkest_channel(image: Image.Image) -> int:
    # _render always produces RGB; Pillow also types extrema for scalar modes.
    extrema = cast(tuple[tuple[int, int], ...], image.getextrema())
    return min(channel[0] for channel in extrema)


def _verify_page(
    page: Any, result: Any, candidates: list[_Candidate]
) -> dict[int, str]:
    from pypdfium2 import raw

    reasons = dict.fromkeys(
        (c.index for c in candidates), "no_verified_tiled_background"
    )
    dimensions = _dimensions(result)
    width, height = page.get_size()
    # Only support the untransformed page coordinate system in this version.
    if (
        dimensions is None
        or page.get_rotation() != 0
        or tuple(page.get_bbox()) != (0.0, 0.0, width, height)
        or abs(dimensions[0] / width - dimensions[1] / height) > 0.01
        or width * height * _RENDER_SCALE**2 > _MAX_RENDER_PIXELS
    ):
        return dict.fromkeys(reasons, "unsupported_page_geometry")
    form = _tiled_form(page, raw)
    if form is None:
        return reasons
    with _render(page) as original:
        page.remove_obj(form)
        try:
            with _render(page) as foreground:
                for candidate in candidates:
                    bbox = candidate.bbox
                    if (
                        bbox is None
                        or bbox[2] > dimensions[0]
                        or bbox[3] > dimensions[1]
                    ):
                        reasons[candidate.index] = "invalid_block_geometry"
                        continue
                    left, top, right, bottom = bbox
                    # Expand, rather than shrink, the region to protect edge content.
                    region = (
                        max(0, math.floor(left / dimensions[0] * original.width) - 2),
                        max(0, math.floor(top / dimensions[1] * original.height) - 2),
                        min(
                            original.width,
                            math.ceil(right / dimensions[0] * original.width) + 2,
                        ),
                        min(
                            original.height,
                            math.ceil(bottom / dimensions[1] * original.height) + 2,
                        ),
                    )
                    with original.crop(region) as crop:
                        darkest = _darkest_channel(crop)
                    with foreground.crop(region) as crop:
                        remaining = _darkest_channel(crop)
                    if darkest < 220:
                        reason = "foreground_contrast"
                    elif darkest >= 250:
                        reason = "no_visible_background_evidence"
                    elif remaining < 250:
                        reason = "content_remains_without_background"
                    else:
                        reason = "repeated_pale_tiled_background_only"
                    reasons[candidate.index] = reason
        finally:
            # The document is disposable and is never saved. Free the detached form.
            form.close()
    return reasons


def filter_watermarks(
    results: list[Any],
    source: Path,
    mode: WatermarkMode,
    *,
    coordinates_transformed: bool = False,
) -> list[dict[str, Any]]:
    """Filter verified blocks and return a recoverable record of every candidate.

    Report mode uses the same checks but never changes results. Unsupported
    geometry, scans, mixed blocks and verification errors all retain content.
    """
    if mode == WatermarkMode.OFF:
        return []
    report = []
    pdf = None
    open_failed = False
    try:
        for index, result in enumerate(results):
            page_index = result.get("page_index")
            page_index = index if page_index is None else page_index
            candidates = _candidates(result, page_index + 1)
            if not candidates:
                continue
            reasons = dict.fromkeys((c.index for c in candidates), "unsupported_input")
            if source.suffix.lower() == ".pdf" and not coordinates_transformed:
                try:
                    if pdf is None and not open_failed:
                        import pypdfium2

                        try:
                            pdf = pypdfium2.PdfDocument(source)
                        except Exception:
                            open_failed = True
                            raise
                    if pdf is not None:
                        page = pdf[page_index]
                        try:
                            reasons = _verify_page(page, result, candidates)
                        finally:
                            page.close()
                    else:
                        reasons = dict.fromkeys(reasons, "verification_failed")
                except Exception:
                    _LOGGER.warning(
                        "Watermark verification failed on page %s; keeping content",
                        page_index + 1,
                        exc_info=True,
                    )
                    reasons = dict.fromkeys(reasons, "verification_failed")
            elif coordinates_transformed:
                reasons = dict.fromkeys(reasons, "preprocessing_changes_coordinates")
            excluded = set()
            for candidate in candidates:
                reason = reasons[candidate.index]
                candidate.record["reason"] = reason
                if reason == "repeated_pale_tiled_background_only":
                    if mode == WatermarkMode.FILTER:
                        excluded.add(candidate.index)
                        candidate.record["decision"] = "excluded"
                    else:
                        candidate.record["decision"] = "would_exclude"
                report.append(candidate.record)
            if excluded:
                result["parsing_res_list"] = [
                    block
                    for i, block in enumerate(result["parsing_res_list"])
                    if i not in excluded
                ]
                _LOGGER.info(
                    "Excluded %d background-only blocks on page %s; use --debug to retain diagnostics",
                    len(excluded),
                    page_index + 1,
                )
    finally:
        if pdf is not None:
            pdf.close()
    return report
