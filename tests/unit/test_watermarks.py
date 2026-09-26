"""Exercise decisions against real PDF rendering without loading OCR models."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from document_extractor.config import (
    DocumentEngineName,
    ExtractionConfig,
    Profile,
    WatermarkMode,
)
from document_extractor.engines.paddle_vl import PaddleVLEngine, adapt_results
from document_extractor.pipeline import ExtractionPipeline
from document_extractor.processing.watermarks import (
    _regular_grid,
    filter_watermarks,
    repeated_phrase,
)


def test_grid_tolerates_small_pdf_spacing_variations_but_not_irregular_layouts():
    rows = [0.2, 147.1, 294.0, 440.9, 584.6]
    points = [(x, y) for x in (0.8, 375.8, 750.8) for y in rows]
    assert _regular_grid(points)
    assert not _regular_grid(points[:-1])
    assert not _regular_grid([(x, y) for x in (0, 100, 230) for y in rows])


def table(phrase="Example Academy", *, extra="", header=False):
    rows = "".join("<tr>" + f"<td>{phrase}</td>" * 4 + "</tr>" for _ in range(4))
    if header:
        rows = "<tr>" + "<th>Results</th>" * 4 + "</tr>" + rows
    return f"<table>{rows}{extra}</table>"


def block(label="table", content=None, bbox=(0, 0, 600, 800), image=None):
    return SimpleNamespace(
        label=label,
        content=table() if content is None else content,
        bbox=bbox,
        image=image,
    )


def result(*blocks):
    return {
        "page_index": 0,
        "width": 600,
        "height": 800,
        "parsing_res_list": list(blocks),
    }


def make_pdf(tmp_path, *, pale=True, foreground="", irregular=False, rotated=False):
    """Small vector PDF with one tiled Form XObject and optional real foreground."""
    color = "0.94 0.95 0.96" if pale else "0 0 0"
    tiles = []
    for row in range(4):
        for column in range(4):
            x = 20 + column * 60
            if irregular and row == column == 3:
                x += 13
            tiles.append(f"{x} {20 + row * 80} 20 12 re f")
    form = (color + " rg\n" + "\n".join(tiles)).encode()
    content = ("q /Fm0 Do Q\n" + foreground).encode()

    def stream(data, attrs=b""):
        return (
            b"<< /Length "
            + str(len(data)).encode()
            + b" "
            + attrs
            + b" >>\nstream\n"
            + data
            + b"\nendstream"
        )

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 400] "
        + (b"/Rotate 90 " if rotated else b"")
        + b"/Resources << /XObject << /Fm0 5 0 R >> >> /Contents 4 0 R >>",
        stream(content),
        stream(
            form,
            b"/Type /XObject /Subtype /Form /BBox [0 0 300 400] /Resources << >>",
        ),
    ]
    data = b"%PDF-1.7\n"
    offsets = []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(data)
    data += b"xref\n0 6\n0000000000 65535 f \n"
    data += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    data += (
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"
        + str(xref).encode()
        + b"\n%%EOF\n"
    )
    source = tmp_path / "tiles.pdf"
    source.write_bytes(data)
    return source


@pytest.mark.parametrize("phrase", ["Mathongo", "SAMPLE", "ABC Academy", "示例水印"])
def test_discovers_different_phrases_without_a_publisher_list(phrase):
    assert repeated_phrase("table", table(phrase)) == (phrase.casefold(), 16)
    assert repeated_phrase("text", (phrase + "\n") * 12) == (phrase.casefold(), 12)


@pytest.mark.parametrize(
    "label,content",
    [
        ("table", table("Yes")),
        ("table", table("0")),
        ("table", table("x²")),
        ("table", table(header=True)),
        ("table", table(extra="<tr><td>Answer</td><td>42</td><td></td><td></td></tr>")),
        ("table", table().replace("Example Academy", "Important answer", 1)),
        ("table", table().replace("<table>", "<table><caption>Real data</caption>")),
        ("table", table().replace("<td>", '<td colspan="2">', 1)),
        ("table", table().replace("Example Academy", '<img src="figure.png">', 1)),
        ("text", "Sample " * 12 + "Answer is 42"),
        ("text", "Sample " * 11),
        ("formula", "Sample " * 30),
        ("image", "Sample " * 30),
    ],
)
def test_repetition_never_selects_mixed_or_rich_content(label, content):
    assert repeated_phrase(label, content) is None


def test_filter_excludes_only_verified_block_and_keeps_source_and_images(tmp_path):
    source = make_pdf(tmp_path)
    original_pdf = source.read_bytes()
    watermark = block()
    question = block("text", "Question 1. What is the answer?")
    figure = block("image", "", image={"img": object()})
    image_table = block(image={"img": object()})
    page = result(watermark, question, figure, image_table)
    report = filter_watermarks([page], source, WatermarkMode.FILTER)
    assert page["parsing_res_list"] == [question, figure, image_table]
    assert report[0]["decision"] == "excluded"
    assert report[0]["content"] == watermark.content
    assert source.read_bytes() == original_pdf


def test_report_mode_is_non_mutating(tmp_path):
    source = make_pdf(tmp_path)
    page = result(block())
    original_blocks = page["parsing_res_list"]
    report = filter_watermarks([page], source, WatermarkMode.REPORT)
    assert page["parsing_res_list"] is original_blocks
    assert report[0]["decision"] == "would_exclude"


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"pale": False}, "foreground_contrast"),
        ({"foreground": "0 0 0 rg 10 350 100 10 re f"}, "foreground_contrast"),
        (
            {"foreground": "0.96 0.96 0.96 rg 10 350 100 10 re f"},
            "content_remains_without_background",
        ),
        ({"irregular": True}, "no_verified_tiled_background"),
        ({"rotated": True}, "unsupported_page_geometry"),
    ],
)
def test_preserves_real_dark_or_faint_content_and_uncertain_groups(
    tmp_path, options, reason
):
    source = make_pdf(tmp_path, **options)
    candidate = block()
    page = result(candidate)
    report = filter_watermarks([page], source, WatermarkMode.FILTER)
    assert page["parsing_res_list"] == [candidate]
    assert report[0]["decision"] == "kept"
    assert report[0]["reason"] == reason


@pytest.mark.parametrize("bbox", [None, [0, 0, float("nan"), 4], [0, 0, 1000, 1000]])
def test_invalid_geometry_keeps_content_and_report_is_serializable(tmp_path, bbox):
    source = make_pdf(tmp_path)
    page = result(block(bbox=bbox))
    report = filter_watermarks([page], source, WatermarkMode.FILTER)
    assert len(page["parsing_res_list"]) == 1
    assert report[0]["reason"] == "invalid_block_geometry"
    json.dumps(report, allow_nan=False)


def test_missing_dimensions_prevents_filtering(tmp_path):
    source = make_pdf(tmp_path)
    page = result(block())
    del page["width"]
    report = filter_watermarks([page], source, WatermarkMode.FILTER)
    assert report[0]["reason"] == "unsupported_page_geometry"


def test_unsupported_or_transformed_inputs_do_not_open_pdf(monkeypatch):
    import pypdfium2

    constructor = Mock(side_effect=AssertionError("Must not open PDF"))
    monkeypatch.setattr(pypdfium2, "PdfDocument", constructor)
    for path, transformed in [("scan.png", False), ("robust.pdf", True)]:
        page = result(block())
        report = filter_watermarks(
            [page],
            Path(path),
            WatermarkMode.FILTER,
            coordinates_transformed=transformed,
        )
        assert report[0]["decision"] == "kept"
        assert len(page["parsing_res_list"]) == 1
    constructor.assert_not_called()


def test_disabled_or_no_candidates_does_not_open_pdf(monkeypatch):
    import pypdfium2

    constructor = Mock(side_effect=AssertionError("Must not open PDF"))
    monkeypatch.setattr(pypdfium2, "PdfDocument", constructor)
    assert (
        filter_watermarks([result(block())], Path("missing.pdf"), WatermarkMode.OFF)
        == []
    )
    assert (
        filter_watermarks(
            [result(block("text", "Ordinary content"))],
            Path("missing.pdf"),
            WatermarkMode.FILTER,
        )
        == []
    )
    constructor.assert_not_called()


def test_verification_error_preserves_content(tmp_path):
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"not a PDF")
    page = result(block())
    report = filter_watermarks([page], source, WatermarkMode.FILTER)
    assert report[0]["reason"] == "verification_failed"
    assert len(page["parsing_res_list"]) == 1


def test_filter_omits_report_without_debug(tmp_path):
    source = make_pdf(tmp_path)
    pages = [result(block())]
    report = filter_watermarks(pages, source, WatermarkMode.FILTER)
    document = adapt_results(pages, source)
    document.watermark_report = report
    engine = Mock()
    engine.extract.return_value = document
    bundle = ExtractionPipeline(
        engine=engine,
        config=ExtractionConfig(watermark_mode=WatermarkMode.FILTER),
    ).run(source, tmp_path / "output")
    assert not (bundle.root_dir / "watermarks.json").exists()
    assert not (bundle.root_dir / "debug").exists()
    assert "Example Academy" not in bundle.document_path.read_text()


def test_filter_saves_report_under_debug_when_requested(tmp_path):
    source = make_pdf(tmp_path)
    pages = [result(block())]
    report = filter_watermarks(pages, source, WatermarkMode.FILTER)
    document = adapt_results(pages, source)
    document.watermark_report = report
    engine = Mock()
    engine.extract.return_value = document
    bundle = ExtractionPipeline(
        engine=engine,
        config=ExtractionConfig(watermark_mode=WatermarkMode.FILTER, debug=True),
    ).run(source, tmp_path / "output")
    saved = json.loads((bundle.root_dir / "debug" / "watermarks.json").read_text())
    assert saved["candidates"][0]["content"] == table()
    assert saved["candidates"][0]["decision"] == "excluded"


def test_paddle_filters_before_reconstruction_without_extra_inference(tmp_path):
    source = make_pdf(tmp_path)
    pages = [result(block())]
    pipeline = Mock()
    pipeline.predict.return_value = pages

    def reconstruct(results, **kwargs):
        assert results[0]["parsing_res_list"] == []
        return results

    pipeline.restructure_pages.side_effect = reconstruct
    engine = PaddleVLEngine(ExtractionConfig(watermark_mode=WatermarkMode.FILTER))
    engine._pipeline = pipeline
    document = engine.extract(source)
    assert document.watermark_report[0]["decision"] == "excluded"
    pipeline.predict.assert_called_once()


def test_configuration_is_optional_and_rejects_unsupported_engine():
    assert ExtractionConfig().watermark_mode == WatermarkMode.OFF
    config = ExtractionConfig(
        watermark_mode=WatermarkMode.REPORT, profile=Profile.ROBUST
    )
    assert config.to_paddle_config().watermark_mode == WatermarkMode.REPORT
    with pytest.raises(ValueError, match="applies only to paddle-vl"):
        ExtractionConfig(
            engine=DocumentEngineName.DOCLING, watermark_mode=WatermarkMode.FILTER
        )


def test_local_jee_background_regression():
    """Optional local fixture: verify the identified empty area without OCR."""
    root = Path(__file__).resolve().parents[2]
    source = next(
        (
            path
            for path in (root / "pdf/jee.pdf", root / "pdfs/jee.pdf")
            if path.exists()
        ),
        None,
    )
    if source is None:
        pytest.skip("Local JEE PDF is not available")
    page = result(block(content=table("mathongo"), bbox=(80, 1000, 1100, 1480)))
    page.update(page_index=13, width=1190, height=1684)
    report = filter_watermarks([page], source, WatermarkMode.REPORT)
    assert report[0]["decision"] == "would_exclude"
    assert len(page["parsing_res_list"]) == 1
