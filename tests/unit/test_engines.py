import sys
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from document_extractor.config import (
    DocumentEngineName,
    ExtractionConfig,
    InferenceBackend,
    PaddleVLConfig,
    Profile,
)
from document_extractor.engines.paddle_vl import PaddleVLEngine, adapt_results
from document_extractor.exporters.document import DocumentExporter


def test_experiment_overrides_reach_paddle_without_changing_baseline(monkeypatch):
    baseline = PaddleVLConfig()
    experiment = replace(
        baseline,
        layout_shape_mode="rect",
        layout_unclip_ratio=(1.0, 1.2),
        layout_merge_bboxes_mode="large",
    )
    pipeline = Mock()
    pipeline.predict.return_value = [{"parsing_res_list": []}]
    constructor = Mock(return_value=pipeline)
    monkeypatch.setitem(
        sys.modules, "paddleocr", SimpleNamespace(PaddleOCRVL=constructor)
    )
    PaddleVLEngine(experiment).extract(Path("sample.png"))
    assert constructor.call_args.kwargs["layout_unclip_ratio"] == [1.0, 1.2]
    assert constructor.call_args.kwargs["layout_merge_bboxes_mode"] == "large"
    pipeline.predict.assert_called_once_with(
        input="sample.png", layout_shape_mode="rect"
    )
    assert baseline.layout_shape_mode == "auto"
    assert baseline.layout_unclip_ratio == (1.0, 1.0)


def block(label: str, content: str = "", image=None):
    return SimpleNamespace(
        label=label, content=content, bbox=[0, 0, 10, 10], image=image
    )


@pytest.mark.parametrize(
    "backend,expected",
    [("local", None), ("llama-cpp", "llama-cpp-server"), ("mlx", "mlx-vlm-server")],
)
def test_runtime_selection_and_model_reuse(monkeypatch, backend, expected):
    pipeline = Mock()
    pipeline.predict.return_value = [{"page_index": None, "parsing_res_list": []}]
    constructor = Mock(return_value=pipeline)
    monkeypatch.setitem(
        sys.modules, "paddleocr", SimpleNamespace(PaddleOCRVL=constructor)
    )
    config = ExtractionConfig(
        backend=InferenceBackend(backend),
        server_url="http://localhost:8111" if expected else None,
        profile=Profile.ROBUST,
        max_concurrency=2,
        layout_threshold=0.6,
    )
    engine = PaddleVLEngine(config)
    engine.extract(Path("first.png"))
    engine.extract(Path("second.png"))
    constructor.assert_called_once()
    options = constructor.call_args.kwargs
    assert options["pipeline_version"] == "v1.6"
    assert options["use_doc_unwarping"] is True
    assert options["vl_rec_max_concurrency"] == 2
    assert options["layout_threshold"] == 0.6
    assert options["format_block_content"] is True
    assert options["merge_layout_blocks"] is True
    assert options["use_queues"] is True
    assert "aside_text" in options["markdown_ignore_labels"]
    pipeline.predict.assert_called_with(input="second.png", layout_shape_mode="auto")
    assert options.get("vl_rec_backend") == expected
    if backend == "mlx":
        assert options["vl_rec_api_model_name"] == "PaddlePaddle/PaddleOCR-VL-1.6"


def test_pdf_reconstruction(monkeypatch):
    pipeline = Mock()
    pages = [{"page_index": 0, "parsing_res_list": []}]
    pipeline.predict.return_value = pages
    pipeline.restructure_pages.return_value = pages
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCRVL=Mock(return_value=pipeline)),
    )
    PaddleVLEngine(ExtractionConfig()).extract(Path("sample.pdf"))
    pipeline.restructure_pages.assert_called_once_with(
        pages, merge_tables=True, relevel_titles=True, concatenate_pages=False
    )


def test_paddle_progress_only_counts_yielded_primary_pages(monkeypatch):
    events = []
    pipeline = Mock()

    def predict(**kwargs):
        assert [
            (event.current, event.completed)
            for event in events
            if event.stage == "inference"
        ] == [(0, False)]
        yield {"page_index": 0, "page_count": 2, "parsing_res_list": []}
        assert [
            (event.current, event.completed)
            for event in events
            if event.stage == "inference"
        ] == [(0, False), (1, False)]
        yield {"page_index": 1, "page_count": 2, "parsing_res_list": []}

    pipeline.predict.side_effect = predict
    pipeline.restructure_pages.side_effect = lambda pages, **kwargs: pages
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCRVL=Mock(return_value=pipeline)),
    )
    PaddleVLEngine(ExtractionConfig(), on_progress=events.append).extract(
        Path("sample.pdf")
    )
    inference = [event for event in events if event.stage == "inference"]
    assert [(event.current, event.total, event.completed) for event in inference] == [
        (0, None, False),
        (1, 2, False),
        (2, 2, False),
        (2, 2, True),
    ]
    assert [event.stage for event in events if event.completed] == [
        "loading",
        "inference",
        "recovery",
        "reconstruction",
    ]
    pipeline.predict.assert_called_once_with(
        input="sample.pdf", layout_shape_mode="auto"
    )


def test_failed_prediction_never_reports_completed_inference():
    events = []
    engine = PaddleVLEngine(ExtractionConfig(), on_progress=events.append)
    engine._pipeline = Mock()
    engine._pipeline.predict.side_effect = RuntimeError("server disconnected")
    with pytest.raises(RuntimeError, match="disconnected"):
        engine.extract(Path("sample.png"))
    assert not any(event.stage == "inference" and event.completed for event in events)


def test_order_and_artifacts_across_pages(tmp_path):
    crop = Image.new("RGB", (10, 10), "red")
    results = [
        {
            "page_index": 0,
            "parsing_res_list": [
                block("doc_title", "# Heading"),
                block("table", "<table><tr><td>42</td></tr></table>"),
                block("image", image={"img": crop}),
            ],
        },
        {
            "page_index": 1,
            "parsing_res_list": [
                block("formula", "x^2"),
                block("image", image={"img": crop}),
                block("text", "End"),
            ],
        },
    ]
    document = adapt_results(results, Path("sample.pdf"))
    assert [p.number for p in document.pages] == [1, 2]
    assert document.pages[0].elements[1].bbox == (0, 0, 10, 10)
    bundle = DocumentExporter().export(document, Path("sample.pdf"), tmp_path)
    markdown = bundle.document_path.read_text()
    assert (
        markdown.index("# Heading")
        < markdown.index("| 42 |")
        < markdown.index("images/img-0.png")
        < markdown.index("x^2")
        < markdown.index("images/img-1.png")
        < markdown.index("End")
    )
    assert "42" in (bundle.tables_dir / "tbl-0.html").read_text()
    assert Image.open(BytesIO((bundle.images_dir / "img-1.png").read_bytes())).size == (
        10,
        10,
    )
    with pytest.raises(FileExistsError):
        DocumentExporter().export(document, Path("sample.pdf"), tmp_path)
    assert bundle.document_path.read_text() == markdown


@pytest.mark.parametrize(
    "options",
    [
        {"backend": InferenceBackend.MLX},
        {"server_url": "http://localhost"},
        {"backend": InferenceBackend.LLAMA_CPP, "server_url": "file:///tmp/model"},
        {"engine": DocumentEngineName.DOCLING, "profile": Profile.ROBUST},
        {
            "engine": DocumentEngineName.DOCLING,
            "backend": InferenceBackend.MLX,
            "server_url": "http://localhost",
        },
    ],
)
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        ExtractionConfig(**options)


@pytest.mark.parametrize(
    "options",
    [
        {"max_concurrency": 0},
        {"max_concurrency": 5},
        {"layout_threshold": 0.1},
        {"layout_threshold": 0.9},
    ],
)
def test_vlm_tuning_rejects_out_of_range_values(options):
    with pytest.raises(ValueError):
        ExtractionConfig(**options)


def test_paddle_preserves_reconstructed_heading_levels():
    heading = block("paragraph_title", "Subsection")
    heading.title_level = 2
    document = adapt_results([{"parsing_res_list": [heading]}], Path("paper.pdf"))
    assert document.pages[0].elements[0].content == "### Subsection"


def test_missing_paddle_image_fails_instead_of_dropping_figure():
    with pytest.raises(RuntimeError, match="without pixels"):
        adapt_results([{"parsing_res_list": [block("image")]}], Path("paper.pdf"))


def test_docling_adapter_and_converter_reuse(monkeypatch):
    from document_extractor.engines.docling import DoclingEngine

    class Table(SimpleNamespace):
        def export_to_html(self, doc):
            return "<table><tr><td>Value</td></tr></table>"

    class Picture(SimpleNamespace):
        def get_image(self, doc):
            return Image.new("RGB", (4, 4))

    bbox = Mock()
    bbox.to_top_left_origin.return_value = SimpleNamespace(l=1, t=2, r=3, b=4)
    provenance = [SimpleNamespace(page_no=1, bbox=bbox)]
    items = [
        SimpleNamespace(
            label=SimpleNamespace(value="section_header"),
            text="Heading",
            level=2,
            prov=provenance,
        ),
        Table(label=SimpleNamespace(value="table"), prov=provenance),
        Picture(label=SimpleNamespace(value="picture"), prov=provenance),
        SimpleNamespace(
            label=SimpleNamespace(value="formula"), text="x=1", prov=provenance
        ),
    ]
    document = SimpleNamespace(
        pages={1: SimpleNamespace(size=SimpleNamespace(height=100))},
        iterate_items=lambda: [(item, 0) for item in items],
    )
    converter = Mock()
    converter.convert.return_value = SimpleNamespace(document=document)
    constructor = Mock(return_value=converter)
    image_options = Mock()
    monkeypatch.setitem(
        sys.modules,
        "docling.datamodel.base_models",
        SimpleNamespace(InputFormat=SimpleNamespace(PDF="pdf", IMAGE="image")),
    )
    monkeypatch.setitem(
        sys.modules,
        "docling.datamodel.pipeline_options",
        SimpleNamespace(PdfPipelineOptions=SimpleNamespace),
    )
    monkeypatch.setitem(
        sys.modules,
        "docling.document_converter",
        SimpleNamespace(
            DocumentConverter=constructor,
            PdfFormatOption=Mock(),
            ImageFormatOption=image_options,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "docling_core.types.doc",
        SimpleNamespace(TableItem=Table, PictureItem=Picture),
    )
    engine = DoclingEngine()
    result = engine.extract(Path("paper.pdf"))
    engine.extract(Path("photo.png"))
    constructor.assert_called_once()
    image_options.assert_called_once()
    elements = result.pages[0].elements
    assert [element.kind for element in elements] == [
        "section_header",
        "table",
        "image",
        "formula",
    ]
    assert elements[0].content == "## Heading"
    assert elements[0].bbox == (1, 2, 3, 4)
    image_png = elements[2].image_png
    assert image_png is not None
    assert image_png.startswith(b"\x89PNG")
    bbox.to_top_left_origin.assert_called_with(page_height=100)
