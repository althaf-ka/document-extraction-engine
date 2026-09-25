from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from document_extractor.engines.paddle_vl import adapt_results, qualify_image_paths
from document_extractor.exporters.document import DocumentExporter
from document_extractor.html_images import rewrite_image_sources


def table(html):
    return SimpleNamespace(label="table", content=html, bbox=[0, 0, 10, 10], image=None)


def page(color, html):
    return {
        "parsing_res_list": [table(html)],
        "imgs_in_doc": [
            {"path": "imgs/crop.jpg", "img": Image.new("RGB", (8, 12), color)}
        ],
    }


def test_embedded_images_resolve_in_markdown_and_tables_across_pages(tmp_path):
    html = '<table><tr><td rowspan="2"><img src="imgs/crop.jpg" alt="Image"" /></td></tr><tr><td><img src="imgs/crop.jpg"></td></tr></table>'
    results = [page("red", html), page("blue", html)]
    # Mix a standalone figure with embedded crops to check shared numbering.
    results[0]["parsing_res_list"].insert(
        0,
        SimpleNamespace(
            label="image",
            content="",
            bbox=[0, 0, 5, 5],
            image={"img": Image.new("RGB", (5, 5), "green")},
        ),
    )
    document = adapt_results(results, Path("sample.pdf"))
    bundle = DocumentExporter().export(document, Path("sample.pdf"), tmp_path)
    markdown = bundle.document_path.read_text()
    assert "![Image 1](images/img-0.png)" in markdown
    assert markdown.count('src="images/img-1.png"') == 2
    assert markdown.count('src="images/img-2.png"') == 2
    assert 'alt="Image""' not in markdown
    assert markdown.count('rowspan="2"') == 2
    assert len(list(bundle.images_dir.iterdir())) == 3
    for number, color in [(1, (255, 0, 0)), (2, (0, 0, 255))]:
        artifact = (bundle.tables_dir / f"tbl-{number - 1}.html").read_text()
        assert artifact.count(f'src="../images/img-{number}.png"') == 2
        with Image.open(bundle.images_dir / f"img-{number}.png") as image:
            assert image.size == (8, 12)
            assert image.getpixel((0, 0)) == color


def test_table_moved_across_pages_keeps_its_original_crop(tmp_path):
    html = '<table><tr><td><img src="imgs/crop.jpg"></td></tr></table>'
    results = [page("red", html), page("blue", html)]
    qualify_image_paths(results)
    # Simulate reconstruction moving the second page's table onto page one.
    results[0]["parsing_res_list"].extend(results[1]["parsing_res_list"])
    results[1]["parsing_res_list"] = []
    document = adapt_results(results, Path("sample.pdf"))
    bundle = DocumentExporter().export(document, Path("sample.pdf"), tmp_path)
    with Image.open(bundle.images_dir / "img-1.png") as image:
        assert image.getpixel((0, 0)) == (0, 0, 255)


def test_missing_embedded_pixels_report_error(tmp_path):
    results = [
        {
            "parsing_res_list": [
                table('<table><tr><td><img src="imgs/missing.jpg"></td></tr></table>')
            ]
        }
    ]
    document = adapt_results(results, Path("sample.pdf"))
    with pytest.raises(ValueError, match="Embedded image has no pixels"):
        DocumentExporter().export(document, Path("sample.pdf"), tmp_path)


def test_image_rewrite_preserves_surrounding_html_and_escapes_attributes():
    html = "<table>\n<tr><td>A &amp; B<img SRC='a&amp;b.jpg' alt='A &quot;quote&quot;' /></td></tr></table>"
    seen = []

    def resolve(source):
        seen.append(source)
        return "images/img-0.png"

    rewritten = rewrite_image_sources(html, resolve)
    assert seen == ["a&b.jpg"]
    assert (
        rewritten
        == '<table>\n<tr><td>A &amp; B<img src="images/img-0.png" alt="A &quot;quote&quot;"></td></tr></table>'
    )
