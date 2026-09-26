from pathlib import Path

from document_extractor.config import MARKDOWN_IGNORE_LABELS
from document_extractor.exporters.bundle import OutputBundle
from document_extractor.exporters.debug import write_json
from document_extractor.exporters.math import normalize_inline_math, render_math_cell
from document_extractor.exporters.questions import render_question_table
from document_extractor.html_images import rewrite_image_sources
from document_extractor.models import FormulaPlacement, NormalizedDocument, TableContent


def render_table(table: TableContent) -> str:
    if table.rows is None or table.has_merged_cells:
        return table.html

    def row(cells):
        return "| " + " | ".join(cells) + " |"

    rows = []
    for cells in table.rows:
        rendered = [render_math_cell(cell) for cell in cells]
        if any(cell is None for cell in rendered):
            return table.html
        rows.append(tuple(cell for cell in rendered if cell is not None))
    header = rows.pop(0) if table.has_header else ("",) * len(rows[0])
    return "\n".join(
        [row(header), row(("---",) * len(header)), *(row(cells) for cells in rows)]
    )


class DocumentExporter:
    def __init__(
        self, *, question_layout: bool = True, export_layout_report: bool = False
    ) -> None:
        self.question_layout = question_layout
        self.export_layout_report = export_layout_report

    def export(
        self, document: NormalizedDocument, source: Path, output_root: Path
    ) -> OutputBundle:
        bundle = OutputBundle.for_input(source, output_root)
        # Refuse accidental replacement, including stale images from a prior run.
        if bundle.root_dir.exists() and any(bundle.root_dir.iterdir()):
            raise FileExistsError(
                f"Output already exists: {bundle.root_dir}. Choose another --output directory."
            )
        bundle.create()
        sections = []
        table_count = image_count = 0
        exported_assets: dict[str, str] = {}
        layout_report: list[dict] = []

        def resolve_image(source: str, prefix: str) -> str:
            nonlocal image_count
            if source not in document.image_assets:
                raise ValueError(f"Embedded image has no pixels: {source}")
            if source not in exported_assets:
                name = f"img-{image_count}.png"
                (bundle.images_dir / name).write_bytes(document.image_assets[source])
                exported_assets[source] = name
                image_count += 1
            return prefix + exported_assets[source]

        for page in document.pages:
            previous_inline = False
            previous_text = False
            for element in page.elements:
                if element.kind in MARKDOWN_IGNORE_LABELS:
                    previous_inline = previous_text = False
                    continue
                content = element.content.strip()
                if element.kind != "table" and "<img" in content.lower():
                    content = rewrite_image_sources(
                        content, lambda src: resolve_image(src, "images/")
                    )
                if element.kind == "table":
                    name = f"tbl-{table_count}.html"
                    html = element.table.html if element.table else element.content
                    artifact_html = rewrite_image_sources(
                        html, lambda src: resolve_image(src, "../images/")
                    )
                    (bundle.tables_dir / name).write_text(
                        artifact_html, encoding="utf-8"
                    )
                    # Resolve assets while the content is still HTML. Decoding
                    # entities into LaTeX comparisons during question rendering
                    # can make a later HTML parser swallow an image tag.
                    display_html = rewrite_image_sources(
                        html, lambda src: resolve_image(src, "images/")
                    )
                    question = (
                        render_question_table(display_html)
                        if self.question_layout
                        else None
                    )
                    if self.question_layout:
                        layout_report.append(
                            {
                                "page": page.number,
                                "table": name,
                                "converted": question is not None,
                                "warnings": question.warnings if question else [],
                                "repairs": question.repairs if question else [],
                            }
                        )
                    sections.append(
                        question.markdown
                        if question
                        else rewrite_image_sources(
                            render_table(element.table) if element.table else html,
                            lambda src: resolve_image(src, "images/"),
                        )
                    )
                    sections.append(f"[Table {table_count + 1}](tables/{name})")
                    table_count += 1
                elif element.image_png is not None:
                    name = f"img-{image_count}.png"
                    (bundle.images_dir / name).write_bytes(element.image_png)
                    sections.append(f"![Image {image_count + 1}](images/{name})")
                    if content:
                        sections.append(content)
                    image_count += 1
                elif content:
                    if element.option_group is not None:
                        group = element.option_group
                        paragraphs = (
                            [normalize_inline_math(group.question)]
                            if group.question
                            else []
                        )
                        paragraphs.extend(
                            f"{option.label} {normalize_inline_math(option.content)}"
                            for option in group.options
                        )
                        sections.extend(paragraphs)
                        previous_inline = previous_text = False
                        continue
                    if element.formula_placement is None:
                        content = normalize_inline_math(content)
                    if element.formula_placement is not None:
                        if element.formula_placement == FormulaPlacement.INLINE:
                            content = f"${content}$"
                        else:
                            content = f"$$\n{content}\n$$"
                    elif element.kind in {
                        "doc_title",
                        "paragraph_title",
                        "title",
                        "section_header",
                    } and not content.startswith("#"):
                        content = f"## {content}"
                    inline = element.formula_placement == FormulaPlacement.INLINE
                    text = element.kind == "text"
                    if sections and (
                        (inline and (previous_text or previous_inline))
                        or (text and previous_inline)
                    ):
                        separator = (
                            ""
                            if content.startswith((".", ",", ";", ":", "!", "?"))
                            else " "
                        )
                        sections[-1] += separator + content
                    else:
                        sections.append(content)
                previous_inline = element.formula_placement == FormulaPlacement.INLINE
                previous_text = element.kind == "text"
        bundle.document_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
        if self.question_layout and self.export_layout_report:
            write_json(
                bundle.root_dir / "debug" / "question-layout.json",
                {
                    "schema_version": 1,
                    "tables": layout_report,
                },
            )
        return bundle
