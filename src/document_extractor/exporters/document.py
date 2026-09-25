from pathlib import Path

from document_extractor.config import MARKDOWN_IGNORE_LABELS
from document_extractor.exporters.bundle import OutputBundle
from document_extractor.models import FormulaPlacement, NormalizedDocument, TableContent


def render_table(table: TableContent) -> str:
    if table.rows is None or table.has_merged_cells:
        return table.html

    def row(cells):
        # Preserve literal cell text rather than interpreting it as Markdown.
        escaped = []
        for cell in cells:
            for char in ("\\", "|", "*", "_", "`", "[", "]", "$"):
                cell = cell.replace(char, "\\" + char)
            cell = cell.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            escaped.append(cell)
        return "| " + " | ".join(escaped) + " |"

    rows = list(table.rows)
    header = rows.pop(0) if table.has_header else ("",) * len(rows[0])
    return "\n".join(
        [row(header), row(("---",) * len(header)), *(row(cells) for cells in rows)]
    )


class DocumentExporter:
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
        for page in document.pages:
            previous_inline = False
            previous_text = False
            for element in page.elements:
                if element.kind in MARKDOWN_IGNORE_LABELS:
                    previous_inline = previous_text = False
                    continue
                content = element.content.strip()
                if element.kind == "table":
                    name = f"tbl-{table_count}.html"
                    html = element.table.html if element.table else element.content
                    (bundle.tables_dir / name).write_text(html, encoding="utf-8")
                    sections.append(
                        render_table(element.table) if element.table else html
                    )
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
                        paragraphs = [group.question] if group.question else []
                        paragraphs.extend(
                            f"{option.label} {option.content}"
                            for option in group.options
                        )
                        sections.extend(paragraphs)
                        previous_inline = previous_text = False
                        continue
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
        return bundle
