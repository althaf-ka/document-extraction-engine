"""Normalize parser output without choosing an export format."""

from html.parser import HTMLParser

from document_extractor.models import FormulaPlacement, TableContent


def normalize_formula(content: str) -> str:
    content = content.strip()
    for opening, closing in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$")):
        if content.startswith(opening) and content.endswith(closing):
            return content[len(opening) : -len(closing)].strip()
    return content


def formula_placement(kind: str, bbox, neighbors=()) -> FormulaPlacement:
    if kind == "inline_formula":
        return FormulaPlacement.INLINE
    if kind == "display_formula":
        return FormulaPlacement.DISPLAY
    # An unclassified formula is inline only with evidence of adjacent text
    # on the same line. Missing geometry defaults to a standalone equation.
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        for neighbor in neighbors:
            if neighbor.label != "text" or neighbor.bbox is None:
                continue
            left, top, right, bottom = neighbor.bbox
            overlap = min(y2, bottom) - max(y1, top)
            if (
                height > 0
                and bottom > top
                and overlap >= 0.8 * max(height, bottom - top)
            ):
                gap = max(left - x2, x1 - right)
                if 0 <= gap <= 2 * height:
                    return FormulaPlacement.INLINE
    return FormulaPlacement.DISPLAY


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.headers: list[list[bool]] = []
        self.cell: list[str] | None = None
        self.in_row = False
        self.tables = 0
        self.depth = 0
        self.complex = False
        self.merged = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "table":
            self.tables += 1
            self.depth += 1
        elif tag == "tr":
            if self.in_row or self.depth != 1:
                self.complex = True
            self.in_row = True
            self.rows.append([])
            self.headers.append([])
        elif tag in {"td", "th"}:
            if self.cell is not None or not self.in_row:
                self.complex = True
                return
            self.cell = []
            self.headers[-1].append(tag == "th")
            for key in ("rowspan", "colspan"):
                if key in attributes and attributes[key] != "1":
                    self.merged = True
            if any(key not in {"rowspan", "colspan"} for key in attributes):
                self.complex = True
        elif tag not in {"thead", "tbody", "tfoot"}:
            # Preserve rich cells, captions, nested tables and line breaks as HTML.
            self.complex = True

    def handle_endtag(self, tag):
        if tag in {"td", "th"}:
            if self.cell is None or not self.rows:
                self.complex = True
                return
            self.rows[-1].append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr":
            if self.cell is not None:
                self.complex = True
            self.in_row = False
        elif tag == "table":
            self.depth -= 1

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)
        elif data.strip():
            self.complex = True


def normalize_table(html: str) -> TableContent:
    parser = _TableParser()
    parser.feed(html)
    parser.close()
    rectangular = bool(parser.rows and parser.rows[0]) and all(
        len(row) == len(parser.rows[0]) for row in parser.rows
    )
    header = bool(parser.headers and parser.headers[0] and all(parser.headers[0]))
    unusual_headers = any(any(row) for row in parser.headers[1:]) or (
        bool(parser.headers) and any(parser.headers[0]) and not header
    )
    safe = (
        rectangular
        and not parser.complex
        and not parser.merged
        and parser.tables == 1
        and parser.depth == 0
        and parser.cell is None
        and not parser.in_row
        and not unusual_headers
    )
    rows = tuple(tuple(row) for row in parser.rows) if safe else None
    return TableContent(html, parser.merged, rows, header)
