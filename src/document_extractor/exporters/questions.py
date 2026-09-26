"""Conservative automatic presentation of exam tables. Never mutate OCR results."""

import re
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser

from document_extractor.exporters.math import (
    INLINE_MATH,
    is_math_start,
    normalize_inline_math,
    visible_prose,
)

_NUMBER = re.compile(r"^(\d{1,3})\.(?:\s|\([A-D]\)|$)")
_CHOICE = re.compile(r"\(([A-D])\)")


@dataclass
class QuestionLayout:
    markdown: str
    warnings: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)


@dataclass
class _Cell:
    content: str = ""
    rowspan: int = 1


class _Rows(HTMLParser):
    """Read outer rows, retaining cell markup including nested data tables."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.rows: list[list[_Cell]] = []
        self.cell: _Cell | None = None
        self.depth = 0
        self.in_row = False
        self.valid = True
        self.tables = 0

    def handle_starttag(self, tag, attrs):
        raw = self.get_starttag_text() or ""
        if tag == "table":
            self.depth += 1
            if self.depth == 1:
                self.tables += 1
                return
        if self.cell is not None:
            if self.depth == 1 and tag in {"td", "th", "tr"}:
                self.valid = False
            self.cell.content += raw
        elif self.depth == 1 and tag == "tr":
            if self.in_row:
                self.valid = False
            self.in_row = True
            self.rows.append([])
        elif self.depth == 1 and tag in {"td", "th"} and self.in_row:
            try:
                span = dict(attrs).get("rowspan", "1")
                if span is None:
                    raise ValueError("rowspan requires a value")
                self.cell = _Cell(rowspan=int(span))
                if self.cell.rowspan < 1:
                    self.valid = False
            except (TypeError, ValueError):
                self.valid = False
                self.cell = _Cell()
            self.rows[-1].append(self.cell)
        elif tag not in {"thead", "tbody", "tfoot"}:
            self.valid = False

    def handle_startendtag(self, tag, attrs):
        if self.cell is not None:
            self.cell.content += self.get_starttag_text() or ""
        else:
            self.valid = False

    def handle_endtag(self, tag):
        if self.depth == 1 and tag in {"td", "th"}:
            if self.cell is None:
                self.valid = False
            self.cell = None
        elif self.depth == 1 and tag == "tr":
            if self.cell is not None or not self.in_row:
                self.valid = False
            self.in_row = False
        elif self.cell is not None:
            self.cell.content += f"</{tag}>"
        elif tag not in {"table", "thead", "tbody", "tfoot"}:
            self.valid = False
        if tag == "table":
            self.depth -= 1

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.content += data
        elif data.strip():
            self.valid = False

    def handle_entityref(self, name):
        self.handle_data(f"&{name};")

    def handle_charref(self, name):
        self.handle_data(f"&#{name};")

    def handle_comment(self, data):
        self.valid = False


def format_question_text(text: str) -> QuestionLayout:
    """Format complete choice groups outside math; report ambiguous equations."""
    warnings = []
    visible = visible_prose(text)
    if visible is None:
        return QuestionLayout(
            text, ["Unclosed math/code delimiter; formatting skipped."]
        )

    # Repair presentation whitespace, never LaTeX commands such as \nu or \neq.
    edits = [
        (m.start(), m.end(), "\n\n")
        for m in re.finditer(r"(?<!\\)\\n(?![a-z])", visible)
    ]
    for start, end, replacement in reversed(edits):
        text = text[:start] + replacement + text[end:]

    # Spaces just inside $ delimiters prevent rendering in some Markdown viewers.
    # Keep every character of the mathematical expression itself.
    text = normalize_inline_math(text)
    text, repairs = _repair_option_boundary(text)
    visible = visible_prose(text)
    assert visible is not None
    matches = _option_matches(visible)
    for formula in INLINE_MATH.finditer(text):
        if not is_math_start(text, formula.start()):
            continue
        if _CHOICE.search(formula[1]) and len(matches) >= 3:
            warnings.append(
                "Possible answer label inside an equation; check the source crop."
            )
            break
    # Handle repeated A-D groups independently, e.g. accessible alternatives.
    groups = [matches[i : i + 4] for i in range(0, len(matches), 4)]
    if len({match[1] for match in matches}) < 3:
        return QuestionLayout(text, warnings, repairs)
    if any(
        len(g) != 4 or sorted(m[1] for m in g) != list("ABCD") or g[0][1] != "A"
        for g in groups
    ):
        warnings.append(
            "Ambiguous or incomplete answer group; option spacing unchanged."
        )
        return QuestionLayout(text, warnings, repairs)
    for match in reversed(matches):
        text = text[: match.start()].rstrip() + "\n\n" + text[match.start() :]
    return QuestionLayout(text.strip(), warnings, repairs)


def _option_matches(visible: str) -> list[re.Match[str]]:
    return [
        match
        for match in _CHOICE.finditer(visible)
        if not re.search(
            r"(?:assertion|reason)\s*$", visible[: match.start()], re.IGNORECASE
        )
        and not visible[match.end() :].startswith(":")
    ]


def _repair_option_boundary(text: str) -> tuple[str, list[str]]:
    """Move a swallowed label across a delimiter only with a complete A-D group.

    This is a presentation heuristic, recorded in the report for source review.
    It never changes mathematical symbols or synthesizes missing answer text.
    """
    visible = visible_prose(text)
    if visible is None:
        return text, []
    outside = _option_matches(visible)
    if len(outside) != 3 or outside[0][1] != "A":
        return text, []
    candidates = []
    for formula in INLINE_MATH.finditer(text):
        if not is_math_start(text, formula.start()):
            continue
        # Restrict recovery to the observed failure: a trailing choice label
        # after a closed parenthesis, followed immediately by another formula.
        label = re.search(r"(?<=\))\(([B-D])\)$", formula[1])
        following = text[formula.end() :].lstrip()
        next_formula = INLINE_MATH.match(following)
        if label and "=" in formula[1] and next_formula and "=" in next_formula[1]:
            candidates.append((formula, label))
    if len(candidates) != 1:
        return text, []
    formula, label = candidates[0]
    if outside[0].start() >= formula.start() or sorted(
        [m[1] for m in outside] + [label[1]]
    ) != list("ABCD"):
        return text, []
    replacement = "$" + formula[1][: label.start()] + "$ " + label[0]
    repaired = text[: formula.start()] + replacement + text[formula.end() :]
    return repaired, [
        f"Moved option {label[0]} outside the preceding math delimiter; verify against source."
    ]


def _cell_text(content: str) -> str | None:
    # Nested tables must retain their structure, not become question prose.
    if re.search(r"<(?!/?(?:img|br)\b)[A-Za-z/]", content, re.IGNORECASE):
        return None
    content = re.sub(r"<br\s*/?>", "\n\n", content, flags=re.IGNORECASE)
    # Keep image tags as-is: the exporter owns asset resolution and escaping.
    parts = re.split(r"(<img\b[^>]*>)", content, flags=re.IGNORECASE)
    return "".join(
        "\n\n" + part + "\n\n" if part.lower().startswith("<img") else unescape(part)
        for part in parts
    ).strip()


def render_question_table(html: str) -> QuestionLayout | None:
    """Require exam headers plus numbered prose rows. Unknown tables stay intact."""
    parser = _Rows()
    parser.feed(html)
    parser.close()
    if (
        not parser.valid
        or parser.depth
        or parser.cell
        or parser.in_row
        or parser.tables != 1
    ):
        return None
    rows = parser.rows
    header = any(
        len(row) == 3
        and [cell.content.strip().casefold() for cell in row]
        in (["q.no.", "questions", "marks"], ["q. no.", "questions", "marks"])
        for row in rows
    )
    section = any(
        len(row) == 1
        and re.search(r"section\s+[a-e]", row[0].content, re.IGNORECASE)
        and "questions" in row[0].content.lower()
        and "marks" in row[0].content.lower()
        for row in rows
    )
    numbered = [
        row
        for row in rows
        if len(row) in {2, 3} and _NUMBER.match(row[0].content.strip())
    ]
    if not (header or section) or len(numbered) < 2:
        return None
    numbers = []
    for row in numbered:
        match = _NUMBER.match(row[0].content.strip())
        assert match is not None
        numbers.append(int(match[1]))
    if numbers != sorted(set(numbers)):
        return None
    if any(len(row[1].content.split()) < 6 for row in numbered):
        return None

    paragraphs = []
    warnings = []
    repairs = []
    continuation = 0
    for row in rows:
        if not row:
            return None
        values = [_cell_text(cell.content) for cell in row]
        if any(value is None for value in values):
            return None
        cells = [value for value in values if value is not None]
        if (
            len(cells) == 3
            and cells[1].casefold() == "questions"
            and cells[2].casefold() == "marks"
        ):
            continue
        if len(cells) in {2, 3} and _NUMBER.match(cells[0]):
            if continuation:
                return None
            if len(cells) == 3 and not re.fullmatch(r"[\d\s\\n]*", cells[2]):
                return None
            continuation = row[0].rowspan - 1
            if row[1].rowspan != 1 or (
                len(row) == 3 and row[2].rowspan != row[0].rowspan
            ):
                return None
            result = format_question_text(cells[1])
            label = cells[0].replace(r"\n", " ")
            paragraphs.append(f"**{label}** {result.markdown}")
            if len(cells) == 3 and cells[2]:
                paragraphs.append("Marks: " + cells[2].replace(r"\n", ", "))
            warnings.extend(
                f"Question {label}: {warning}" for warning in result.warnings
            )
            repairs.extend(f"Question {label}: {repair}" for repair in result.repairs)
        elif len(cells) == 1 and (continuation or "section" in cells[0].lower()):
            result = format_question_text(cells[0])
            paragraphs.append(result.markdown)
            warnings.extend(result.warnings)
            repairs.extend(result.repairs)
            if continuation:
                continuation -= 1
        elif len(cells) == 3 and not cells[0] and not cells[2]:
            result = format_question_text(cells[1])
            paragraphs.append(result.markdown)
            warnings.extend(result.warnings)
            repairs.extend(result.repairs)
        else:
            return None
    if continuation:
        return None
    return QuestionLayout("\n\n".join(paragraphs), warnings, repairs)
