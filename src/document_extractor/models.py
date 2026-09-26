from dataclasses import dataclass, field
from enum import StrEnum


class FormulaPlacement(StrEnum):
    INLINE = "inline"
    DISPLAY = "display"


@dataclass(frozen=True)
class DocumentElement:
    """An ordered block. Coordinates use the source page's top-left origin."""

    kind: str
    content: str = ""
    bbox: tuple[float, ...] | None = None
    order: int | None = None
    image_png: bytes | None = None
    formula_placement: FormulaPlacement | None = None
    table: "TableContent | None" = None
    option_group: "OptionGroup | None" = None


@dataclass(frozen=True)
class AnswerOption:
    label: str
    content: str


@dataclass(frozen=True)
class OptionGroup:
    question: str
    options: tuple[AnswerOption, ...]


@dataclass(frozen=True)
class TableContent:
    """Original HTML and losslessly convertible rectangular cell text, if any."""

    html: str
    has_merged_cells: bool
    rows: tuple[tuple[str, ...], ...] | None = None
    has_header: bool = False


@dataclass
class Page:
    number: int
    elements: list[DocumentElement] = field(default_factory=list)


@dataclass
class NormalizedDocument:
    pages: list[Page]
    source: str
    engine: str | None = None
    debug_results: list[dict] | None = None
    image_assets: dict[str, bytes] = field(default_factory=dict)
    watermark_report: list[dict] = field(default_factory=list)
