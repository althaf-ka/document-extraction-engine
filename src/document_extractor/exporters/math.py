"""Render recognized inline LaTeX without changing the extracted expressions."""

import re

from document_extractor.option_normalization import _outside_math

INLINE_MATH = re.compile(r"(?<![\\$])\$(?!\$)((?:\\.|[^$\\])*)(?<!\\)\$(?!\$)")


def _mask_tags(text: str) -> str:
    return re.sub(r"<[^>]*>", lambda match: " " * len(match[0]), text)


def visible_prose(text: str) -> str | None:
    return _outside_math(_mask_tags(text))


def is_math_start(text: str, start: int) -> bool:
    masked = _mask_tags(text)
    return masked[start] == "$" and _outside_math(masked[:start]) is not None


def inline_math_spans(text: str) -> list[re.Match[str]]:
    """Exclude code, HTML attributes, display math and unbalanced delimiters."""
    if visible_prose(text) is None:
        return []
    return [
        match
        for match in INLINE_MATH.finditer(text)
        if is_math_start(text, match.start())
        # A price pair such as '$5 and $10' is not a math expression.
        and not (
            text[match.end() : match.end() + 1].isdigit()
            and re.fullmatch(r"\d[\d.,]*\s+[A-Za-z\s]+", match[1])
        )
    ]


def normalize_inline_math(text: str) -> str:
    """Provide valid Markdown delimiter boundaries, including '$...$C'."""
    for match in reversed(inline_math_spans(text)):
        before = text[match.start() - 1 : match.start()] if match.start() else ""
        after = text[match.end() : match.end() + 1]
        left = " " if before and (before.isalnum() or before == "_") else ""
        right = " " if after and (after.isalnum() or after == "_") else ""
        replacement = left + "$" + match[1].strip() + "$" + right
        text = text[: match.start()] + replacement + text[match.end() :]
    return text


def _escape_cell_text(text: str) -> str:
    for char in ("\\", "|", "*", "_", "`", "[", "]", "$"):
        text = text.replace(char, "\\" + char)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_math_cell(text: str) -> str | None:
    """Escape table prose, retaining recognized math. None requires HTML fallback."""
    text = normalize_inline_math(text)
    spans = inline_math_spans(text)
    parts = []
    offset = 0
    for match in spans:
        # Pipe characters and multiline expressions need more than a simple
        # Markdown cell. Preserve the original HTML instead of altering LaTeX.
        if "|" in match[1] or "\n" in match[1]:
            return None
        parts.extend((_escape_cell_text(text[offset : match.start()]), match[0]))
        offset = match.end()
    parts.append(_escape_cell_text(text[offset:]))
    return "".join(parts)
