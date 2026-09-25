"""Recognize clearly labeled answer groups without modifying source text."""

import re
from dataclasses import replace

from document_extractor.models import AnswerOption, NormalizedDocument, OptionGroup

_QUESTION = re.compile(r"(?m)^\s*(?:Q(?:uestion)?\s*\d+\s*[.:])", re.IGNORECASE)
_OPTION = re.compile(r"(?<!\S)\(([1-9])\)(?=\s)")


def _outside_math(text: str) -> str | None:
    """Mask math/code spans while preserving character offsets. Fail closed."""
    masked = list(text)
    closing = None
    index = 0
    while index < len(text):
        if closing is not None:
            if text.startswith(closing, index):
                width = len(closing)
                masked[index : index + width] = " " * width
                index += width
                closing = None
            else:
                masked[index] = " "
                if text[index] == "\\" and index + 1 < len(text):
                    masked[index + 1] = " "
                    index += 1
                index += 1
            continue
        opening = next(
            (
                value
                for value in ("$$", r"\(", r"\[", "$", "`")
                if text.startswith(value, index)
            ),
            None,
        )
        if opening is not None:
            if opening == "`":
                end = index
                while end < len(text) and text[end] == "`":
                    end += 1
                opening = text[index:end]
            closing = {r"\(": r"\)", r"\[": r"\]"}.get(opening, opening)
            masked[index : index + len(opening)] = " " * len(opening)
            index += len(opening)
        elif text[index] == "\\":
            index += 2
        else:
            index += 1
    return "".join(masked) if closing is None else None


def recognize_options(
    content: str, follows_question: bool = False
) -> OptionGroup | None:
    # HTML has its own structure; do not split table cells or inline markup.
    if re.search(r"</?[A-Za-z][^>]*>", content):
        return None
    visible = _outside_math(content)
    if visible is None:
        return None
    questions = list(_QUESTION.finditer(visible))
    if len(questions) > 1:
        return None
    matches = list(_OPTION.finditer(visible))
    if len(matches) < 2:
        return None
    before_options = content[: matches[0].start()]
    prefix = before_options.strip()
    if not (
        (questions and questions[0].start() < matches[0].start())
        or (follows_question and not prefix)
    ):
        return None
    # Require a colon or a separate answer line, not parenthesized numbers
    # embedded in the question's prose.
    whitespace = before_options[len(before_options.rstrip()) :]
    if prefix and not (prefix.endswith(":") or "\n" in whitespace):
        return None
    labels = [int(match.group(1)) for match in matches]
    if labels[0] != 1 or sorted(labels) != list(range(1, len(labels) + 1)):
        return None
    options = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        value = content[match.end() : end].strip()
        if not value:
            return None
        options.append(AnswerOption(content[match.start() : match.end()], value))
    return OptionGroup(prefix, tuple(options))


def normalize_options(document: NormalizedDocument) -> None:
    """Attach presentation structure; retain original content and provenance."""
    for page in document.pages:
        follows_question = False
        for index, element in enumerate(page.elements):
            if element.kind != "text" or element.formula_placement is not None:
                follows_question = False
                continue
            group = recognize_options(element.content, follows_question)
            if group is not None:
                page.elements[index] = replace(element, option_group=group)
            follows_question = bool(_QUESTION.match(element.content)) and group is None
