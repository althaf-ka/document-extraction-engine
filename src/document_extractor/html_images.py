"""Rewrite image tags without reserializing surrounding table markup."""

from collections.abc import Callable
from html import escape
from html.parser import HTMLParser


def rewrite_image_sources(html: str, resolve: Callable[[str], str]) -> str:
    class Images(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=False)
            self.edits: list[tuple[int, int, str]] = []

        def handle_starttag(self, tag, attrs):
            if tag != "img":
                return
            attributes = []
            for key, value in attrs:
                # Paddle can emit alt="Image"". HTMLParser exposes the stray
                # quote as an attribute; discard it when rebuilding this tag.
                if key in {'"', "'"}:
                    continue
                if key == "src" and value is not None:
                    value = resolve(value)
                attributes.append(
                    key if value is None else f'{key}="{escape(value, quote=True)}"'
                )
            line, column = self.getpos()
            start = (
                sum(len(part) for part in html.splitlines(keepends=True)[: line - 1])
                + column
            )
            original = self.get_starttag_text()
            assert original is not None
            self.edits.append(
                (start, start + len(original), "<img " + " ".join(attributes) + ">")
            )

        def handle_startendtag(self, tag, attrs):
            self.handle_starttag(tag, attrs)

    parser = Images()
    parser.feed(html)
    parser.close()
    for start, end, replacement in reversed(parser.edits):
        html = html[:start] + replacement + html[end:]
    return html
