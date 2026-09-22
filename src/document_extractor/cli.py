from pathlib import Path
from typing import Annotated

import typer


def extract(
    input_path: Path,
    output: Annotated[
        Path | None,
        typer.Option(show_default="output"),
    ] = None,
    language: str = "en",
) -> None:
    """Extract structured content from a document."""

    output_path = output if output is not None else Path("output")

    typer.echo(f"Input: {input_path}")
    typer.echo(f"Output: {output_path}")
    typer.echo(f"Language: {language}")


def main() -> None:
    typer.run(extract)
