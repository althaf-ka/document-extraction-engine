import logging
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from document_extractor.config import (
    DocumentEngineName,
    ExtractionConfig,
    InferenceBackend,
    Profile,
    WatermarkMode,
)
from document_extractor.exceptions import DocumentExtractorError
from document_extractor.pipeline import ExtractionPipeline
from document_extractor.progress import TerminalProgress

_DEFAULTS = ExtractionConfig()


def extract(
    input_path: Path,
    output: Annotated[Path, typer.Option()] = Path("output"),
    engine: Annotated[
        DocumentEngineName, typer.Option(envvar="DOCUMENT_ENGINE")
    ] = _DEFAULTS.engine,
    backend: Annotated[
        InferenceBackend, typer.Option(envvar="VLM_BACKEND")
    ] = _DEFAULTS.backend,
    server_url: Annotated[
        str | None, typer.Option(envvar="VLM_SERVER_URL")
    ] = _DEFAULTS.server_url,
    profile: Annotated[Profile, typer.Option()] = _DEFAULTS.profile,
    max_concurrency: Annotated[
        int, typer.Option(envvar="VLM_MAX_CONCURRENCY")
    ] = _DEFAULTS.max_concurrency,
    layout_threshold: Annotated[
        float, typer.Option(envvar="VLM_LAYOUT_THRESHOLD")
    ] = _DEFAULTS.layout_threshold,
    unclip_ratio: Annotated[float, typer.Option()] = _DEFAULTS.layout_unclip_ratio,
    debug: Annotated[
        bool, typer.Option(help="Save extraction diagnostics in debug/.")
    ] = _DEFAULTS.debug,
    question_layout: Annotated[
        bool,
        typer.Option(
            help="Automatically format clearly identified exam tables as questions. Disable with --no-question-layout."
        ),
    ] = _DEFAULTS.question_layout,
    watermark_mode: Annotated[
        WatermarkMode,
        typer.Option(
            envvar="WATERMARK_MODE",
            help="Repeated background text: off, report only, or conservative filter.",
        ),
    ] = _DEFAULTS.watermark_mode,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed application logs and upstream warnings.",
        ),
    ] = False,
) -> None:
    """Extract a PDF or image into Markdown, HTML tables, and PNG images."""
    try:
        config = ExtractionConfig(
            engine=engine,
            backend=backend,
            server_url=server_url,
            profile=profile,
            max_concurrency=max_concurrency,
            layout_threshold=layout_threshold,
            layout_unclip_ratio=unclip_ratio,
            debug=debug,
            question_layout=question_layout,
            watermark_mode=watermark_mode,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    try:
        with TerminalProgress(input_path, config, verbose=verbose) as progress:
            logging.getLogger(__name__).debug(
                "Paddle settings: threshold=%s, shape=%s, unclip=%s, concurrency=%s",
                config.layout_threshold,
                config.to_paddle_config().layout_shape_mode,
                config.layout_unclip_ratio,
                config.max_concurrency,
            )
            bundle = ExtractionPipeline(
                config=config, on_progress=progress.on_event
            ).run(input_path, output)
    except DocumentExtractorError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Output: {bundle.root_dir}")
    if config.watermark_mode == WatermarkMode.REPORT:
        typer.echo(
            "Watermark report only: no content was removed. Review watermarks.json; "
            "set WATERMARK_MODE=filter in .env to apply verified exclusions."
        )
    progress.print_total()


def main() -> None:
    # Read only the invocation directory; explicit environment and flags win.
    load_dotenv(Path.cwd() / ".env", override=False)
    typer.run(extract)
