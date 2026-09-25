import logging
import warnings
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from document_extractor.config import ExtractionConfig, InferenceBackend
from document_extractor.events import ProgressEvent
from document_extractor.progress import TerminalProgress


def reporter(*, interactive=False, verbose=False, config=None):
    stream = StringIO()
    console = Console(
        file=stream,
        force_terminal=interactive,
        force_interactive=interactive,
        color_system=None,
        width=90,
    )
    return TerminalProgress(
        Path("[bold]paper.pdf"), config, console=console, verbose=verbose
    ), stream


def test_plain_output_reports_only_events_and_preserves_literal_paths():
    progress, stream = reporter()
    with progress:
        progress.on_event(ProgressEvent("validation", "Validating input"))
        progress.on_event(
            ProgressEvent("validation", "Input validated", total=2, completed=True)
        )
        progress.on_event(ProgressEvent("inference", "Extracting", current=0))
        progress.on_event(ProgressEvent("inference", "Extracting", current=1))
        progress.on_event(
            ProgressEvent("inference", "Extracted", current=2, completed=True)
        )
    progress.print_total()
    output = stream.getvalue()
    assert "[bold]paper.pdf" in output
    assert "0/2 pages received" in output
    assert "1/2 pages received" in output
    assert "✓ Extracted · 2/2 pages received" in output
    assert "Total:" in output
    assert "Waiting" not in output
    assert "\x1b" not in output
    assert "%" not in output
    assert progress.live is None


def test_unknown_page_count_and_interactive_indeterminate_task():
    progress, stream = reporter(interactive=True)
    with progress:
        progress.on_event(ProgressEvent("inference", "Extracting", current=0))
        task = progress.progress.tasks[progress.tasks["inference"]]
        assert task.total is None
        assert task.description == "Extracting · 0 pages received"
        progress.on_event(ProgressEvent("inference", "Extracting", current=1, total=2))
        assert task.total is None  # Received pages do not imply work percentage.
        progress.on_event(
            ProgressEvent("inference", "Extracted", current=2, total=2, completed=True)
        )
    assert progress.live is not None and not progress.live.is_started
    assert "2/2 pages received" in stream.getvalue()
    assert "%" not in stream.getvalue()


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
@pytest.mark.parametrize("interactive", [False, True])
def test_failure_and_cancellation_stop_live_and_restore_logger(error, interactive):
    logger = logging.getLogger("document_extractor")
    original = (logger.level, logger.propagate, list(logger.handlers))
    original_filters = list(warnings.filters)
    progress, stream = reporter(interactive=interactive)
    with pytest.raises(error), progress:
        progress.on_event(ProgressEvent("inference", "Extracting", current=0))
        raise error("stopped")
    assert (logger.level, logger.propagate, logger.handlers) == original
    assert warnings.filters == original_filters
    assert progress.live is None or not progress.live.is_started
    assert (
        "Cancelled" if error is KeyboardInterrupt else "Failed"
    ) in stream.getvalue()
    assert "✓ Extracting" not in stream.getvalue()


@pytest.mark.parametrize("verbose", [False, True])
def test_verbose_controls_details_without_hiding_application_warnings(verbose):
    progress, stream = reporter(verbose=verbose)
    with progress:
        logger = logging.getLogger("document_extractor.engines")
        logger.info("Loading detailed model settings")
        logger.warning("Recovery could not complete")
    assert ("Loading detailed model settings" in stream.getvalue()) is verbose
    assert "Recovery could not complete" in stream.getvalue()


@pytest.mark.parametrize("verbose", [False, True])
def test_only_known_upstream_llama_pixel_warnings_are_filtered(verbose):
    config = ExtractionConfig(
        backend=InferenceBackend.LLAMA_CPP, server_url="http://localhost:8111/v1"
    )
    progress, _ = reporter(config=config, verbose=verbose)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with progress:
            for message, module in [
                (
                    "'llama-cpp-server' does not support `min_pixels`.",
                    "paddlex.inference.models.doc_vlm.predictor",
                ),
                (
                    "'llama-cpp-server' does not support `max_pixels`.",
                    "paddlex.inference.models.doc_vlm.predictor",
                ),
                ("'llama-cpp-server' does not support `min_pixels`.", "another.module"),
                (
                    "Unexpected model warning",
                    "paddlex.inference.models.doc_vlm.predictor",
                ),
                ("No ccache found", "paddle.utils.cpp_extension.extension_utils"),
            ]:
                warnings.warn_explicit(
                    message, UserWarning, "predictor.py", 1, module=module
                )
    assert len(caught) == (5 if verbose else 3)
    assert any("Unexpected model warning" in str(item.message) for item in caught)
    assert any("No ccache found" in str(item.message) for item in caught)
