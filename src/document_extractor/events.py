"""Factual extraction events, independent of terminal presentation."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    stage: str
    message: str
    current: int | None = None
    total: int | None = None
    completed: bool = False


ProgressCallback = Callable[[ProgressEvent], None]


def ignore_progress(event: ProgressEvent) -> None:
    """Default for callers that do not need progress reporting."""
