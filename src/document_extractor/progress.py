"""Rich terminal presentation for extraction events; no model knowledge here."""

import logging
import warnings
from contextlib import ExitStack
from pathlib import Path
from time import monotonic

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.progress import (
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
)
from rich.table import Table
from rich.text import Text

from document_extractor.config import (
    DocumentEngineName,
    ExtractionConfig,
    InferenceBackend,
)
from document_extractor.events import ProgressEvent


def duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}m {seconds:02d}s"


class _StatusColumn(SpinnerColumn):
    def render(self, task: Task) -> RenderableType:
        if task.fields.get("error"):
            return Text("×", style="red")
        return super().render(task)


class _ElapsedColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        return Text(duration(elapsed or 0), style="dim", justify="right")


class TerminalProgress(logging.Handler):
    def __init__(
        self,
        input_path: Path | None = None,
        config: ExtractionConfig | None = None,
        *,
        verbose: bool = False,
        console: Console | None = None,
    ) -> None:
        super().__init__(logging.DEBUG if verbose else logging.WARNING)
        self.input_path = input_path
        self.config = config or ExtractionConfig()
        self.verbose = verbose
        self.console = console or Console(stderr=True)
        self.started = monotonic()
        self.elapsed = 0.0
        self.total_pages: int | None = None
        self.tasks: dict[str, TaskID] = {}
        self.active_stage: str | None = None
        self.progress = Progress(
            _StatusColumn(style="cyan", finished_text="[green]✓[/green]"),
            TextColumn("{task.description}", markup=False),
            _ElapsedColumn(),
            console=self.console,
            auto_refresh=False,
            expand=False,
        )
        self.live: Live | None = None
        self.logger = logging.getLogger("document_extractor")
        self.stack = ExitStack()

    def _header(self) -> Group:
        metadata = Table.grid(padding=(0, 2))
        metadata.add_column(style="dim")
        metadata.add_column(overflow="fold")
        if self.input_path is not None:
            metadata.add_row("Input", Text(str(self.input_path)))
        metadata.add_row(
            "Pages", str(self.total_pages) if self.total_pages else "unknown"
        )
        paddle = self.config.engine == DocumentEngineName.PADDLE_VL
        metadata.add_row(
            "Engine",
            "PaddleOCR-VL " + self.config.to_paddle_config().pipeline_version
            if paddle
            else "Docling",
        )
        if paddle:
            metadata.add_row(
                "Backend",
                {
                    InferenceBackend.LOCAL: "local Paddle",
                    InferenceBackend.LLAMA_CPP: "llama.cpp",
                    InferenceBackend.MLX: "MLX",
                }[self.config.backend],
            )
            metadata.add_row("Concurrency", str(self.config.max_concurrency))
        return Group(Text("Document Extractor", style="bold"), Text(), metadata, Text())

    def _display(self) -> Group:
        return Group(self._header(), self.progress)

    def on_event(self, event: ProgressEvent) -> None:
        old_total = self.total_pages
        if event.total is not None:
            self.total_pages = event.total
        description = event.message
        if event.current is not None:
            count = str(event.current)
            if self.total_pages is not None:
                count += f"/{self.total_pages}"
            description += f" · {count} pages received"
        task_id = self.tasks.get(event.stage)
        new_stage = task_id is None
        if task_id is None:
            task_id = self.progress.add_task(description, total=None)
            self.tasks[event.stage] = task_id
        else:
            self.progress.update(task_id, description=description)
        self.active_stage = event.stage
        if event.completed:
            self.progress.update(task_id, total=1, completed=1)
            self.progress.stop_task(task_id)
        if self.live is not None:
            self.live.update(self._display(), refresh=True)
        else:
            if old_total != self.total_pages:
                self.console.print(Text(f"Pages: {self.total_pages}"))
            if event.completed:
                task = self.progress.tasks[task_id]
                self.console.print(
                    Text(f"✓ {description}  {duration(task.finished_time or 0)}")
                )
            elif new_stage or event.current is not None:
                self.console.print(Text(description))

    def emit(self, record: logging.LogRecord) -> None:
        self.console.print(Text(f"{record.levelname}: {self.format(record)}"))

    def __enter__(self):
        self.started = monotonic()
        self.previous_level = self.logger.level
        self.previous_propagate = self.logger.propagate
        self.logger.setLevel(logging.DEBUG if self.verbose else logging.WARNING)
        self.logger.propagate = False
        self.logger.addHandler(self)
        self.stack.enter_context(warnings.catch_warnings())
        if not self.verbose and self.config.backend == InferenceBackend.LLAMA_CPP:
            # PaddleX itself injects these values. Its remote predictor already
            # omits them from llama.cpp requests. Keep every other warning.
            warnings.filterwarnings(
                "ignore",
                message=r"^'llama-cpp-server' does not support `(min|max)_pixels`\.$",
                category=UserWarning,
                module=r"^paddlex\.inference\.models\.doc_vlm\.predictor$",
            )
        if self.console.is_interactive:
            self.live = Live(
                self._display(), console=self.console, refresh_per_second=4
            )
            self.live.start()
        else:
            self.console.print(self._header())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = monotonic() - self.started
        try:
            if exc_type:
                status = (
                    "Cancelled" if issubclass(exc_type, KeyboardInterrupt) else "Failed"
                )
                if self.active_stage is not None:
                    task_id = self.tasks[self.active_stage]
                    task = self.progress.tasks[task_id]
                    if not task.finished:
                        self.progress.update(
                            task_id,
                            description=f"{status}: {task.description}",
                            error=True,
                            total=1,
                            completed=1,
                        )
                        self.progress.stop_task(task_id)
                if self.live is None:
                    self.console.print(Text(status, style="red"))
            if self.live is not None:
                self.live.update(self._display(), refresh=True)
                self.live.stop()
        finally:
            self.stack.close()
            self.logger.removeHandler(self)
            self.logger.setLevel(self.previous_level)
            self.logger.propagate = self.previous_propagate
            self.close()

    def print_total(self) -> None:
        self.console.print(Text(f"Total: {duration(self.elapsed)}"))
