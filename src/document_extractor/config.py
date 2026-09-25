from dataclasses import dataclass, replace
from enum import StrEnum
from urllib.parse import urlsplit

MARKDOWN_IGNORE_LABELS = (
    "number",
    "footnote",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "aside_text",
)


class DocumentEngineName(StrEnum):
    PADDLE_VL = "paddle-vl"
    DOCLING = "docling"


class InferenceBackend(StrEnum):
    LOCAL = "local"
    LLAMA_CPP = "llama-cpp"
    MLX = "mlx"


class Profile(StrEnum):
    BALANCED = "balanced"
    ROBUST = "robust"


@dataclass(frozen=True, slots=True)
class VLMConfig:
    backend: InferenceBackend = InferenceBackend.LOCAL
    server_url: str | None = None
    max_concurrency: int = 1
    mlx_model_name: str = "PaddlePaddle/PaddleOCR-VL-1.6"

    def __post_init__(self) -> None:
        if not 1 <= self.max_concurrency <= 4:
            raise ValueError("--max-concurrency must be between 1 and 4")
        if self.backend == InferenceBackend.LOCAL:
            if self.server_url:
                raise ValueError("--server-url requires --backend llama-cpp or mlx")
        else:
            url = urlsplit(self.server_url or "")
            if url.scheme not in {"http", "https"} or not url.hostname:
                raise ValueError("This backend requires an HTTP(S) --server-url")


@dataclass(frozen=True, slots=True)
class PaddleVLConfig:
    pipeline_version: str = "v1.6"
    layout_threshold: float = 0.5
    layout_shape_mode: str = "auto"
    layout_unclip_ratio: tuple[float, float] = (1.0, 1.0)
    layout_merge_bboxes_mode: str | None = None
    merge_layout_blocks: bool = True
    use_layout_detection: bool = True
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_chart_recognition: bool = False
    format_block_content: bool = True
    use_queues: bool = True
    markdown_ignore_labels: tuple[str, ...] = MARKDOWN_IGNORE_LABELS
    mcq_retry: bool = True
    mcq_crop_margin: int = 8
    table_retry: bool = True
    table_crop_margin: int = 8
    vlm: VLMConfig = VLMConfig()

    def __post_init__(self) -> None:
        if not 0.2 <= self.layout_threshold <= 0.8:
            raise ValueError("--layout-threshold must be between 0.2 and 0.8")
        if len(self.layout_unclip_ratio) != 2 or not all(
            1.0 <= value <= 2.0 for value in self.layout_unclip_ratio
        ):
            raise ValueError("--unclip-ratio must be between 1.0 and 2.0")
        if self.layout_shape_mode not in {"auto", "rect", "quad", "poly"}:
            raise ValueError("layout_shape_mode must be auto, rect, quad, or poly")
        if self.mcq_crop_margin < 0:
            raise ValueError("mcq_crop_margin must be nonnegative")
        if self.table_crop_margin < 0:
            raise ValueError("table_crop_margin must be nonnegative")


DEFAULT_PADDLE_CONFIG = PaddleVLConfig()


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """Application/CLI settings, resolved to Paddle settings at the adapter boundary."""

    engine: DocumentEngineName = DocumentEngineName.PADDLE_VL
    backend: InferenceBackend = DEFAULT_PADDLE_CONFIG.vlm.backend
    server_url: str | None = DEFAULT_PADDLE_CONFIG.vlm.server_url
    profile: Profile = Profile.BALANCED
    max_concurrency: int = DEFAULT_PADDLE_CONFIG.vlm.max_concurrency
    layout_threshold: float = DEFAULT_PADDLE_CONFIG.layout_threshold
    debug: bool = False
    layout_unclip_ratio: float = DEFAULT_PADDLE_CONFIG.layout_unclip_ratio[0]

    def to_paddle_config(self) -> PaddleVLConfig:
        robust = self.profile == Profile.ROBUST
        return replace(
            DEFAULT_PADDLE_CONFIG,
            layout_threshold=self.layout_threshold,
            layout_unclip_ratio=(self.layout_unclip_ratio, self.layout_unclip_ratio),
            use_doc_orientation_classify=robust,
            use_doc_unwarping=robust,
            vlm=VLMConfig(
                backend=self.backend,
                server_url=self.server_url,
                max_concurrency=self.max_concurrency,
            ),
        )

    def __post_init__(self) -> None:
        self.to_paddle_config()
        if self.engine == DocumentEngineName.DOCLING:
            if self.backend != InferenceBackend.LOCAL or self.server_url:
                raise ValueError("--backend and --server-url apply only to paddle-vl")
            if self.profile != Profile.BALANCED:
                raise ValueError("--profile robust applies only to paddle-vl")
