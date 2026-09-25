from document_extractor.config import DocumentEngineName, ExtractionConfig
from document_extractor.engines.base import DocumentEngine
from document_extractor.events import ProgressCallback, ignore_progress


def create_engine(
    config: ExtractionConfig, on_progress: ProgressCallback = ignore_progress
) -> DocumentEngine:
    if config.engine == DocumentEngineName.DOCLING:
        from document_extractor.engines.docling import DoclingEngine

        return DoclingEngine(on_progress=on_progress)
    from document_extractor.engines.paddle_vl import PaddleVLEngine

    return PaddleVLEngine(config, on_progress=on_progress)
