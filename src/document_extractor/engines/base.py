from pathlib import Path
from typing import Protocol

from document_extractor.models import NormalizedDocument


class DocumentEngine(Protocol):
    def extract(self, path: Path) -> NormalizedDocument: ...
