from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OutputBundle:
    """Paths that make up one document's output bundle."""

    root_dir: Path
    document_path: Path
    tables_dir: Path
    images_dir: Path

    @classmethod
    def for_input(cls, input_path: Path, output_root: Path) -> "OutputBundle":
        """Build the output paths for an input document."""
        root_dir = output_root / input_path.stem
        return cls(
            root_dir=root_dir,
            document_path=root_dir / "document.md",
            tables_dir=root_dir / "tables",
            images_dir=root_dir / "images",
        )

    def create(self) -> None:
        """Create the output bundle without overwriting existing content."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.document_path.touch(exist_ok=True)
