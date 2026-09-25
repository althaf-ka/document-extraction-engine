from pathlib import Path

from document_extractor.exceptions import (
    InputFileNotFoundError,
    UnsupportedFileError,
)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpeg",
        ".jpg",
        ".pdf",
        ".png",
    }
)


def validate_input(path: Path) -> Path:
    """Validate an input document and return its path."""
    if not path.exists():
        raise InputFileNotFoundError(f"Input file does not exist: {path}")

    if not path.is_file():
        raise InputFileNotFoundError(f"Input path is not a file: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        displayed_extension = extension or "<none>"
        raise UnsupportedFileError(f"Unsupported file format: {displayed_extension}")

    return path
