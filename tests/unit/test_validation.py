from pathlib import Path

import pytest

from document_extractor.exceptions import (
    InputFileNotFoundError,
    UnsupportedFileError,
)
from document_extractor.validation import validate_input


def test_validate_input_rejects_missing_file(tmp_path: Path) -> None:
    input_path = tmp_path / "missing-file.pdf"

    with pytest.raises(
        InputFileNotFoundError,
        match=f"Input file does not exist: {input_path}",
    ):
        _ = validate_input(input_path)


def test_validate_input_rejects_unsupported_extension(tmp_path: Path) -> None:
    input_path = tmp_path / "virus.exe"
    input_path.touch()

    with pytest.raises(UnsupportedFileError, match="Unsupported file format: .exe"):
        _ = validate_input(input_path)


def test_validate_input_accepts_uppercase_extension(tmp_path: Path) -> None:
    input_path = tmp_path / "photo.JPG"
    input_path.touch()

    assert validate_input(input_path) == input_path


def test_validate_input_rejects_directory(tmp_path: Path) -> None:
    input_path = tmp_path / "documents.pdf"
    input_path.mkdir()

    with pytest.raises(
        InputFileNotFoundError,
        match=f"Input path is not a file: {input_path}",
    ):
        _ = validate_input(input_path)
