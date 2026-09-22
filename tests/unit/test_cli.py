from pathlib import Path

import pytest

from document_extractor.cli import extract


def test_extract_prints_configuration(capsys: pytest.CaptureFixture[str]) -> None:
    extract(Path("input.pdf"), Path("results"), "en")

    assert capsys.readouterr().out == (
        "Input: input.pdf\nOutput: results\nLanguage: en\n"
    )


def test_extract_uses_default_output(capsys: pytest.CaptureFixture[str]) -> None:
    extract(Path("input.pdf"))

    assert capsys.readouterr().out == (
        "Input: input.pdf\nOutput: output\nLanguage: en\n"
    )
