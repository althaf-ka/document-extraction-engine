from pathlib import Path

from document_extractor.exporters.bundle import OutputBundle


def test_for_input_builds_expected_paths(tmp_path: Path) -> None:
    output_root = tmp_path / "output"

    bundle = OutputBundle.for_input(Path("JEE Main 2025.pdf"), output_root)

    assert bundle.root_dir == output_root / "JEE Main 2025"
    assert bundle.document_path == output_root / "JEE Main 2025" / "document.md"
    assert bundle.tables_dir == output_root / "JEE Main 2025" / "tables"
    assert bundle.images_dir == output_root / "JEE Main 2025" / "images"


def test_create_builds_bundle_contract(tmp_path: Path) -> None:
    bundle = OutputBundle.for_input(
        Path("JEE Main 2025.pdf"),
        tmp_path / "output",
    )

    bundle.create()

    assert bundle.document_path.is_file()
    assert bundle.document_path.read_text() == ""
    assert bundle.tables_dir.is_dir()
    assert bundle.images_dir.is_dir()


def test_create_preserves_existing_document(tmp_path: Path) -> None:
    bundle = OutputBundle.for_input(Path("input.pdf"), tmp_path / "output")
    bundle.create()
    bundle.document_path.write_text("# Existing extraction\n")

    bundle.create()

    assert bundle.document_path.read_text() == "# Existing extraction\n"
