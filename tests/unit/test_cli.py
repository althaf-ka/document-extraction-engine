from pathlib import Path
from unittest.mock import patch

import typer
from typer.testing import CliRunner

from document_extractor.cli import extract
from document_extractor.config import InferenceBackend
from document_extractor.exporters.bundle import OutputBundle

app = typer.Typer()
app.command()(extract)
runner = CliRunner()


def test_extract_configuration_and_env(tmp_path: Path) -> None:
    bundle = OutputBundle.for_input(Path("paper.pdf"), tmp_path)
    with patch("document_extractor.cli.ExtractionPipeline") as pipeline:
        pipeline.return_value.run.return_value = bundle
        result = runner.invoke(
            app,
            [
                "paper.pdf",
                "--backend",
                "mlx",
                "--server-url",
                "http://localhost:8111",
                "--max-concurrency",
                "2",
                "--layout-threshold",
                "0.6",
            ],
            env={"VLM_BACKEND": "llama-cpp"},
        )
    assert result.exit_code == 0, result.output
    config = pipeline.call_args.kwargs["config"]
    assert config.backend == InferenceBackend.MLX
    assert config.max_concurrency == 2
    assert config.layout_threshold == 0.6
    assert str(bundle.root_dir) in result.output


def test_extract_uses_default_vlm_tuning(tmp_path: Path) -> None:
    bundle = OutputBundle.for_input(Path("paper.pdf"), tmp_path)
    with patch("document_extractor.cli.ExtractionPipeline") as pipeline:
        pipeline.return_value.run.return_value = bundle
        result = runner.invoke(app, ["paper.pdf"])

    assert result.exit_code == 0, result.output
    config = pipeline.call_args.kwargs["config"]
    assert config.max_concurrency == 1
    assert config.layout_threshold == 0.5


def test_extract_rejects_missing_server() -> None:
    result = runner.invoke(app, ["paper.pdf", "--backend", "mlx"])
    assert result.exit_code == 2
    assert "requires an HTTP(S)" in result.output


def test_extract_reports_invalid_input(tmp_path: Path) -> None:
    result = runner.invoke(app, [str(tmp_path / "missing.pdf")])
    assert result.exit_code == 1
    assert "does not exist" in result.output
    assert "Traceback" not in result.output


def test_help_without_loading_models() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--engine" in result.output
    assert "--verbose" in result.output


def test_verbose_is_presentation_only(tmp_path):
    bundle = OutputBundle.for_input(Path("paper.pdf"), tmp_path)
    with patch("document_extractor.cli.ExtractionPipeline") as pipeline:
        pipeline.return_value.run.return_value = bundle
        normal = runner.invoke(app, ["paper.pdf"])
        normal_config = pipeline.call_args.kwargs["config"]
        verbose = runner.invoke(app, ["paper.pdf", "--verbose"])
        assert pipeline.call_args.kwargs["config"] == normal_config
        assert callable(pipeline.call_args.kwargs["on_progress"])
    assert normal.exit_code == verbose.exit_code == 0
    assert "Paddle settings:" not in normal.output
    assert "Paddle settings:" in verbose.output
    assert "Total:" in normal.output


def test_main_loads_dotenv_with_environment_and_flag_precedence(tmp_path, monkeypatch):
    from document_extractor.cli import main

    monkeypatch.chdir(tmp_path)
    for key in ("DOCUMENT_ENGINE", "VLM_BACKEND", "VLM_SERVER_URL"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env").write_text(
        "DOCUMENT_ENGINE=paddle-vl\nVLM_BACKEND=llama-cpp\nVLM_SERVER_URL=http://localhost:8111/v1\n"
    )
    monkeypatch.setenv("VLM_BACKEND", "mlx")
    bundle = OutputBundle.for_input(Path("paper.pdf"), tmp_path)
    with (
        patch("document_extractor.cli.typer.run") as run,
        patch("document_extractor.cli.ExtractionPipeline") as pipeline,
    ):
        pipeline.return_value.run.return_value = bundle
        main()
        run.assert_called_once_with(extract)
        result = runner.invoke(app, ["paper.pdf"])
        assert result.exit_code == 0, result.output
        config = pipeline.call_args.kwargs["config"]
        assert config.backend == InferenceBackend.MLX
        assert config.server_url == "http://localhost:8111/v1"
        result = runner.invoke(app, ["paper.pdf", "--backend", "llama-cpp"])
        assert result.exit_code == 0, result.output
        assert pipeline.call_args.kwargs["config"].backend == InferenceBackend.LLAMA_CPP
