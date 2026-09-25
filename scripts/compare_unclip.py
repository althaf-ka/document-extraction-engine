"""Compare unchanged crops with vertical expansion on the known math fixture."""

import argparse
import hashlib
import os
import time
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv
from validate_fixture import FIXTURE_SHA256, validate

from document_extractor.config import (
    ExtractionConfig,
    InferenceBackend,
)
from document_extractor.engines.paddle_vl import PaddleVLEngine
from document_extractor.exporters.debug import export_debug, write_json
from document_extractor.exporters.document import DocumentExporter
from document_extractor.option_normalization import normalize_options


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("pdfs/test2.pdf"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != FIXTURE_SHA256:
        parser.error("Input must match the known test2.pdf fixture")
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    load_dotenv(Path.cwd() / ".env", override=False)
    config = ExtractionConfig(
        backend=InferenceBackend.LLAMA_CPP,
        server_url=os.environ.get("VLM_SERVER_URL", "http://localhost:8111/v1"),
        layout_threshold=0.5,
        max_concurrency=2,
        debug=True,
    )
    baseline = config.to_paddle_config()
    reports = {}
    for name, ratio in (("baseline", (1.0, 1.0)), ("vertical-1.2", (1.0, 1.2))):
        print(f"Running {name}: {ratio}", flush=True)
        run_config = replace(baseline, layout_unclip_ratio=ratio)
        started = time.perf_counter()
        document = PaddleVLEngine(run_config, debug=True).extract(args.input)
        normalize_options(document)
        bundle = DocumentExporter().export(document, args.input, args.output / name)
        export_debug(
            document, bundle.root_dir / "debug", config, paddle_config=run_config
        )
        reports[name] = {
            "config": asdict(run_config),
            "versions": {name: version(name) for name in ("paddleocr", "paddlex")},
            "pages": len(document.pages),
            "elapsed_seconds_including_model_load": time.perf_counter() - started,
            **validate(bundle.document_path.read_text(encoding="utf-8")),
        }
        reports[name]["config"]["vlm"].pop("server_url")
        write_json(args.output / "comparison.json", reports)
        print(reports[name], flush=True)


if __name__ == "__main__":
    main()
