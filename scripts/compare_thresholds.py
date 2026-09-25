"""Run one controlled test2.pdf comparison with the current native CLI settings."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from validate_fixture import FIXTURE_SHA256, validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("pdfs/test2.pdf"))
    parser.add_argument(
        "--output", type=Path, required=True, help="A fresh comparison directory"
    )
    args = parser.parse_args()
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != FIXTURE_SHA256:
        parser.error("Input does not match the known test2.pdf fixture")
    if args.output.exists():
        parser.error("Choose a new output directory; comparisons are never overwritten")
    args.output.mkdir(parents=True)
    reports = {}
    for threshold in (0.5, 0.4):
        root = args.output / str(threshold)
        command = [
            sys.executable,
            "-c",
            "from document_extractor.cli import main; main()",
            str(args.input),
            "--engine",
            "paddle-vl",
            "--output",
            str(root),
            "--layout-threshold",
            str(threshold),
            "--debug",
        ]
        start = time.perf_counter()
        result = subprocess.run(command, check=False)
        elapsed = time.perf_counter() - start
        if result.returncode:
            raise SystemExit(result.returncode)
        markdown = root / args.input.stem / "document.md"
        reports[str(threshold)] = {
            "config": json.loads(
                (root / args.input.stem / "debug" / "elements.json").read_text()
            )["config"],
            "versions": {name: version(name) for name in ("paddleocr", "paddlex")},
            "elapsed_seconds": elapsed,
            **validate(markdown.read_text(encoding="utf-8")),
        }
    # Timings include startup and model loading. One run is not a throughput benchmark.
    payload = json.dumps(reports, indent=2, ensure_ascii=False) + "\n"
    temporary = args.output / "comparison.json.tmp"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(args.output / "comparison.json")
    print(payload)


if __name__ == "__main__":
    main()
