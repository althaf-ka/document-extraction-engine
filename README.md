# Multi-asset document & image extraction engine

A local, self-hosted pipeline that converts PDFs and images into structured
Markdown, separate HTML tables, and cropped visual assets. It uses PaddleOCR-VL
1.6 for document understanding and llama.cpp for CPU VLM inference.

## Features

- PDF, PNG, JPG, and JPEG input, including scanned and born-digital documents.
- Layout-aware parsing and reading-order reconstruction for mixed-content pages.
- Markdown text and headings, with LaTeX for recognized formulas.
- Separate HTML tables and PNG figures, embedded or referenced in document order.
- Local inference with no metered OCR or LLM API.
- Configurable CLI, diagnostics, and targeted recovery of incomplete regions.

## Architecture

```text
PDF / PNG / JPG
       |
Input validation
       |
PaddleOCR-VL pipeline
  Page rendering / preprocessing
       |
  Layout analysis + reading order
       |
  Detected region crops
       |
  VLM recognition ------------> llama.cpp server
       |                        PaddleOCR-VL 1.6 GGUF
  Page reconstruction <-------- recognized content
       |
Targeted recovery + normalization
       |
NormalizedDocument
       |
Document exporter
  +----+----------+
  |    |          |
Markdown HTML tables PNG images
```

Paddle detects regions and reading order, recognizes the crops, and reconstructs
the document. llama.cpp serves the VLM stage; Python still runs the Paddle layout
pipeline. This follows [Paddle's documented architecture](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html).

### Why llama.cpp?

CPU execution is the reference deployment. Paddle provides
[official GGUF weights](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF)
for llama.cpp, allowing local VLM inference without an NVIDIA GPU.
The inference service runs separately from the extraction application.

## Quick start

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and Docker with Compose.
The Docker setup targets Linux x86-64, including Windows WSL2.
Initial setup needs internet access to download dependencies and models.

Run from the repository root:

```sh
uv sync --locked --python 3.12
docker compose up -d --wait --wait-timeout 1500 vlm
uv run doc-extract pdfs/math.pdf --output output \
  --backend llama-cpp --server-url http://localhost:8111/v1
```

Results go to `output/math/`. Existing nonempty bundles are never overwritten;
choose a fresh output root for repeat runs. First startup downloads model weights
and can take several minutes.

To save backend settings, copy `.env.example` to `.env`. Subsequent commands
can omit the backend and server flags. Without configuration, the CLI defaults
to local Paddle. Flags override environment variables, then `.env`, then defaults.
The example uses port 8111; keep the URL in sync if changing `VLM_PORT`.

For an entirely containerized run:

```sh
mkdir -p input
cp pdfs/math.pdf input/document.pdf
docker compose up --build --abort-on-container-exit --exit-code-from app
```

This writes `output/document/`. Model caches persist in Docker volumes.
Use `docker compose logs -f vlm` to inspect startup and
`docker compose down` to stop the services.

## Usage

```sh
uv run doc-extract --help

# With backend settings saved in .env
uv run doc-extract scan.jpg --output output/scans --profile robust
uv run doc-extract pdfs/jee.pdf --output output/review --debug
```

The balanced profile keeps orientation correction and unwarping off.
The robust profile enables both. Both retain layout, table, and formula parsing.
`--max-concurrency` bounds concurrent VLM requests from 1 to 4. The native CLI
defaults to 1; Compose configures its app and server for 2.
`--verbose` enables detailed logs.

## Output structure

Each input produces a directory named after its file stem:

```text
output/<input_file_name>/
├── document.md
├── tables/
│   ├── tbl-0.html
│   └── tbl-1.html
└── images/
    ├── img-0.png
    └── img-1.png
```

Markdown follows the parser's reconstructed reading order. Simple tables appear
inline as Markdown; tables with merged cells or rich content remain HTML.
Every table also has an HTML file preserving the recognized structure, including
any row and column spans.

Figures use relative references such as `![Image 1](images/img-0.png)` in the
extracted sequence. Available recognized content follows the image.
Images inside tables are also saved in `images/`. Their HTML references use
`images/` in Markdown and `../images/` in the separate table files.
Formula rendering requires a Markdown viewer with LaTeX support.

`--debug` adds `debug/elements.json` with normalized elements and
`debug/paddle-result.json` with primary Paddle results.
Normal bundles do not include metadata JSON.

## Design decisions

The engine adapter converts Paddle results into a vendor-independent
`NormalizedDocument`. The exporter writes Markdown, HTML, and PNG files without
importing the OCR engine. An optional Docling adapter uses the same contract.

Targeted retries handle certain incomplete multiple-choice blocks and matching
tables using rectangular crops with layout detection disabled. They avoid
rerunning the whole page. Missing answers are only added when recovered by
recognition; failed retries retain the primary result.

Processing runs locally after model downloads. The dependency lockfile supports
repeatable setup, but OCR output can vary across hardware and runtime versions.

## Performance notes

CPU recognition is expensive, especially for pages with many formulas, tables,
or detected regions. Bounded concurrency limits competing requests; increasing
it does not guarantee higher throughput.

The [hosting summary](docs/architecture.md) records the workstation profile
and available timing evidence. Historical [benchmark notes](docs/benchmark-findings.md)
are not measurements of the current llama.cpp pipeline.

## Known limitations

- Layout errors can exclude content before recognition. A reported question crop
  omitted two answer choices; a rectangular crop recovered them. Recovery covers
  specific suspicious patterns with safe crop boundaries.
- Small text, degraded scans, formulas, and complex tables can be misrecognized.
  List and callout formatting depends on the parser.
- Image placement follows inferred reading order, not exact page coordinates.
  Alt text is generic; contextual figure descriptions are not generated.
- Headers, footers, page numbers, footnotes, and asides are omitted from Markdown.
  Misclassified content can be lost. Debug output retains these elements.
- Artifact removal relies on detected labels. There is no general watermark,
  border, or separator filter.
- The llama.cpp container uses a rolling image tag. Pin a tested image digest
  and model revision for a reproducible deployment.

## Testing

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

Unit tests use fake inference results to cover validation, adapters, retries, and
export contracts. They do not establish OCR accuracy. Review generated documents
against the supplied inputs before submission.

## Project structure

```text
src/document_extractor/
├── cli.py                 # CLI and environment settings
├── config.py              # Extraction and backend configuration
├── pipeline.py            # Validation, extraction, and export
├── models.py              # Normalized document contract
├── engines/               # PaddleOCR-VL and optional Docling adapters
├── processing/            # Targeted question and table recovery
├── exporters/             # Markdown, HTML, PNG, and diagnostics
├── option_normalization.py
├── validation.py
├── events.py
└── progress.py
```

## Submission deliverables

The source and setup instructions are in this repository. The brief
[technical architecture and hosting summary](docs/architecture.md) covers the
approach, alternatives, compute profile, performance evidence, and deployment.

At this documentation review, the local generated sample is `output/jee/`,
with Markdown, 4 HTML tables, and 31 PNG images. Git ignores this directory.
Include reviewed `output/<input_file_name>/` bundles for every supplied sample
in the submission archive, or explicitly add the selected files to the repository.
A repository checkout alone currently omits generated outputs.

## Future improvements

Validate every supplied sample, record current end-to-end timings and peak memory,
improve contextual image descriptions and artifact filtering, and evaluate
GPU-backed VLM serving for higher throughput.
