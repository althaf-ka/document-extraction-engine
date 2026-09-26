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

### Automatic question paper layout

Exam tables are detected automatically during export. No extra flag is needed:

```sh
uv run doc-extract pdf/maths.pdf --output output/questions
```

This export step uses the existing extraction. It recognizes question
tables from question/marks headers or section instructions, followed by multiple
numbered prose rows. It renders those rows as Markdown questions, preserves marks
and images, and puts complete A–D answer groups in separate paragraphs. It also
removes whitespace immediately inside inline math delimiters so math-enabled
Markdown viewers can render the extracted LaTeX outside HTML table cells.
Units following a formula are separated by a space, for example
`$\frac{5}{6}$ cm`, so VS Code recognizes the closing math delimiter.

Original table HTML remains in `tables/`. Data tables, unsupported markup,
nested tables, and ambiguous structures keep the existing table export. Detection
is conservative and may leave some exam tables unchanged. It does not alter
model settings or the normalized OCR content. Use `--no-question-layout` to
disable automatic question formatting for a particular extraction; the existing
`--question-layout` flag remains supported.

With `--debug`, `debug/question-layout.json` records converted tables, warnings,
and structural repairs. Normal extraction does not write this diagnostic file.
A narrowly constrained repair moves a trailing answer label out of an equation
when it completes an otherwise unambiguous A–D group and both adjacent formulas
contain equations.
The report identifies each such repair for source review. Extracted option order
is preserved, including A, C, B, D when that is the recognized order.

This mode fixes presentation and that specific delimiter error; it does not
validate mathematical correctness, recover missing symbols, or reconstruct data
tables already flattened by recognition. Unclosed delimiters are reported and
left unchanged. Review equations against the PDF when recognition is uncertain.

### Repeated watermark text

Configure watermark handling in `.env` in the directory where you run
`doc-extract`. No watermark flag is needed on each command:

```dotenv
WATERMARK_MODE=filter
```

| Value | Behavior |
| --- | --- |
| `off` | Disable watermark checks. Default when unset. |
| `report` | Keep all content and record candidate decisions in `watermarks.json`. |
| `filter` | Exclude verified background-only text/tables. Add `--debug` to retain their diagnostic record. |

Change this one setting before your next extraction. For example:

```sh
uv run doc-extract pdf/jee.pdf --output output/watermark-filtered
```

Existing output is not updated automatically; choose a fresh output directory
when rerunning. For an unfamiliar document, start with `WATERMARK_MODE=report`
and inspect its decisions before enabling `filter`.

The setting also passes into the Docker Compose app. Recreate the app container
after changing it. An explicit `--watermark-mode` flag overrides the environment;
existing process environment variables override `.env`. `.env.example` uses
`off` so new installations keep the original behavior. This option applies to
PaddleOCR-VL; leave it `off` for Docling. Python callers can set
`ExtractionConfig(watermark_mode=WatermarkMode.FILTER)` directly; `.env` loading
belongs to the CLI entry point.

These modes use your existing backend configuration and require no additional
OCR or AI calls. The existing PDFium dependency inspects and renders candidate
pages; no watermark model or new dependency is downloaded.

The filter discovers repeated phrases without a publisher-name list. It only
considers plain text consisting entirely of at least 12 repetitions of a short
alphabetic phrase, or simple tables with at least 12 identical nonempty cells
across three or more rows and columns. Mixed text, formulas, numeric cells,
headers, rich tables, and blocks carrying images are preserved.

Repetition alone never excludes content. A candidate must also occupy a pale
region backed by a single PDF graphics group whose vector outlines repeat in a
regular grid. The region must become effectively blank when that group is hidden
in a disposable in-memory rendering. Original PDFs and exported images are not
modified. Each excluded block's complete text or HTML, original page, and bounding
box remain in `debug/watermarks.json` when `--debug` is enabled. Report mode writes
`watermarks.json` at the bundle root because producing that report is its purpose.

This is a conservative heuristic, not a guarantee against false positives. A real
document containing pale, tiled vector data can resemble a watermark. Review
report mode on representative inputs before enabling filtering. Uncertain cases,
scans with baked-in watermarks, image inputs, rotated or oversized pages, missing
geometry, nested or multiple graphics groups, and robust-profile preprocessing
retain their content. Image watermarks and watermark words mixed into actual
questions are intentionally preserved.

PDF inspection happens only on pages containing candidates. Verification renders
each supported candidate page twice, at 1.5 pixels per PDF point, with a four-million
pixel limit per rendering. This adds CPU work; no speedup or measured overhead is
claimed. Filtering runs before recovery and cross-page table reconstruction.

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
Inline math formatting applies to ordinary text, answer groups, and simple table
cells. The exporter separates adjacent text from dollar delimiters, such as
`$27^{\circ}$ C`, and preserves LaTeX commands and subscripts inside table formulas
while escaping ordinary cell text. Formulas containing pipe characters retain
the original HTML table to avoid changing their meaning during Markdown export.
These are presentation changes; missing symbols or list labels in the recognized
content still require source review or recognition recovery.

`--debug` adds `debug/elements.json` with normalized elements and
`debug/paddle-result.json` with primary Paddle results.
With `--debug`, automatic question formatting writes
`debug/question-layout.json`, recording table conversion decisions and any
repairs or warnings. Normal bundles contain no question-layout diagnostic JSON.

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
- Optional watermark filtering covers only verified repetitive background blocks.
  There is no universal watermark, border, or separator filter.
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
