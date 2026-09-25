# Technical architecture and hosting summary

## Approach and alternatives

PaddleOCR-VL 1.6 handles layout detection, reading order, region recognition, and
page reconstruction. The Python application validates input, normalizes parser
results into ordered pages and elements, retries selected incomplete regions,
and exports Markdown, HTML tables, and PNG figures. The VLM runs in a separate
llama.cpp service using Paddle's official GGUF weights. Layout inference remains
in the Python process.

This approach supports mixed text, formulas, tables, and figures in one parser.
Docling and separate Paddle layout/formula models were evaluated on a two-page
maths fixture. The retained [experiment notes](benchmark-findings.md) describe
missing formulas and content in those alternatives. They are fixture-specific
observations, not a general ranking. Docling remains an optional adapter, with
explicit selection and no automatic fallback.

## Compute and memory

The workstation inspected for this submission exposes an Intel Core Ultra 5 225H,
14 logical CPUs, and approximately 15 GiB RAM through WSL2. This is the available
development environment, not a measured minimum requirement or evidence of the
hardware used for every historical benchmark. The documented deployment uses
CPU inference and requires no NVIDIA GPU.

Use Python 3.12 and Docker Compose on Linux x86-64. Model caches require persistent
disk space and an initial download. Peak application/server RAM and cache size
have not been recorded. Memory demand depends on page size, model context, and
concurrency, so capacity must be measured with representative documents.
Compose defaults to two VLM slots and a 4096-token context setting; copying
`.env.example` changes that context setting to 8192.

## Observed performance

The retained historical CPU experiment processed a two-page maths document in
341.558 seconds with Paddle tables and formulas enabled, or 170.779 seconds per
page. Initialization took another 12.951 seconds; export was excluded. The source
is the [benchmark record](benchmark-findings.md), whose raw artifacts were removed.

That experiment predates the current deployment and is not a PaddleOCR-VL 1.6
llama.cpp throughput claim. Current end-to-end latency and peak memory remain
unmeasured in the retained submission evidence. The CLI reports stage and total
elapsed time. A submission benchmark should record these for each sample,
together with page count, hardware, model/runtime versions, context, concurrency,
and whether caches were warm. Targeted retries add requests only for qualifying
regions; unit coverage does not establish their model accuracy or latency.

## Self-hosting and scaling

The Dockerfile installs locked Python dependencies. Compose runs the CLI container
and llama.cpp service, waits for VLM health, mounts input read-only, and persists
output and model caches. The server binds to host loopback port 8111; the app
uses the private Compose address `http://vlm:8080/v1`. Documents require no
external OCR or LLM API after setup.

For production, pin the container digest and model revision, provision model
caches, and measure resource limits. A job queue with bounded extraction workers
could share a VLM service; additional inference replicas could increase capacity.
These are deployment extensions, not implemented queueing or scaling features.
Keep request concurrency within server capacity and benchmark GPU serving before
claiming a throughput improvement.

## Contract and practical limits

The normalized document separates parser-specific data from export formatting.
Tables retain recognized HTML spans, and figures have relative Markdown links.
Existing output bundles are protected from replacement. Diagnostics preserve
primary and normalized results when requested.

Reading order, table fidelity, and figure detection depend on model predictions.
Generic image descriptions, excluded footnotes/asides, and incomplete artifact
filtering remain gaps against the requested output contract. Generated bundles
need visual review. The local `output/jee/` sample is Git-ignored, and outputs for
the full supplied sample set must accompany the submission.

Optional Docling setup is `uv sync --extra docling`, followed by
`uv run doc-extract input.pdf --engine docling` with Paddle backend settings
unset. Apple Silicon can use `sh scripts/start_mlx.sh` and
`uv run --extra mlx doc-extract input.pdf --backend mlx --server-url http://localhost:8111`.
