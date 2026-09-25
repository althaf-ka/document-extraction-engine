#!/bin/sh
set -eu
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    echo "MLX requires native Apple Silicon macOS." >&2
    exit 1
fi
cd "$(dirname "$0")/.."
exec uv run --extra mlx mlx_vlm.server --port 8111
