#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/agent-rl-train/bin/python}"

cd "$PROJECT_ROOT"
exec uv pip install --python "$PYTHON_BIN" -e ./verl --no-deps
