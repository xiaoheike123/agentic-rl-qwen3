#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/vllm/bin/python}"
CONFIG_PATH="${1:-configs/eval/robustness.yaml}"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m agent_rl.eval.robustness_eval --config "$CONFIG_PATH"
