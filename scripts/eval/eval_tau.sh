#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/agent-rl/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/eval/qwen3_8b_deepseek_v4_airline.yaml}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Evaluation config not found: $CONFIG_PATH" >&2
    exit 1
fi

cd "$PROJECT_ROOT"

exec "$PYTHON_BIN" \
    -m agent_rl.eval.tau_eval \
    --config "$CONFIG_PATH"

