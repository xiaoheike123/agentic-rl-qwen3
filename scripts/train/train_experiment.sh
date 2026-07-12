#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/train/e1_grpo_sequence.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/agent-rl-train/bin/python}"
shift || true

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

exec "$PYTHON_BIN" -m agent_rl.trainer.verl_entry \
    --config "$CONFIG_PATH" \
    "$@"
