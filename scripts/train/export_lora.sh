#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/agent-rl-train/bin/python}"
CHECKPOINT="${1:?usage: export_lora.sh ACTOR_CHECKPOINT_DIR TARGET_DIR}"
TARGET="${2:?usage: export_lora.sh ACTOR_CHECKPOINT_DIR TARGET_DIR}"

exec "$PYTHON_BIN" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$CHECKPOINT" \
    --target_dir "$TARGET"
