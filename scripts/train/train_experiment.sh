#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/train/e1_grpo_sequence.yaml}"

exec python -m agent_rl.trainer.verl_entry \
    --config "$CONFIG_PATH" \
    "${@:2}"
