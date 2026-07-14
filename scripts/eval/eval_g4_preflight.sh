#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/agent-rl-train/bin/python}"
INPUT="${1:?usage: eval_g4_preflight.sh ROLLOUT_JSONL [OUTPUT_JSON]}"
OUTPUT="${2:-/root/autodl-tmp/agent-rl-outputs/g4_preflight/group_diagnostics.json}"

exec "$PYTHON_BIN" -m agent_rl.eval.group_diagnostics \
    "$INPUT" \
    --group-size 4 \
    --output "$OUTPUT"
