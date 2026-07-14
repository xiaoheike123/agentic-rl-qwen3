#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/agent-rl-train/bin/python}"
INPUT="${1:?usage: eval_official_airline.sh VALIDATION_JSONL OUTPUT_DIR}"
OUTPUT_DIR="${2:?usage: eval_official_airline.sh VALIDATION_JSONL OUTPUT_DIR}"

exec "$PYTHON_BIN" -m agent_rl.eval.official_airline \
    "$INPUT" \
    --output-dir "$OUTPUT_DIR"
