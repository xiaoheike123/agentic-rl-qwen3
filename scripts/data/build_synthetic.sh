#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${AGENT_RL_SYNTHETIC_ROOT:-/root/autodl-tmp/agent-rl-data/synthetic}"
SEED="${SEED:-42}"
MAX_PER_SPLIT_PER_DOMAIN="${MAX_PER_SPLIT_PER_DOMAIN:-128}"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m agent_rl.data.synthetic.builder \
  --output-root "$OUTPUT_ROOT" \
  --domains airline retail telecom \
  --seed "$SEED" \
  --max-per-split-per-domain "$MAX_PER_SPLIT_PER_DOMAIN"

"$PYTHON_BIN" -m agent_rl.data.build_dataset synthetic \
  --corpus-root "$OUTPUT_ROOT" \
  --output "${OUTPUT_ROOT}/../balanced/train.jsonl" \
  --split train \
  --seed "$SEED"

"$PYTHON_BIN" -m agent_rl.data.build_dataset synthetic \
  --corpus-root "$OUTPUT_ROOT" \
  --output "${OUTPUT_ROOT}/../balanced/validation.jsonl" \
  --split validation \
  --seed "$((SEED + 100000))"

echo "Synthetic corpus: $OUTPUT_ROOT"
echo "Manifest: $OUTPUT_ROOT/manifest.json"
