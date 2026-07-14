#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${AGENT_RL_SYNTHETIC_ROOT:-/root/autodl-tmp/agent-rl-data/synthetic}"
TRAINING_DB_ROOT="${AGENT_RL_TRAINING_DB_ROOT:-/root/autodl-tmp/agent-rl-data/training_db}"
SEED="${SEED:-43}"
TELECOM_CLONE_FACTOR="${TELECOM_CLONE_FACTOR:-16}"
MAX_TRAIN_PER_DOMAIN="${MAX_TRAIN_PER_DOMAIN:-128}"
MAX_VALIDATION_PER_DOMAIN="${MAX_VALIDATION_PER_DOMAIN:-22}"
read -r -a DOMAINS <<< "${SYNTHETIC_DOMAINS:-airline retail}"

if [[ "${#DOMAINS[@]}" -eq 0 ]]; then
  echo "SYNTHETIC_DOMAINS must contain at least one domain" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m agent_rl.data.synthetic.training_db \
  --output-root "$TRAINING_DB_ROOT" \
  --domains "${DOMAINS[@]}" \
  --seed "$SEED" \
  --telecom-clone-factor "$TELECOM_CLONE_FACTOR"

"$PYTHON_BIN" -m agent_rl.data.synthetic.builder \
  --output-root "$OUTPUT_ROOT" \
  --domains "${DOMAINS[@]}" \
  --seed "$SEED" \
  --training-database-root "$TRAINING_DB_ROOT" \
  --telecom-clone-factor "$TELECOM_CLONE_FACTOR" \
  --max-train-per-domain "$MAX_TRAIN_PER_DOMAIN" \
  --max-validation-per-domain "$MAX_VALIDATION_PER_DOMAIN"

"$PYTHON_BIN" -m agent_rl.data.synthetic.audit \
  --corpus-root "$OUTPUT_ROOT" \
  --domains "${DOMAINS[@]}" \
  --json-output "$OUTPUT_ROOT/quality_audit.json" \
  --markdown-output "$OUTPUT_ROOT/quality_audit.md" \
  --fail-on-quality-gate

"$PYTHON_BIN" -m agent_rl.data.build_dataset synthetic \
  --corpus-root "$OUTPUT_ROOT" \
  --output "${OUTPUT_ROOT}/../balanced/train.jsonl" \
  --split train \
  --domains "${DOMAINS[@]}" \
  --seed "$SEED"

"$PYTHON_BIN" -m agent_rl.data.build_dataset synthetic \
  --corpus-root "$OUTPUT_ROOT" \
  --output "${OUTPUT_ROOT}/../balanced/validation.jsonl" \
  --split validation \
  --domains "${DOMAINS[@]}" \
  --seed "$((SEED + 100000))"

echo "Synthetic corpus: $OUTPUT_ROOT"
echo "Training databases: $TRAINING_DB_ROOT"
echo "Manifest: $OUTPUT_ROOT/manifest.json"
