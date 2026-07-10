#!/usr/bin/env bash
set -euo pipefail

VLLM_BIN="${VLLM_BIN:-/root/autodl-tmp/conda-envs/vllm/bin/vllm}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-8B}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

if [[ ! -x "$VLLM_BIN" ]]; then
    echo "vLLM executable not found: $VLLM_BIN" >&2
    exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Model directory not found: $MODEL_PATH" >&2
    exit 1
fi

exec "$VLLM_BIN" serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
