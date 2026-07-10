#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8000}"
vllm serve "$MODEL" --port "$PORT"

