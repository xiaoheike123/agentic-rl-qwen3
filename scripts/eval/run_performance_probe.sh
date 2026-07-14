#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:?usage: run_performance_probe.sh CONFIG_PATH LABEL}"
LABEL="${2:?usage: run_performance_probe.sh CONFIG_PATH LABEL}"
LOG_ROOT="${AGENT_RL_LOG_ROOT:-/root/autodl-tmp/agent-rl-logs/performance}"
mkdir -p "$LOG_ROOT"

RUN_LOG="$LOG_ROOT/${LABEL}.log"
GPU_LOG="$LOG_ROOT/${LABEL}_gpu.csv"
TIME_LOG="$LOG_ROOT/${LABEL}_time.txt"

nvidia-smi \
  --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv \
  --loop=2 >"$GPU_LOG" 2>&1 &
MONITOR_PID=$!

cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "config=$CONFIG_PATH"
echo "label=$LABEL"
echo "run_log=$RUN_LOG"
echo "gpu_log=$GPU_LOG"
echo "time_log=$TIME_LOG"

set +e
/usr/bin/time -v -o "$TIME_LOG" \
  bash scripts/train/train_experiment.sh "$CONFIG_PATH" \
  2>&1 | tee "$RUN_LOG"
STATUS=${PIPESTATUS[0]}
set -e

echo "exit_status=$STATUS" | tee -a "$RUN_LOG"
exit "$STATUS"
