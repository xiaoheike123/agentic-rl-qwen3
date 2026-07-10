#!/usr/bin/env bash
set -euo pipefail
python -m agent_rl.trainer.verl_entry --config configs/train/hgpo_qwen3_8b_tau.yaml "$@"

