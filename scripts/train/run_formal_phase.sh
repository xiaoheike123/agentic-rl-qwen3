#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:?usage: run_formal_phase.sh <config>}"

if [[ $# -ne 1 ]]; then
    echo "usage: run_formal_phase.sh <config>" >&2
    exit 2
fi

exec bash scripts/train/train_experiment.sh "$CONFIG_PATH" \
    trainer.total_epochs=75 \
    trainer.save_freq=5 \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=1
