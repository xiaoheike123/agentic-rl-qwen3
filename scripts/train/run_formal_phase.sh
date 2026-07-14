#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:?usage: run_formal_phase.sh <config> <30|75>}"
PHASE="${2:?usage: run_formal_phase.sh <config> <30|75>}"
CKPT_KEEP="${CKPT_KEEP:-1}"

case "$PHASE" in
    30)
        TOTAL_EPOCHS=30
        SAVE_FREQ=30
        RESUME_MODE=disable
        ;;
    75)
        TOTAL_EPOCHS=75
        SAVE_FREQ=75
        RESUME_MODE=auto
        ;;
    *)
        echo "phase must be 30 or 75, got: $PHASE" >&2
        exit 2
        ;;
esac

exec bash scripts/train/train_experiment.sh "$CONFIG_PATH" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.max_actor_ckpt_to_keep="$CKPT_KEEP"
