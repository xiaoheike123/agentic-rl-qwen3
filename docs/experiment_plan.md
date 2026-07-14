# Final Experiment Plan

## Data boundary

The project uses only the upstream tau2 airline split:

- Train: 30 official `train` tasks.
- Test: 20 disjoint official `test` tasks.
- Final evaluation: four fixed seeds for every test task.

The test set is touched only after a checkpoint and its inference settings are
frozen. It must not select epochs, hyperparameters, prompts, or checkpoints.
Any decision to extend beyond one epoch uses train-only diagnostics such as
reward, nonzero-advantage coverage, KL, entropy, clipping ratio, and stability.

## Preflight

Run eight train tasks with four rollouts each. The report groups all 32
trajectories by task and measures the fraction of groups with nonzero reward
variance:

- At least 50%: keep `G=4`.
- 30% to 50%: first adjust sampling or prompt behavior and repeat G=4.
- Below 30%: diagnose collapse, then consider G=6 or G=8.

The same smoke run must show that only LoRA parameters are trainable, an
optimizer step changes adapter weights, verl refreshes the vLLM adapter, and
the rollout cache does not preserve the pre-update adapter.

## Training constants

- Base model: Qwen3-8B.
- LoRA rank/alpha: 64/64.
- Targets: q/k/v/o and gate/up/down projections.
- One RTX PRO 6000 96GB, tensor parallel size 1.
- 30 prompts, G=4, 120 trajectories per epoch.
- PPO epochs 1; formal training target 75 global steps.
- Learning rate 1e-5; gradient clipping 1.0; KL coefficient 0.001.
- Temperature 0.8; top-p 0.95.
- Maximum 64 agent turns during training and 256 generated tokens per turn.
- Frozen E0 and checkpoint evaluation use tau2's 200-step text-run limit.
- 16K total prompt/response budget.

## Checkpoint schedule

Formal training is launched in two phases so that checkpoints are written only
at the intended decision points:

- Phase A: train from scratch to global step 30 and save step 30.
- Phase B: resume from step 30, train to global step 75, and save step 75.

Use `scripts/train/run_formal_phase.sh` for both phases. On the default 100GB
AutoDL data disk, `CKPT_KEEP=1` is the safe setting because one verl training
checkpoint is roughly one full 8B-model training state. Set `CKPT_KEEP=2` only
after expanding the disk or moving the step-30 checkpoint elsewhere.

## Experiments

1. E0: frozen base-model test evaluation.
2. E1: outcome GRPO with sequence aggregation.
3. E2: outcome GRPO with balanced aggregation.
4. E5: balanced aggregation plus lightweight environment-process reward and
   hindsight turn credit.

E1, E2, and E5 are the three formal training runs. They are independent runs
from the identical base model and fixed data order. E3 and E4 configs are kept
as ablation templates, but they are not part of the default three-run plan.
Using test results to decide which extra runs to launch would turn the same test
set into a tuning set. Async rollout is a systems optimization and must not
change policy version within one collected batch.

## Reporting

For every frozen checkpoint, run exactly 20 tasks x 4 seeds. Report:

- success rate;
- tau2 pass^1, pass^2, pass^3, pass^4;
- mean turns and generated tokens;
- mean tool calls, tool-error rate, invalid-action rate;
- maximum-turn rate;
- per-task and per-seed CSV files.

Robustness perturbations are secondary evaluations and use frozen selected
checkpoints only.
