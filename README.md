# Agent RL Qwen3

LoRA-GRPO research on the official tau2 `airline` environment with Qwen3-8B,
verl, vLLM, and a DeepSeek V4 Pro user simulator.

## Locked protocol

- Domain: `airline` only.
- Training data: the 30 official tau2 airline `train` task IDs.
- Final evaluation: the 20 disjoint official airline `test` task IDs.
- Final trials: four fixed seeds per test task, 80 episodes per checkpoint.
- Training: independent 75-step LoRA-GRPO runs from the same Qwen3-8B base.
- GRPO group size: `G=4`, subject to an eight-task preflight diagnostic.
- Official metric: tau2 `pass^1` through `pass^4`, plus success and efficiency diagnostics.

The locked IDs and seeds live in
`configs/data/airline_official.json`. Training never reads the official test
split, and test results are not used to decide whether to train another epoch.

## Experiment matrix

| ID | Aggregation | Scalar reward | Turn credit |
| --- | --- | --- | --- |
| E0 | none | none | base-model final evaluation |
| E1 | sequence | outcome | none |
| E2 | balanced | outcome | none |
| E3 | sequence | outcome + environment process | ablation template |
| E4 | sequence | outcome | ablation template |
| E5 | balanced | outcome + light process | hindsight turn credit |

The formal three-run plan is E1, E2, and E5. E3 and E4 remain available as
ablation templates. Every trained experiment starts independently from the same
base model; E2 is not initialized from E1.

## Main commands

```bash
# Eight train tasks x four rollouts, one update.
bash scripts/train/train_experiment.sh configs/train/g4_preflight.yaml

# Formal E1 phase A: train from scratch to step 30.
bash scripts/train/run_formal_phase.sh configs/train/e1_grpo_sequence.yaml 30

# Formal E1 phase B: resume from step 30 and train to step 75.
bash scripts/train/run_formal_phase.sh configs/train/e1_grpo_sequence.yaml 75

# Base-model final evaluation (80 pre-expanded rows).
bash scripts/train/train_experiment.sh configs/train/e0_base_eval.yaml

# Evaluate a trained LoRA adapter.
AGENT_RL_LORA_ADAPTER_PATH=/path/to/adapter \
  bash scripts/train/train_experiment.sh configs/train/final_lora_eval.yaml \
  trainer.experiment_name=e1_final
```

Use `scripts/eval/eval_g4_preflight.sh` for the group diagnostic and
`scripts/eval/eval_official_airline.sh` to produce the final JSON/CSV report.

## Repository layout

- `configs/`: locked data, model, runtime, rollout, and experiment manifests.
- `src/agent_rl/data/`: official split validation and verl dataset export.
- `src/agent_rl/rollout/`: tau2 agent loop and vLLM interaction.
- `src/agent_rl/rewards/`: environment-verifiable process evidence.
- `src/agent_rl/credit/`: hindsight turn-level credit.
- `src/agent_rl/trainer/`: verl command construction and custom estimators.
- `src/agent_rl/eval/`: G=4 diagnostics and official four-seed evaluation.
- `tau2-bench/`, `verl/`: pinned upstream submodules.

Synthetic-data utilities remain available for auxiliary analysis, but no formal
E0-E5 configuration imports or trains on them.
