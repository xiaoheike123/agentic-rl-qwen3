# Agent RL Qwen3

An airline-only agentic reinforcement-learning study built on Qwen3-8B,
[verl](https://github.com/verl-project/verl), vLLM, and the official
[tau2-bench](https://github.com/sierra-research/tau2-bench) environment. The
project compares sequence GRPO, sign-balanced loss aggregation, and lightweight
hindsight turn credit under one locked training and evaluation protocol.

## Research scope

- **Domain:** official tau2 `airline` only.
- **Base policy:** Qwen3-8B with LoRA rank/alpha `64/64`; no SFT stage.
- **Training set:** 30 locked official airline train tasks, one row per task.
- **Final evaluation:** 20 disjoint official airline test tasks at seeds
  `42`, `43`, `44`, and `45` (80 episodes per checkpoint).
- **Training horizon:** at most 64 agent turns per episode.
- **Evaluation horizon:** tau2's 200-step text-run limit.
- **GRPO sampling:** four trajectories per task (`G=4`).
- **Formal runs:** 75 optimizer steps, saved every five steps with one retained
  actor checkpoint.

The task IDs, evaluation seeds, and pinned tau2 commit are defined in
[`configs/data/airline_official.json`](configs/data/airline_official.json).
Training code cannot read the official test split, and evaluation outcomes are
not used to continue the same training run.

## Experiment matrix

| ID | Loss aggregation | Scalar reward | Turn credit | Role |
| --- | --- | --- | --- | --- |
| E0 | none | none | none | frozen base-model evaluation |
| E1 | sequence mean | outcome | none | standard GRPO baseline |
| E2 | sign-balanced | outcome | none | balanced-GRPO comparison |
| E3 | sequence mean | outcome + process | none | process-reward ablation |
| E4 | sequence mean | outcome | hindsight | credit-only ablation |
| E5 | sign-balanced | outcome + light process | hindsight | combined method |

E1, E2, and E5 are the formal comparison. E3 and E4 isolate the two E5
components. Every trained experiment starts independently from the same base
model; E2 and E5 are not initialized from E1.

### What changes in E2 and E5

- **Sign-balanced aggregation** computes the positive- and negative-advantage
  trajectory losses separately before combining them. It changes loss
  aggregation, not tau2's binary outcome reward or PPO's token-level ratio and
  clipping.
- **Light process reward** adds a `0.1`-weighted, bounded score from
  environment-verifiable evidence such as invalid actions, tool failures,
  unresolved errors, and successful recovery. Repeated calls are diagnostic,
  not automatically penalized.
- **Hindsight credit** assigns post-trajectory turn weights using the sign of
  the GRPO advantage and process evidence, broadcasts those weights to response
  tokens, clips them to `[0.05, 3.0]`, and normalizes their trajectory mean to
  one. It is a heuristic credit assignment rule, not a learned PRM or causal
  estimator.

## Safety and comparability

- The policy receives public policy text, tool schemas, user messages, and tool
  observations only. Hidden task or evaluator fields are never rendered into
  model messages.
- Every episode creates a fresh tau2 environment and database state. Parallel
  trajectories do not share mutations.
- Asynchronous collection uses a batch barrier so all four members of a GRPO
  group come from the same frozen policy version.
- A lightweight context compressor preserves the system prompt, tool contract,
  latest user state, unresolved tool calls, and recent interaction suffix. The
  project does not implement long-term memory.
- Terminal or boundary tool calls without a tau2 result are retained as
  explicit failed evidence instead of crashing the whole rollout batch.

## Runtime

The validated remote profile targets one NVIDIA RTX PRO 6000 Blackwell GPU
(96 GB), Python 3.12, PyTorch with CUDA, vLLM `0.24.0`, and editable pinned
`tau2-bench` and `verl` submodules. DeepSeek V4 Pro is used at temperature zero
for both the tau2 user simulator and evaluator.

Clone submodules and install the unified environment:

```bash
git submodule update --init --recursive

# uv must already be available. Override ENV_PREFIX when needed.
bash scripts/setup/install_runtime.sh

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/agent-rl-train
set -a && source .env && set +a
```

The default runtime paths are under `/root/autodl-tmp`. Set at least the
DeepSeek API credentials in `.env`; never commit that file. If training is
resumed after a restart, apply
[`patches/verl_checkpoint_retention.patch`](patches/verl_checkpoint_retention.patch)
to the pinned verl submodule so the previously loaded checkpoint is removed
only after the next checkpoint is saved successfully.

## Prepare locked datasets

```bash
python -m agent_rl.data.build_dataset \
  --split train \
  --output /root/autodl-tmp/agent-rl-data/official_airline/train.jsonl

python -m agent_rl.data.build_dataset \
  --split test \
  --output /root/autodl-tmp/agent-rl-data/official_airline/test_4seed.jsonl

wc -l /root/autodl-tmp/agent-rl-data/official_airline/*.jsonl
# Expected: 30 train rows and 80 test rows.
```

These JSONL files contain task references, not hidden official task bodies.

## Validate before training

```bash
python -m compileall -q src tests
python -m pytest tests/unit tests/remote -q

# Inspect the resolved verl command without starting a GPU run.
python -m agent_rl.trainer.verl_entry \
  --config configs/train/g4_preflight.yaml \
  --dry-run

# Eight train tasks x four trajectories, one optimizer step.
bash scripts/train/train_experiment.sh configs/train/g4_preflight.yaml
```

Run the group diagnostic on the generated rollout before committing to a formal
run:

```bash
bash scripts/eval/eval_g4_preflight.sh \
  /root/autodl-tmp/agent-rl-outputs/g4_preflight/rollouts/1.jsonl
```

The full hardware and failure-recovery checklist is in
[`docs/remote_validation_checklist.md`](docs/remote_validation_checklist.md).

## Train

```bash
# Each command starts or resumes its own independent 75-step run.
bash scripts/train/run_formal_phase.sh configs/train/e1_grpo_sequence.yaml
bash scripts/train/run_formal_phase.sh configs/train/e2_grpo_balanced.yaml
bash scripts/train/run_formal_phase.sh configs/train/e5_balanced_hindsight.yaml
```

Resolved outputs default to:

```text
/root/autodl-tmp/agent-rl-outputs/<experiment>/checkpoints/
/root/autodl-tmp/agent-rl-outputs/<experiment>/rollouts/<step>.jsonl
/root/autodl-tmp/agent-rl-logs/
```

Do not run two experiments against the same output directory. A failed run may
leave an incomplete checkpoint; verify the actor files and
`latest_checkpointed_iteration.txt` before resuming.

## Export and evaluate

Export a saved FSDP actor checkpoint, then audit the actual LoRA subdirectory:

```bash
bash scripts/train/export_lora.sh \
  /root/autodl-tmp/agent-rl-outputs/e5/checkpoints/global_step_35/actor \
  /root/autodl-tmp/agent-rl-artifacts/e5_step35_export

python -m agent_rl.trainer.lora_audit \
  /root/autodl-tmp/agent-rl-artifacts/e5_step35_export/lora_adapter
```

Run the frozen adapter on the 80-row official test grid:

```bash
AGENT_RL_LORA_ADAPTER_PATH=/root/autodl-tmp/agent-rl-artifacts/e5_step35_export/lora_adapter \
  bash scripts/train/train_experiment.sh configs/train/final_lora_eval.yaml \
  trainer.experiment_name=e5_step35_final

bash scripts/eval/eval_official_airline.sh \
  /root/autodl-tmp/agent-rl-outputs/e5_step35_final/validation/0.jsonl \
  /root/autodl-tmp/agent-rl-outputs/e5_step35_final/report
```

The report command rejects incomplete or duplicated grids and writes
`summary.json`, `tasks.csv`, and `trials.csv`. Primary metrics are success rate
and tau2 `pass^1` through `pass^4`; turns, response tokens, tool errors, invalid
actions, and max-turn rate are diagnostics.

## Repository layout

- `configs/`: locked data protocol, model, environment, rollout, reward, and
  experiment manifests.
- `src/agent_rl/data/`: official split validation and verl dataset export.
- `src/agent_rl/envs/`: tau2 environment construction and action conversion.
- `src/agent_rl/rollout/`: asynchronous tau2 agent loop, context compression,
  and trajectory persistence.
- `src/agent_rl/algorithms/`: GRPO advantage and loss aggregation utilities.
- `src/agent_rl/rewards/`: outcome and deterministic process evidence.
- `src/agent_rl/credit/`: hindsight turn-level credit assignment.
- `src/agent_rl/trainer/`: verl command construction, adapters, and custom
  estimators.
- `src/agent_rl/eval/`: G=4 diagnostics and locked official evaluation.
- `tests/`: unit and remote integration coverage.
- `tau2-bench/`, `verl/`: pinned upstream submodules.

Synthetic-data utilities remain available for auxiliary analysis, but no formal
E0-E5 manifest imports synthetic tasks. Checkpoints, LoRA adapters, official
task data, API keys, and downloaded experiment artifacts are intentionally not
part of the repository.

## Limitations

This is a controlled single-domain, single-base-model study. E5's process and
hindsight signals are hand-designed and may improve credit density or stability
without outperforming standard GRPO. Claims should therefore be based on the
locked multi-seed official evaluation, not training reward curves alone.

Design rationale and known runtime issues are documented in
[`docs/project_design.md`](docs/project_design.md),
[`docs/reward_design.md`](docs/reward_design.md), and
[`docs/issue_notes.md`](docs/issue_notes.md).
