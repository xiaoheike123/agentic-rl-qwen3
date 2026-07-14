# Remote Validation Checklist

Run all commands from `/root/autodl-tmp/agent-rl-qwen3` in the unified training
environment.

## 1. Sync and activate

```bash
git pull --ff-only
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/agent-rl-train
set -a && source .env && set +a
```

## 2. Static and unit checks

```bash
python -m compileall -q src tests
python -m pytest tests/unit tests/remote -q
git submodule status
nvidia-smi
```

Expected local baseline: 76 passing tests; GPU-only remote tests add coverage
for torch, VERL, and the custom worker.

## 3. Verify locked datasets

```bash
python -m agent_rl.data.build_dataset \
  --split train \
  --output /root/autodl-tmp/agent-rl-data/official_airline/train.jsonl

python -m agent_rl.data.build_dataset \
  --split test \
  --output /root/autodl-tmp/agent-rl-data/official_airline/test_4seed.jsonl

wc -l /root/autodl-tmp/agent-rl-data/official_airline/*.jsonl
```

Expected counts are 30 and 80. Never print hidden tau2 task bodies into logs.

## 4. Dry-run command audit

```bash
python -m agent_rl.trainer.verl_entry \
  --config configs/train/g4_preflight.yaml \
  --dry-run | tee /tmp/g4_command.txt

grep -E 'lora_rank=64|rollout.n=4|use_orig_params=true|bypass_mode=false' \
  /tmp/g4_command.txt
```

## 5. G=4 preflight

```bash
bash scripts/train/train_experiment.sh configs/train/g4_preflight.yaml \
  2>&1 | tee /root/autodl-tmp/agent-rl-logs/g4_preflight.log
```

The log must contain `LORA_OPTIMIZER_AUDIT` and a post-update vLLM message such
as `vLLM load weights, loaded_params`. The checkpoint must contain
`lora_train_meta.json` with rank/alpha 64. The run must save one optimizer step.

Export and audit the adapter after locating the saved actor directory:

```bash
bash scripts/train/export_lora.sh \
  /root/autodl-tmp/agent-rl-outputs/g4_preflight/checkpoints/global_step_1/actor \
  /root/autodl-tmp/agent-rl-outputs/g4_preflight/export

python -m agent_rl.trainer.lora_audit \
  /root/autodl-tmp/agent-rl-outputs/g4_preflight/export/lora_adapter \
  --log /root/autodl-tmp/agent-rl-logs/g4_preflight.log
```

This checks rank/alpha, adapter-only tensors, at least one changed LoRA-B
tensor, optimizer audit output, vLLM adapter refresh, cache-reset configuration,
and disabled bypass mode.

Locate the rollout JSONL and run:

```bash
bash scripts/eval/eval_g4_preflight.sh \
  /root/autodl-tmp/agent-rl-outputs/g4_preflight/rollouts/0.jsonl
```

Keep G=4 only when the report recommendation is `KEEP_G4`. A lower result is a
diagnostic decision, not permission to inspect the official test set.

## 6. E0 and formal training

```bash
bash scripts/train/train_experiment.sh configs/train/e0_base_eval.yaml
bash scripts/train/train_experiment.sh configs/train/e1_grpo_sequence.yaml
```

E2-E4 use their matching manifests. Each starts from the unchanged base model
and writes to its own output directory. Do not resume E2 from E1.

## 7. Frozen LoRA evaluation

```bash
AGENT_RL_LORA_ADAPTER_PATH=/path/to/exported/adapter \
  bash scripts/train/train_experiment.sh configs/train/final_lora_eval.yaml \
  trainer.experiment_name=e1_final

bash scripts/eval/eval_official_airline.sh \
  /root/autodl-tmp/agent-rl-outputs/e1_final/validation/0.jsonl \
  /root/autodl-tmp/agent-rl-outputs/e1_final/report
```

The evaluator rejects anything other than the exact 20-task x four-seed grid.
Archive `summary.json`, `tasks.csv`, `trials.csv`, the resolved command, git
commit, submodule commits, model checksum, and SwanLab run URL.

## 8. Shutdown safety

Before stopping the instance, verify checkpoints and reports exist on the data
disk, push code commits, and copy irreplaceable logs off the system disk.
