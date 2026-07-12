# Remote Validation Checklist

Target host:

- RTX PRO 6000 Blackwell 96GB
- driver 580.95.05, CUDA capability 13.0
- Python 3.12
- repository `/root/autodl-tmp/agent-rl-qwen3`
- model `/root/autodl-tmp/models/Qwen3-8B`

Do not start a standalone port-8000 vLLM server during training. verl owns the
training rollout engine and synchronizes its weights.

## 1. Restore repository

```bash
cd /root/autodl-tmp/agent-rl-qwen3
git pull --ff-only
git submodule update --init --recursive
git status --short
```

Expected: no unexpected local changes.

## 2. Verify the existing standalone inference runtime

```bash
PY=/root/autodl-tmp/conda-envs/vllm/bin/python
$PY -c "import torch,vllm; print(torch.__version__, vllm.__version__); print(torch.cuda.get_device_name()); assert torch.cuda.is_available()"
nvidia-smi
df -h /root/autodl-tmp
```

Expected: Qwen-capable vLLM imports, PRO 6000 visible, enough data-disk space.

The pinned verl 0.9.0.dev metadata declares vLLM `<=0.12.0`, while its checked
in async-server source explicitly handles vLLM v0.20+ behavior. Because this
metadata and implementation disagree, the training environment pins the
already proven vLLM 0.24.0 and installs verl with `--no-deps`. Keep the
standalone environment unchanged as a fallback and comparison point.

## 3. Install the separate unified training runtime

```bash
bash scripts/setup/install_runtime.sh
```

This creates `/root/autodl-tmp/conda-envs/agent-rl-train` with vLLM 0.24.0,
torch selected for the host CUDA driver, tau2, verl, and the project. It does
not modify the working standalone environment. The one-step smoke is the
authority on whether this exact pinned pair is runtime-compatible.

## 4. Load the DeepSeek secret

Create `.env` in the project root with `DEEPSEEK_API_KEY=...`. The training
launcher exports it to Ray workers. Then verify without printing the key:

```bash
set -a; source .env; set +a
test -n "${DEEPSEEK_API_KEY:-}" && echo DEEPSEEK_KEY_OK
```

## 5. Static and unit validation

```bash
PY=/root/autodl-tmp/conda-envs/agent-rl-train/bin/python
$PY -m compileall -q src tests
$PY -m pytest tests/unit -q
$PY -m pytest tests/remote -q
$PY -c "import agent_rl.rollout.tau_agent_loop; import agent_rl.trainer.tau_agent_loop_manager; import agent_rl.trainer.verl_algorithms; print('VERL_EXTENSIONS_OK')"
tau2 check-data
git diff --check
```

Expected: compilation and every unit test pass.

## 6. tau2 identity-only datasets

```bash
$PY -m agent_rl.data.build_dataset --domain airline --split train --output /root/autodl-tmp/agent-rl-data/airline/train.jsonl
$PY -m agent_rl.data.build_dataset --domain airline --split validation --output /root/autodl-tmp/agent-rl-data/airline/validation.jsonl
head -n 1 /root/autodl-tmp/agent-rl-data/airline/train.jsonl
```

Expected: rows contain domain/task IDs and dummy prompts, never evaluation
criteria or hidden answers.

## 7. Resolve the E1 command without training

```bash
$PY -m agent_rl.trainer.verl_entry --config configs/train/e1_grpo_sequence.yaml --dry-run
```

Check model path, dataset paths, `algorithm.adv_estimator=grpo`,
`loss_agg_mode=seq-mean-token-mean`, full-run group size 8, and one GPU.

## 8. One-step E1 training smoke

Before updating weights, record the exact-pipeline E0 baseline:

```bash
WANDB_MODE=offline bash scripts/train/train_experiment.sh \
  configs/train/e0_base_eval.yaml
```

This uses the same TauAgentLoop and prompt as training. The older `tau2 run`
smoke remains useful for official-stack diagnostics, but it is not the E0
research baseline because its agent prompt implementation differs.

```bash
WANDB_MODE=offline bash scripts/train/train_experiment.sh \
  configs/train/e1_grpo_sequence.yaml \
  trainer.total_training_steps=1 \
  data.train_batch_size=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.rollout.n=4
```

Acceptance checks:

- verl launches one vLLM rollout engine;
- four smoke rollouts share one `uid` (full experiments use eight);
- DeepSeek user calls succeed;
- tau2 returns rewards;
- actor loss and gradient update complete;
- checkpoint/output stays under `/root/autodl-tmp`.

## 9. Algorithm smoke matrix

Run one step for E2-E5 using the same overrides. Confirm:

- E2: native GRPO + `token-mean`;
- E3: `tau_balanced_grpo`;
- E4: E3 plus nonzero process-reward metrics;
- E5: `tau_hindsight_balanced_grpo` and nonuniform credit when evidence differs.

Also inspect `tau_context_rotations`, `tau_total_turns`, and
`tau_retained_credit_turns`. Context rotation is truncated backpropagation:
turns removed from the active token window remain in reward computation but do
not receive token gradients. Report the rotation rate and retained-turn ratio;
do not describe this as full-history credit assignment.

## 10. Full experiments

Run E1-E5 separately with their unmodified manifests. Before each run record
the Git commit, submodule commits, model hash, seed, prompt hash, and config.
Use W&B for curves and keep checkpoints on the data disk.

The 100GB data disk is suitable for one active experiment, not an unlimited
archive of full checkpoints from E1-E5. The runtime retains one actor
checkpoint per experiment; archive final adapters/metrics and remove obsolete
optimizer checkpoints before starting the next full run, or expand the disk.

## 11. Independent clean evaluation

Stop training, start the standalone evaluation server, then run tau2:

```bash
bash scripts/vllm/serve_qwen3_8b.sh
# In another terminal:
bash scripts/eval/eval_tau.sh
```

## 12. Robustness evaluation

Use a reviewed JSONL manifest containing meaning-preserving paraphrase and
information-order variants. Run clean and all three perturbations with the
same task IDs, seeds, prompt, sampling settings, and checkpoint. Report
perturbed success, success drop, recovery success, and extra steps.

## 13. Before shutting down AutoDL

```bash
git status --short
df -h /root/autodl-tmp
find /root/autodl-tmp/agent-rl-outputs -maxdepth 3 -type f | head
```

Push source/config changes and confirm checkpoints, W&B offline logs, datasets,
and evaluation outputs are under `/root/autodl-tmp`, not the 30GB system disk.
