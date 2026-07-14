# PRO 6000 Performance Validation Checklist

This checklist tunes execution without shortening context, reducing G, dropping
trajectories, changing rewards, or changing the official airline task split.
Formal E1-E5 remain locked to `bypass_mode=false` until the bypass experiment is
reviewed and the protocol is deliberately amended.

## 1. Invariants

Every probe must retain all of the following:

- Qwen3-8B with LoRA rank/alpha 64/64 and the locked target modules.
- Eight official-airline training tasks with four rollouts each (32 episodes).
- 16 agent turns, 256 action tokens per turn, and a 16K model token budget.
- Identical prompt, tool, reward, context-compression, sampling, and seed rules.
- One episode per agent-loop worker and no official test evaluation.
- One PPO epoch and no resume from a previous probe.

The probes may change only scheduling, batch capacity, episode concurrency, or
old-logprob recomputation.

## 2. Before Every Probe

Run from `/root/autodl-tmp/agent-rl-qwen3`:

```bash
git pull --ff-only
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/agent-rl-train
set -a && source .env && set +a

ray stop --force || true
nvidia-smi
python -m pytest tests/unit/test_verl_evaluation_config.py -q
mkdir -p /root/autodl-tmp/agent-rl-logs/performance
```

Do not run two probes at once. Move an existing output directory aside before
rerunning the same probe; the probe runtimes deliberately disable resume.

## 3. Probe Matrix

| Probe | Config | Difference from previous probe |
|---|---|---|
| P0 | `g4_preflight.yaml` | Stable baseline: concurrency 8, micro batch 1, 16K batch-token budgets |
| P1 | `g4_perf_tokens32k.yaml` | P0 plus 32K vLLM/PPO batch-token budgets |
| P2 | `g4_perf_micro2.yaml` | P1 plus micro batch 2 |
| P3 | `g4_perf_concurrency16.yaml` | P2 plus 16 isolated workers/episodes |
| B1 | `g4_bypass_on.yaml` | P3 plus rollout-logprob bypass; P3 is the bypass-off and all-on engineering control |

Run one probe in a detached screen session. Replace `CONFIG` and `LABEL` with
one row from the commands below:

```bash
screen -dmS perf_p0 bash -lc '
cd /root/autodl-tmp/agent-rl-qwen3
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/agent-rl-train
set -a && source .env && set +a
bash scripts/eval/run_performance_probe.sh \
  configs/train/g4_preflight.yaml p0
'

tail -f /root/autodl-tmp/agent-rl-logs/performance/p0.log
```

The remaining command arguments are:

```text
configs/train/g4_perf_tokens32k.yaml       p1
configs/train/g4_perf_micro2.yaml          p2
configs/train/g4_perf_concurrency16.yaml   p3
configs/train/g4_bypass_on.yaml            b1
```

The wrapper saves the combined run log, GNU time report, and a two-second GPU
utilization/VRAM trace under `agent-rl-logs/performance/`.

## 4. Mandatory Integrity Checks

For every successful probe:

```bash
grep -E "Traceback|CUDA out of memory|429|timed out|EpisodeDataError" \
  /root/autodl-tmp/agent-rl-logs/performance/LABEL.log || true

grep "Maximum resident set size" \
  /root/autodl-tmp/agent-rl-logs/performance/LABEL_time.txt

bash scripts/eval/eval_g4_preflight.sh \
  /root/autodl-tmp/agent-rl-outputs/EXPERIMENT/rollouts/0.jsonl
```

Use these experiment directories:

```text
P0: g4_preflight
P1: g4_perf_tokens32k
P2: g4_perf_micro2
P3: g4_perf_concurrency16
B1: g4_bypass_on
```

A probe fails immediately if it has an OOM, NaN/Inf loss, incomplete G=4
group, missing tool result, database-state collision, sustained API rate limit,
or nonzero process exit. Peak GPU memory must leave at least 5 GiB free.

Record the following from GNU time, the GPU CSV, the run log, and SwanLab:

| Field | P0 | P1 | P2 | P3 | B1 |
|---|---:|---:|---:|---:|---:|
| Wall time | | | | | |
| Peak GPU memory MiB | | | | | |
| Mean GPU utilization | | | | | |
| `timing_s/gen` | | | | | |
| `timing_s/old_log_prob` | | | | | skipped |
| `timing_s/update_actor` | | | | | |
| API retries/timeouts | | | | | |
| Complete G=4 groups | | | | | |

## 5. Scheduling Decisions

### P1 versus P0: token budgets

Accept 32K vLLM/PPO batch-token budgets when P1:

- passes all integrity checks;
- does not increase peak GPU memory above the 5-GiB reserve;
- reduces `timing_s/gen` or `timing_s/update_actor` by at least 10%, or reduces
  total wall time by at least 5% without a reliability regression.

Otherwise keep the P0 token budgets.

### P2 versus P1: micro batch

Accept micro batch 2 when P2 passes all integrity checks, keeps the 5-GiB VRAM
reserve, and reduces actor/update time by at least 10%. Otherwise retain micro
batch 1 while keeping the independently accepted token-budget result.

### P3 versus P2: concurrency

Accept concurrency 16 when P3:

- passes all integrity checks;
- has no sustained 429 response and an infrastructure retry rate below 1%;
- reduces rollout wall time by at least 15%.

Otherwise use concurrency 8. A concurrency change is operational only, but the
chosen value must be identical across formal E1-E5 runs.

## 6. Bypass A/B Decision

P3 is the bypass-off control and B1 changes only bypass. P3 is also the maximum
combination that preserves the standard GRPO objective; B1 is the maximum
combination including experimental bypass. Before considering B1, inspect P3's
verl metrics:

```text
training/rollout_probs_diff_valid
training/rollout_probs_diff_mean
training/rollout_probs_diff_max
training/rollout_actor_probs_pearson_corr
timing_s/old_log_prob
```

Bypass is a candidate only when:

- the diff metric is valid and finite;
- mean rollout/actor probability difference is at most `1e-3`;
- Pearson correlation is at least `0.99`;
- B1 has finite loss, KL, clip fraction, and gradient norm;
- B1 skips `timing_s/old_log_prob` and improves total training-step time by at
  least 10%;
- all integrity checks still pass.

Passing does not automatically modify E1-E5. Enabling bypass changes the PPO
anchor from recomputed actor probabilities to rollout-engine probabilities, so
it requires one explicit protocol decision, the same setting for E1-E5, and a
written disclosure in the experiment report. The conservative formal setting
remains bypass off.

## 7. Final Selection

Write the selected execution profile and all measurements into the experiment
record before E1 begins. Once E1 starts, do not retune scheduling between E1,
E2, E3, and E4. Never use the official test results to choose the performance
profile.
