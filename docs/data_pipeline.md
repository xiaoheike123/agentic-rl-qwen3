# Synthetic Data Pipeline

Formal E1-E5 training never consumes the public tau2 `base` tasks. The data
pipeline builds independent task files for airline, retail, and telecom and
then exports a domain-balanced verl dataset.

## Guarantees

1. Generated records contain native `tau2.data_model.tasks.Task` payloads.
2. Every oracle action executes without error and changes the environment DB.
3. Exact task payloads, identical action arguments, and high-similarity tasks
   are rejected against the official `base` split.
4. Train and validation are separated by their primary customer/user entity.
5. The verl export contains equal numbers of airline, retail, and telecom rows.
6. Official `base` IDs are exported only by the `official-eval` command.

## Build

```bash
bash scripts/data/build_synthetic.sh
```

The default cap is 128 accepted records per split and domain. Override it with
`MAX_PER_SPLIT_PER_DOMAIN`; the builder stops verifying a split after its cap
is reached instead of repeatedly cloning a large environment database.

For the pinned tau2 data and seed 42, the current generators produce 128
training records per domain. Telecom has fewer independent customer groups and
produces 22 validation records, so balanced export contains 384 training rows
and 66 validation rows. The manifest remains the authority if upstream data
changes.

Outputs default to:

```text
/root/autodl-tmp/agent-rl-data/
├── synthetic/
│   ├── airline/{train,validation}.jsonl
│   ├── retail/{train,validation}.jsonl
│   ├── telecom/{train,validation}.jsonl
│   ├── manifest.json
│   └── rejections.jsonl
└── balanced/
    ├── train.jsonl
    └── validation.jsonl
```

`manifest.json` is the audit artifact. A task rejected for overlap is recorded
by ID and reason, but official task contents are never copied into training
files.

Training validates the manifest seed, domains, split ratio, overlap threshold,
and per-domain cap before launching. Configuration drift fails fast instead of
silently reusing a different corpus.

## Evaluation Boundary

The official leaderboard set is exported separately:

```bash
python -m agent_rl.data.build_dataset official-eval \
  --output /root/autodl-tmp/agent-rl-data/official_eval/base.jsonl \
  --domains airline retail telecom \
  --split base
```

Do not use that file for training, checkpoint selection, or hyperparameter
tuning.
