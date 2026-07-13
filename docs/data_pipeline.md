# Synthetic Data Pipeline

Formal E1-E5 training never consumes the public tau2 `base` tasks. The data
pipeline builds independent task files for airline, retail, and telecom and
then exports a domain-balanced verl dataset.

## Guarantees

1. Generated records contain native `tau2.data_model.tasks.Task` payloads.
2. Every Oracle trajectory executes without error and changes assistant or
   user-device environment state.
3. Generation reads policies, tools, schemas, and domain databases, but never
   reads official task files or trajectories.
4. Train and validation are separated by their primary customer/user entity.
5. The verl export contains equal numbers of airline, retail, and telecom rows.
6. Official `base` IDs are exported only by the `official-eval` command.
7. Private policy metadata and evaluation criteria never enter model messages.
8. Synthetic task generation and rollout use deterministic pseudonymized
   training databases; official identifiers have zero overlap with training
   identifiers.
9. Every episode receives a deep copy of its domain database, so concurrent
   rollouts cannot mutate each other or the JSON source file.

Each domain has an independent policy validator. Airline recomputes fare,
baggage, cancellation, and payment constraints. Retail verifies authentication,
order state, item/product compatibility, refund destinations, and confirmation.
Telecom verifies account writes and clean-room device fault/fix compositions.

## Build

```bash
bash scripts/data/build_synthetic.sh
```

The default caps are 128 training and 22 validation records per domain. Override
them with `MAX_TRAIN_PER_DOMAIN` and `MAX_VALIDATION_PER_DOMAIN`; the builder
stops verifying a split after its own cap is reached.

For the pinned tau2 data and synthetic seed 43, the formal target is 128
training and 22 validation records per domain, so the balanced export contains
384 training and 66 validation rows. Training randomness remains independently
seeded at 42. The manifest remains the authority if upstream data changes.

Outputs default to:

```text
/root/autodl-tmp/agent-rl-data/
+-- training_db/
|   +-- airline/db.json
|   +-- retail/db.json
|   +-- telecom/db.json
|   +-- manifest.json
+-- synthetic/
|   +-- airline/{train,validation}.jsonl
|   +-- retail/{train,validation}.jsonl
|   +-- telecom/{train,validation}.jsonl
|   +-- manifest.json
|   +-- rejections.jsonl
+-- balanced/
    +-- train.jsonl
    +-- validation.jsonl
```

`manifest.json` records build configuration and acceptance counts.
`rejections.jsonl` records policy, Oracle, and duplicate failures without
copying official task contents. The build script and training entry run the
corpus audit automatically; it checks
template concentration, parameter variants, compositional near-duplicates,
entity isolation, policy metadata, Oracle verification, and long-horizon
coverage.

Training validates the manifest seed, domains, split ratio, generator version,
and per-domain cap before launching. Configuration drift fails fast instead of
silently reusing a different corpus.

## Training Database Boundary

The builder reads official domain database schemas and policy-relevant state,
then deterministically replaces user, reservation, order, item, payment,
flight, customer, line, bill, device, plan, and phone identifiers. Personal
fields are replaced as well. Telecom is cloned into independent pseudonymous
entities to avoid training on only four source customers.

The training-database manifest records source and generated hashes, entity
statistics, and an explicit zero-overlap check. Its portable fingerprint is
copied into the corpus manifest and every balanced verl row. This gives each
checkpoint an auditable data provenance chain without exposing database
contents to the policy.

This boundary prevents entity memorization and cross-episode state pollution.
It does not claim that arbitrary generated language can be mathematically
proven dissimilar to unseen official tasks. That contamination boundary is
maintained by clean-room generation: the pipeline never loads official task
files or trajectories.

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

Official evaluation rows do not contain `synthetic_task`, so TauAgentLoop does
not inject a training database for them. They continue through tau2's original
environment constructor and original database.
