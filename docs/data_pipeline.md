# Official Airline Data Pipeline

## Source of truth

`configs/data/airline_official.json` locks the upstream tau2 commit, the 30
train IDs, the 20 test IDs, and four evaluation seeds. At runtime,
`OfficialAirlineSplit.validate_against_tau2()` compares the manifest with the
installed tau2 registry and refuses to continue if they differ.

The exported JSONL rows contain only routing metadata: domain, task ID, seed,
split, and a neutral prompt. They do not copy task instructions, oracle actions,
reward criteria, database targets, or evaluator assertions into the policy
prompt.

## Training export

`build_official_train_dataset()` writes 30 rows. verl samples each row four
times, and the agent loop derives episode seeds as `base_seed + rollout_index`.
One epoch therefore contains 120 complete trajectories.

The eight-task preflight uses the first eight locked train IDs and the same
G=4 logic. It is a systems and advantage-diversity test, not a final benchmark.

## Evaluation export

`build_official_test_dataset()` pre-expands the Cartesian product of 20 test
tasks and four fixed seeds into exactly 80 rows. Evaluation uses rollout `n=1`,
so every `(task_id, seed)` pair is explicit, unique, auditable, and resumable.

Training disables `val_before_train` and sets `test_freq=-1`; therefore the
required verl validation-file field points to the train references without
actually executing validation. Official test rows are built only by an
evaluation-only manifest.

## Synthetic utilities

The clean-room synthetic generator is retained only for auxiliary analysis.
It is not imported by `verl_entry.py`, is not used by E0-E5, and must not be
reported as the formal training or evaluation source.
