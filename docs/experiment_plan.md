# Experiment Plan

Build order:

1. Define the trajectory schema and JSONL persistence.
2. Build and audit three-domain synthetic train/validation data.
3. Implement one synchronous episode worker and batch collector.
4. Freeze the official `base` evaluation protocol and record E0 once.
5. Integrate verl and run E1 vanilla GRPO with sequence aggregation.
6. Compare E2 token and E3 balanced aggregation under identical settings.
7. Add environment-verifiable process rewards for E4.
8. Add hindsight turn-level credit assignment for E5.
9. Compare synchronous and asynchronous rollout throughput as S0/S1.
10. Evaluate frozen E0, E3, and E5 under interaction perturbations.

All E1-E5 experiments use the same generated corpus and the same balanced
airline/retail/telecom row order. Synthetic validation is used for checkpoint
selection. Official `base` results are not used for tuning.

Algorithm experiments:

- `E0`: Base Qwen3, no RL.
- `E1`: Vanilla GRPO with sequence aggregation.
- `E2`: GRPO with token aggregation.
- `E3`: GRPO with balanced aggregation.
- `E4`: E3 plus environment-verifiable process reward.
- `E5`: E4 plus hindsight turn-level credit.

System experiments:

- `S0`: Fixed-policy episode collection with a batch barrier.
- `S1`: Concurrent fixed-policy episode collection with ready-group streaming.

S0 and S1 use the same frozen policy version for every compared batch. Policy
staleness is deliberately excluded, so this axis measures throughput rather
than changing the optimization objective.

Robustness evaluation:

- `R0`: Clean interactions.
- `R1`: User paraphrase.
- `R2`: Information-order shift.
- `R3`: Recoverable tool failure.
