# Algorithm Notes

The optimizer comparison uses three GRPO loss reductions:

- Sequence aggregation is the vanilla GRPO baseline.
- Token aggregation is the DAPO-style long-response baseline.
- Balanced aggregation separates positive and negative advantages before
  combining token means with sequence-count weights.

The project contribution is not a new name for aggregation. It combines
environment-verifiable process rewards with hindsight turn-level credit for
long tool-agent trajectories. Async rollout is evaluated separately as a
systems optimization.
