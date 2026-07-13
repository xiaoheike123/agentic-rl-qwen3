# Project Design

This project trains Qwen3-8B as a robust long-horizon tool agent on
tau2/tau3-bench.

High-level data flow:

```text
three domain generators -> oracle verification -> base-overlap filter
         -> domain-balanced synthetic task -> AgentGymEnv
         -> policy action -> trajectory evidence
         -> outcome/process reward -> hindsight turn credit
         -> GRPO aggregation -> verl update
```

Official `base` tasks are loaded through a separate evaluation-only exporter.
The training environment receives a complete native tau2 `Task` payload from
each synthetic dataset row, so it never resolves a training ID through the
official task registry.

Training uses a custom verl `AgentLoop`, so rollout tokens are sampled from the
same policy version that is updated. The separate OpenAI-compatible
`VLLMPolicy` is an evaluation adapter and is not the training data source.
Long episodes use deterministic rolling prompt reconstruction; no persistent
memory or auxiliary summarization model is involved.

Module boundaries:

- `rewards/` determines where verifiable reward signals come from.
- `credit/` determines which turns receive those signals.
- `algorithms/` determines how masked token losses are aggregated.
- `rollout/` collects trajectories; async scheduling must not own algorithm logic.
- `robustness/` perturbs evaluation interactions without changing task goals.
- `data/synthetic/` owns generation, oracle verification, overlap filtering,
  entity-disjoint splits, and balanced export.
