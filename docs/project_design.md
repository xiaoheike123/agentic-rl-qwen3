# Project Design

## Scope

The formal project studies credit assignment for long-horizon tool agents in
one controlled environment: official tau2 airline. Restricting the domain keeps
the data boundary, simulator behavior, reward semantics, and evaluation budget
constant while the algorithm changes.

## Runtime path

```text
locked official train task ID
-> verl repeats prompt four times
-> TauAgentLoop receives rollout index and derives an episode seed
-> Qwen3-8B LoRA policy generates one action at a time through vLLM
-> tau2 runs a fresh per-episode airline environment and DeepSeek V4 Pro user
-> tau2 outcome plus optional environment process evidence
-> sequence, balanced, or hindsight-weighted GRPO update
-> adapter weights synchronize back into vLLM
```

The policy receives only the domain policy, public tool schemas, user messages,
and tool observations. Oracle actions, target database state, evaluator
criteria, and hidden task fields remain environment-side.

## Algorithm separation

- Aggregation controls how trajectory token losses are reduced.
- Process reward controls the scalar reward used by E3.
- Hindsight credit controls which generated turns receive more or less of the
  trajectory-level GRPO advantage in E4/E5.
- Async rollout controls throughput only; it must preserve one policy version
  inside a collected update batch.

E3 and E4 intentionally branch from E1. This isolates scalar reward shaping
from turn-level credit instead of stacking every idea into one comparison.

## State isolation

Every official episode creates a fresh tau2 environment and database instance.
Concurrent episodes therefore do not mutate a shared live database. The locked
task manifest contains IDs only and never duplicates the benchmark oracle.

## LoRA closure

The project forces FSDP `use_orig_params=true` and installs an optimizer guard
through the configured VERL external library. Frozen base parameters are
filtered from optimizer parameter groups; in LoRA mode any non-`lora_`
trainable parameter aborts the run. VERL's colocated vLLM path removes the old
adapter, adds synchronized adapter tensors, and resets prefix cache on wake-up.
