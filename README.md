# Agent RL Qwen3

RL training for robust long-horizon tool agents on tau2/tau3-bench.

## Stack

- Agent model: Qwen3-8B
- Training: verl
- Rollout inference: vLLM
- User simulator: DeepSeek V4 Flash API through LiteLLM
- Benchmark/environment: tau2/tau3-bench
- Optimization: GRPO with sequence, token, and balanced aggregation
- Credit: environment-verifiable process rewards and hindsight turn credit
- Evaluation: robustness under interaction perturbations
                                      
## Layout

- `tau2-bench/`: upstream benchmark and simulator.
- `verl/`: pinned upstream training framework submodule.
- `src/agent_rl/`: environment, rollout, reward, credit, training, and robustness code.
- `configs/`: experiment configuration.
- `scripts/`: setup, data, serving, training, and evaluation launchers.
- `docs/`: design notes and experiment plans.
- `experiments/`: local checkpoints and outputs.

## Research axes

```text
Algorithm:   E0 base -> E1 sequence -> E2 token -> E3 balanced
Credit:      E3 -> E4 process reward -> E5 hindsight turn credit
Systems:     S0 synchronous rollout vs S1 asynchronous rollout
Robustness:  clean vs paraphrase, information-order, and tool-failure tests
```
