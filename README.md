# Agent RL Qwen3

RL training project for long-horizon tool agents on tau2/tau3-bench.

## Stack

- Agent model: Qwen3-8B
- Training: verl
- Rollout inference: vLLM
- User simulator: DeepSeek V4 Flash API through LiteLLM
- Benchmark/environment: tau2/tau3-bench
- Algorithm ideas: GiGPO, HGPO, GRPO
                                      
## Layout

- `tau2-bench/`: upstream benchmark and simulator.
- `verl/`: upstream training framework, to be cloned later if we need source edits.
- `src/agent_rl/`: project-specific adapters, prompts, rollout logic, rewards, and trainer glue.
- `configs/`: experiment configuration.
- `scripts/`: setup, data, serving, training, and evaluation launchers.
- `docs/`: design notes and experiment plans.
- `experiments/`: local checkpoints and outputs.

