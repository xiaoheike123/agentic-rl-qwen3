# Project Design

This project trains Qwen3-8B as a long-horizon tool agent on tau2/tau3-bench.

High-level data flow:

```text
tau task id -> tau2 AgentGymEnv -> prompt/memory -> Qwen3 action -> tau2 step -> reward -> verl update
```

