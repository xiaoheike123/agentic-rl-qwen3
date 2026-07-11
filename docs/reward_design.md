# Reward and Credit Design

Reward generation and credit assignment are separate:

- Outcome reward uses the official tau2 evaluator.
- Process reward uses deterministic environment evidence such as tool results,
  state transitions, policy preconditions, and explicit confirmation.
- Hindsight credit assigns completed-trajectory evidence to earlier turns.

Reward code must not depend on an LLM judge for the primary training signal.
