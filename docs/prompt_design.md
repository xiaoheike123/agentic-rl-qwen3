# Prompt Design

The action prompt is fixed across E0-E5 so that algorithm comparisons do not
inherit prompt changes as a confounding variable.

Each policy request contains:

- The tau2 domain policy in the system message.
- The current observation, optionally shortened by deterministic context
  compression.
- OpenAI-compatible tool schemas passed through the native tool interface.
- A one-action constraint: either one user-facing message or one tool call.

Context compression is stateless and scoped to the current episode. It does
not maintain persistent memory, call another LLM, or carry information across
tasks. The raw observation and the exact compressed prompt are both retained
in the trajectory for auditing.

During verl training, oversized individual observations are clipped before
tokenization. If the accumulated response window can no longer fit another
action, the loop deterministically compresses the episode's observation
history into a fresh prompt and continues the same tau2 episode. Earlier
windows remain in the audit trajectory but only the final on-policy window is
used for that update (truncated backpropagation through time).

Each episode stores a SHA-256 hash of the system prompt and the compression
configuration so experiments can verify that all comparison groups used the
same prompt surface.
