# Reward and Credit Design

Reward generation and credit assignment are separate:

- Outcome reward uses the official tau2 evaluator.
- Process reward uses deterministic environment evidence: authoritative tool
results, tool errors, repeated identical calls, and recovery after a tool
error.
- Hindsight credit assigns completed-trajectory evidence to earlier turns.

The current hindsight rule is an advantage-aligned process-evidence heuristic,
not a counterfactual causal estimator. Signed turn evidence is collected only
after the environment has returned authoritative tool results. After GRPO
computes each trajectory's group-relative advantage, positive trajectories
emphasize good process events while negative trajectories emphasize errors.
Weights are normalized to mean one and the evidence is transported through
verl as a sum-preserving token-reward residual, including for zero rewards.

Reward code must not depend on an LLM judge for the primary training signal.
Natural-language policy compliance and confirmation are left to tau2's
official outcome evaluator unless a deterministic verifier is added later.
An ordinary successful tool call receives no positive process reward; this
prevents reward hacking through unnecessary but executable calls. Recovery is
credited only when a later successful call matches the failed tool and exact
arguments.

Invalid actions are detected with tau2's own action parser and penalized at the
responsible turn. Reaching tau2's maximum-step termination or a local
response-length truncation receives one penalty on the final retained turn.
Ordinary tool success remains neutral.

Repeated identical calls are logged for diagnosis but do not affect training
reward. A valid no-progress penalty requires result equality and proof that no
intervening action changed environment state; call fingerprints alone are not
sufficient and would incorrectly punish legitimate post-update verification.
