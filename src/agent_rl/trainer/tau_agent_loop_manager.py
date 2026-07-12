"""Agent-loop manager that preserves token-level tau credit signals."""

from __future__ import annotations

import ray
import torch

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopWorker,
)

from agent_rl.trainer.credit_encoding import encode_evidence_in_token_rewards


class TauAgentLoopWorker(AgentLoopWorker):
    """Replace scalar terminal rm_scores with sum-preserving credit scores."""

    def _postprocess(self, inputs, input_non_tensor_batch=None, validate=False):
        output = super()._postprocess(
            inputs,
            input_non_tensor_batch=input_non_tensor_batch,
            validate=validate,
        )
        response_mask = output.batch["response_mask"].to(torch.float32)
        token_scores = torch.zeros_like(response_mask)

        for row, item in enumerate(inputs):
            score = item.reward_score
            if score is None:
                continue

            raw_evidence = item.extra_fields.get("tau_hindsight_evidence") or []
            width = response_mask.shape[1]
            evidence = torch.zeros(
                width,
                dtype=torch.float32,
                device=response_mask.device,
            )
            count = min(len(raw_evidence), width)
            if count:
                evidence[:count] = torch.tensor(
                    raw_evidence[:count],
                    dtype=torch.float32,
                    device=response_mask.device,
                )
            token_scores[row] = encode_evidence_in_token_rewards(
                score=float(score),
                raw_evidence=evidence,
                response_mask=response_mask[row],
            )

        output.batch["rm_scores"] = token_scores
        return output


class TauAgentLoopManager(AgentLoopManager):
    """Use TauAgentLoopWorker on every rollout node."""

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(TauAgentLoopWorker)
        super().__init__(*args, **kwargs)
