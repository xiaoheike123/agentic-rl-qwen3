"""Sum-preserving transport of signed process evidence through verl."""

from __future__ import annotations

import torch


EVIDENCE_ENCODING_SCALE = 1e-3


def encode_evidence_in_token_rewards(
    *,
    score: float,
    raw_evidence: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Encode centered evidence without changing trajectory reward."""

    mask = response_mask.to(raw_evidence.dtype)
    active_count = mask.sum()
    if active_count <= 0:
        return torch.zeros_like(raw_evidence)

    evidence = raw_evidence * mask
    centered = (evidence - evidence.sum() / active_count) * mask
    encoded = float(score) / active_count * mask
    encoded = encoded + EVIDENCE_ENCODING_SCALE * centered

    first_active = torch.nonzero(response_mask, as_tuple=False)[0, 0]
    encoded[first_active] += float(score) - encoded.sum()
    return encoded


def decode_centered_evidence(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Recover zero-mean signed process evidence for every trajectory."""

    mask = response_mask.to(token_level_rewards.dtype)
    active_count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    score = (token_level_rewards * mask).sum(dim=-1, keepdim=True)
    baseline = score / active_count
    evidence = (token_level_rewards - baseline * mask) / EVIDENCE_ENCODING_SCALE
    evidence = evidence * mask
    evidence = evidence - evidence.sum(dim=-1, keepdim=True) / active_count * mask
    return evidence
