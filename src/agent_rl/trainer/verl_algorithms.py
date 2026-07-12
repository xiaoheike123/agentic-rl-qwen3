"""Custom advantage estimators registered into verl at process import."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est

from agent_rl.trainer.credit_encoding import decode_centered_evidence


def _group_relative_scalars(
    token_level_rewards: torch.Tensor,
    index: np.ndarray,
    *,
    epsilon: float,
    normalize_by_std: bool,
) -> torch.Tensor:
    scores = token_level_rewards.sum(dim=-1)
    groups: dict[Any, list[int]] = defaultdict(list)
    for row, group_id in enumerate(index):
        groups[group_id].append(row)

    scalars = torch.zeros_like(scores)
    for group_id, rows in groups.items():
        if len(rows) < 2:
            raise ValueError(f"GRPO group {group_id!r} contains only one trajectory")
        row_index = torch.tensor(rows, device=scores.device)
        values = scores[row_index]
        centered = values - values.mean()
        if normalize_by_std:
            std = values.std(unbiased=True)
            if std <= epsilon:
                centered = torch.zeros_like(centered)
            else:
                centered = centered / (std + epsilon)
        scalars[row_index] = centered
    return scalars


def _balanced_row_weights(
    scalars: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    token_counts = response_mask.sum(dim=-1).to(torch.float32)
    if torch.any(token_counts <= 0):
        raise ValueError("every trajectory must contain generated tokens")

    total_tokens = token_counts.sum()
    positive = scalars > 0
    negative = scalars < 0
    nonzero_count = positive.sum() + negative.sum()
    weights = torch.zeros_like(scalars)
    if nonzero_count == 0:
        return weights

    if positive.any():
        weights[positive] = (
            total_tokens
            * positive.sum()
            / (nonzero_count * token_counts[positive].sum())
        )
    if negative.any():
        weights[negative] = (
            total_tokens
            * negative.sum()
            / (nonzero_count * token_counts[negative].sum())
        )
    return weights


def _hindsight_token_weights(
    scalars: torch.Tensor,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    process_alignment_scale: float,
    minimum_weight: float,
    maximum_weight: float,
) -> torch.Tensor:
    if process_alignment_scale < 0:
        raise ValueError("process_alignment_scale must be non-negative")
    if minimum_weight <= 0:
        raise ValueError("minimum_weight must be positive")
    if maximum_weight < minimum_weight:
        raise ValueError("maximum_weight must not be less than minimum_weight")

    evidence = decode_centered_evidence(
        token_level_rewards,
        response_mask,
    )
    direction = torch.sign(scalars).unsqueeze(-1)
    weights = 1.0 + direction * process_alignment_scale * evidence
    weights = weights.clamp(min=minimum_weight, max=maximum_weight) * response_mask
    active_count = response_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    weight_sum = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return weights * active_count / weight_sum


def _tau_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    *,
    balanced: bool,
    hindsight: bool,
    norm_adv_by_std_in_grpo: bool = True,
    epsilon: float = 1e-6,
    process_alignment_scale: float = 1.0,
    minimum_weight: float = 0.05,
    maximum_weight: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        scalars = _group_relative_scalars(
            token_level_rewards,
            index,
            epsilon=epsilon,
            normalize_by_std=norm_adv_by_std_in_grpo,
        )
        row_weights = (
            _balanced_row_weights(scalars, response_mask)
            if balanced
            else torch.ones_like(scalars)
        )
        token_weights = (
            _hindsight_token_weights(
                scalars,
                token_level_rewards,
                response_mask,
                process_alignment_scale=process_alignment_scale,
                minimum_weight=minimum_weight,
                maximum_weight=maximum_weight,
            )
            if hindsight
            else response_mask.to(token_level_rewards.dtype)
        )
        advantages = (
            scalars.unsqueeze(-1)
            * row_weights.unsqueeze(-1)
            * token_weights
            * response_mask
        )
    return advantages, advantages


@register_adv_est("tau_balanced_grpo")
def compute_tau_balanced_grpo(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if config is not None:
        norm_adv_by_std_in_grpo = bool(
            config.get("norm_adv_by_std_in_grpo", norm_adv_by_std_in_grpo)
        )
    return _tau_grpo_advantage(
        token_level_rewards,
        response_mask,
        index,
        balanced=True,
        hindsight=False,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )


@register_adv_est("tau_hindsight_balanced_grpo")
def compute_tau_hindsight_balanced_grpo(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    process_alignment_scale = 1.0
    minimum_weight = 0.05
    maximum_weight = 3.0
    if config is not None:
        norm_adv_by_std_in_grpo = bool(
            config.get("norm_adv_by_std_in_grpo", norm_adv_by_std_in_grpo)
        )
        process_alignment_scale = float(
            config.get("hindsight_process_alignment_scale", 1.0)
        )
        minimum_weight = float(config.get("hindsight_minimum_weight", 0.05))
        maximum_weight = float(config.get("hindsight_maximum_weight", 3.0))
    return _tau_grpo_advantage(
        token_level_rewards,
        response_mask,
        index,
        balanced=True,
        hindsight=True,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        process_alignment_scale=process_alignment_scale,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
    )
