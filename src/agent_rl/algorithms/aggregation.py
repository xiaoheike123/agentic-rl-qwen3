"""Loss-aggregation weights for long-horizon GRPO trajectories.

The project keeps verl's native token-mean PPO reduction and expresses each
research aggregation mode as a multiplicative token-advantage weight.  Because
the clipped PPO objective is linear in the advantage after the clipping branch
is selected, this is algebraically equivalent to changing the final reduction
while remaining invariant to micro-batch boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Sequence


class AggregationMode(str, Enum):
    SEQUENCE = "sequence"
    TOKEN = "token"
    BALANCED = "balanced"


@dataclass(frozen=True, slots=True)
class AggregationExample:
    """The aggregation-relevant metadata for one complete trajectory."""

    episode_id: str
    advantage: float
    active_tokens: int

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("episode_id must not be empty")
        if not isfinite(self.advantage):
            raise ValueError("advantage must be finite")
        if self.active_tokens <= 0:
            raise ValueError("active_tokens must be greater than zero")


@dataclass(frozen=True, slots=True)
class AggregationWeights:
    mode: AggregationMode
    episode_weights: tuple[float, ...]
    positive_episodes: int
    negative_episodes: int
    zero_episodes: int


def compute_aggregation_weights(
    examples: Sequence[AggregationExample],
    mode: AggregationMode | str,
) -> AggregationWeights:
    """Return per-token multipliers for a global token-mean reduction.

    Let ``T`` be the total active token count.  The returned multiplier ``w_i``
    is applied to every active token in trajectory ``i`` before verl computes
    ``sum(loss * mask) / T``.

    * token: every token is unchanged;
    * sequence: every trajectory contributes one trajectory-length-normalized
      term, then trajectories are averaged;
    * balanced: positive and negative subsets each use a token mean and are
      combined according to their trajectory counts.  Zero-advantage episodes
      have no gradient and receive weight zero.
    """

    resolved_mode = AggregationMode(mode)
    if not examples:
        raise ValueError("examples must not be empty")

    episode_ids = [example.episode_id for example in examples]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode_id values must be unique")

    total_tokens = sum(example.active_tokens for example in examples)
    positive = [example for example in examples if example.advantage > 0]
    negative = [example for example in examples if example.advantage < 0]
    zero = [example for example in examples if example.advantage == 0]

    if resolved_mode is AggregationMode.TOKEN:
        weights = tuple(1.0 for _ in examples)
    elif resolved_mode is AggregationMode.SEQUENCE:
        episode_count = len(examples)
        weights = tuple(
            total_tokens / (episode_count * example.active_tokens)
            for example in examples
        )
    else:
        nonzero_count = len(positive) + len(negative)
        if nonzero_count == 0:
            weights = tuple(0.0 for _ in examples)
        else:
            positive_tokens = sum(item.active_tokens for item in positive)
            negative_tokens = sum(item.active_tokens for item in negative)

            positive_weight = (
                total_tokens * len(positive) / (nonzero_count * positive_tokens)
                if positive_tokens
                else 0.0
            )
            negative_weight = (
                total_tokens * len(negative) / (nonzero_count * negative_tokens)
                if negative_tokens
                else 0.0
            )
            weights = tuple(
                positive_weight
                if example.advantage > 0
                else negative_weight
                if example.advantage < 0
                else 0.0
                for example in examples
            )

    return AggregationWeights(
        mode=resolved_mode,
        episode_weights=weights,
        positive_episodes=len(positive),
        negative_episodes=len(negative),
        zero_episodes=len(zero),
    )


def aggregate_scalar_losses(
    token_losses: Sequence[Sequence[float]],
    examples: Sequence[AggregationExample],
    mode: AggregationMode | str,
) -> float:
    """Reference implementation used for tests and experiment auditing."""

    if len(token_losses) != len(examples):
        raise ValueError("token_losses must align with examples")

    for losses, example in zip(token_losses, examples):
        if len(losses) != example.active_tokens:
            raise ValueError(
                f"episode {example.episode_id!r} has {len(losses)} losses but "
                f"declares {example.active_tokens} active tokens"
            )
        if any(not isfinite(value) for value in losses):
            raise ValueError("token losses must be finite")

    weights = compute_aggregation_weights(examples, mode).episode_weights
    total_tokens = sum(example.active_tokens for example in examples)
    weighted_sum = sum(
        weight * sum(losses) for weight, losses in zip(weights, token_losses)
    )
    return weighted_sum / total_tokens
