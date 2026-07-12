"""Adapt episode-level aggregation and credit to verl token advantages."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from agent_rl.algorithms.aggregation import (
    AggregationExample,
    AggregationMode,
    AggregationWeights,
    compute_aggregation_weights,
)
from agent_rl.credit.hindsight_credit import HindsightCreditResult
from agent_rl.credit.turn_advantage import (
    TurnAdvantageResult,
    project_turn_advantages,
)
from agent_rl.data.schemas import EpisodeRecord, EpisodeStatus


@dataclass(frozen=True, slots=True)
class PreparedEpisodeAdvantages:
    episode_id: str
    trajectory_advantage: float
    aggregation_weight: float
    active_token_count: int
    turn_advantages: tuple[float, ...]
    token_advantages: tuple[tuple[float, ...], ...]

    @property
    def active_tokens(self) -> int:
        return self.active_token_count


@dataclass(frozen=True, slots=True)
class PreparedAdvantageBatch:
    episodes: tuple[PreparedEpisodeAdvantages, ...]
    aggregation: AggregationWeights

    @property
    def flat_token_advantages(self) -> tuple[float, ...]:
        return tuple(
            value
            for episode in self.episodes
            for turn_values in episode.token_advantages
            for value in turn_values
        )


def prepare_advantage_batch(
    episodes: Sequence[EpisodeRecord],
    trajectory_advantages: Sequence[float],
    *,
    aggregation_mode: AggregationMode | str,
    credits: Sequence[HindsightCreditResult | None] | None = None,
) -> PreparedAdvantageBatch:
    """Project trajectory advantages to tokens and apply aggregation weights.

    The caller must pass the complete optimization batch.  Computing weights on
    individual micro-batches would make the objective depend on micro-batching.
    """

    if not episodes:
        raise ValueError("episodes must not be empty")
    if len(episodes) != len(trajectory_advantages):
        raise ValueError("trajectory_advantages must align with episodes")

    resolved_credits: Sequence[HindsightCreditResult | None]
    if credits is None:
        resolved_credits = (None,) * len(episodes)
    else:
        if len(credits) != len(episodes):
            raise ValueError("credits must align with episodes")
        resolved_credits = credits

    projected: list[TurnAdvantageResult] = []
    examples: list[AggregationExample] = []

    for episode, advantage, credit in zip(
        episodes,
        trajectory_advantages,
        resolved_credits,
    ):
        if episode.status is not EpisodeStatus.COMPLETED:
            raise ValueError(f"episode {episode.episode_id!r} must be completed")
        if not isfinite(advantage):
            raise ValueError("trajectory advantages must be finite")

        projection = project_turn_advantages(
            episode,
            trajectory_advantage=advantage,
            credit=credit,
            require_token_traces=True,
        )
        active_tokens = sum(
            sum(turn.token_trace.response_loss_mask)
            if turn.token_trace.response_loss_mask
            else len(turn.token_trace.response_token_ids)
            for turn in episode.turns
        )
        if active_tokens <= 0:
            raise ValueError(
                f"episode {episode.episode_id!r} has no active response tokens"
            )

        projected.append(projection)
        examples.append(
            AggregationExample(
                episode_id=episode.episode_id,
                advantage=advantage,
                active_tokens=active_tokens,
            )
        )

    aggregation = compute_aggregation_weights(examples, aggregation_mode)
    prepared: list[PreparedEpisodeAdvantages] = []

    for episode, advantage, projection, weight in zip(
        episodes,
        trajectory_advantages,
        projected,
        aggregation.episode_weights,
    ):
        prepared.append(
            PreparedEpisodeAdvantages(
                episode_id=episode.episode_id,
                trajectory_advantage=advantage,
                aggregation_weight=weight,
                active_token_count=examples[len(prepared)].active_tokens,
                turn_advantages=tuple(
                    value * weight for value in projection.turn_advantages
                ),
                token_advantages=tuple(
                    tuple(value * weight for value in turn_values)
                    for turn_values in projection.token_advantages
                ),
            )
        )

    return PreparedAdvantageBatch(
        episodes=tuple(prepared),
        aggregation=aggregation,
    )
