"""Validated access to official tau2 outcome rewards."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from agent_rl.data.schemas import EpisodeRecord, EpisodeStatus


@dataclass(frozen=True, slots=True)
class OutcomeRewardResult:
    """Official tau2 score and its evaluator breakdown."""

    score: float
    components: dict[str, float]


def get_outcome_reward(episode: EpisodeRecord) -> OutcomeRewardResult:
    if episode.status is not EpisodeStatus.COMPLETED:
        raise ValueError("outcome reward requires a completed episode")

    score = episode.reward.outcome

    if score is None:
        raise ValueError("completed episode has no outcome reward")

    if not isfinite(score):
        raise ValueError("outcome reward must be finite")

    return OutcomeRewardResult(
        score=float(score),
        components=dict(episode.reward.components),
    )
