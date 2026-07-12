"""Advantage-aligned hindsight credit from verified process evidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from agent_rl.data.schemas import EpisodeRecord, EpisodeStatus


@dataclass(frozen=True, slots=True)
class HindsightCreditConfig:
    process_alignment_scale: float = 1.0
    minimum_weight: float = 0.05
    maximum_weight: float = 3.0

    def __post_init__(self) -> None:
        if self.process_alignment_scale < 0:
            raise ValueError("process_alignment_scale must be non-negative")
        if self.minimum_weight <= 0:
            raise ValueError("minimum_weight must be positive")
        if self.maximum_weight < self.minimum_weight:
            raise ValueError("maximum_weight must not be less than minimum_weight")


@dataclass(frozen=True, slots=True)
class HindsightCreditResult:
    """Signed process evidence; final weights require the GRPO advantage."""

    turn_evidence: tuple[float, ...]


class HindsightCreditAssigner:
    """Extract post-episode turn evidence without guessing update direction."""

    def __init__(
        self,
        config: HindsightCreditConfig | None = None,
    ) -> None:
        self.config = config or HindsightCreditConfig()

    def assign(self, episode: EpisodeRecord) -> HindsightCreditResult:
        if episode.status is not EpisodeStatus.COMPLETED:
            raise ValueError("hindsight credit requires a completed episode")
        if not episode.turns:
            raise ValueError("hindsight credit requires at least one turn")

        evidence = tuple(float(turn.process_reward or 0.0) for turn in episode.turns)
        if any(not isfinite(value) for value in evidence):
            raise ValueError("turn process evidence must be finite")
        return HindsightCreditResult(turn_evidence=evidence)


def compute_advantage_aligned_turn_weights(
    evidence: Sequence[float],
    *,
    trajectory_advantage: float,
    config: HindsightCreditConfig | None = None,
) -> tuple[float, ...]:
    """Reference implementation of the token-side E5 weighting rule."""

    if not evidence:
        raise ValueError("evidence must not be empty")
    if not isfinite(trajectory_advantage):
        raise ValueError("trajectory_advantage must be finite")
    if any(not isfinite(value) for value in evidence):
        raise ValueError("evidence must be finite")

    resolved = config or HindsightCreditConfig()
    mean_evidence = sum(evidence) / len(evidence)
    direction = 1.0 if trajectory_advantage > 0 else -1.0 if trajectory_advantage < 0 else 0.0
    raw = [
        min(
            resolved.maximum_weight,
            max(
                resolved.minimum_weight,
                1.0
                + direction
                * resolved.process_alignment_scale
                * (value - mean_evidence),
            ),
        )
        for value in evidence
    ]
    mean_weight = sum(raw) / len(raw)
    return tuple(value / mean_weight for value in raw)
