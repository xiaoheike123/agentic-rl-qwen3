"""Combine official outcome and verifiable process rewards."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from agent_rl.data.schemas import EpisodeRecord
from agent_rl.rewards.outcome_reward import get_outcome_reward
from agent_rl.rewards.process_reward import ProcessRewardResult


@dataclass(frozen=True, slots=True)
class RewardMixerConfig:
    outcome_weight: float = 1.0
    process_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.outcome_weight < 0:
            raise ValueError("outcome_weight must be non-negative")
        if self.process_weight < 0:
            raise ValueError("process_weight must be non-negative")
        if self.outcome_weight == 0 and self.process_weight == 0:
            raise ValueError("at least one reward weight must be positive")


@dataclass(frozen=True, slots=True)
class MixedRewardResult:
    outcome: float
    process: float
    total: float


class RewardMixer:
    def __init__(
        self,
        config: RewardMixerConfig | None = None,
    ) -> None:
        self.config = config or RewardMixerConfig()

    def mix(
        self,
        episode: EpisodeRecord,
        process_result: ProcessRewardResult | None = None,
    ) -> MixedRewardResult:
        outcome_result = get_outcome_reward(episode)
        process_score = process_result.total if process_result is not None else 0.0
        total = (
            self.config.outcome_weight * outcome_result.score
            + self.config.process_weight * process_score
        )

        episode.reward.process = process_score if process_result is not None else None
        episode.reward.total = total
        episode.reward.components["mixed/outcome"] = outcome_result.score
        episode.reward.components["mixed/process"] = process_score
        episode.reward.components["mixed/total"] = total

        if process_result is not None:
            episode.metadata["process_reward"] = {
                "total": process_result.total,
                "turn_scores": list(process_result.turn_scores),
                "normalization_count": process_result.normalization_count,
                "checks": [asdict(check) for check in process_result.checks],
            }

        return MixedRewardResult(
            outcome=outcome_result.score,
            process=process_score,
            total=total,
        )
