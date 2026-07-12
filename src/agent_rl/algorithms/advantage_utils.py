"""Group-relative trajectory advantages used by every GRPO experiment."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Sequence

from agent_rl.data.schemas import EpisodeRecord, EpisodeStatus


@dataclass(frozen=True, slots=True)
class GroupAdvantageConfig:
    epsilon: float = 1e-6
    normalize_by_std: bool = True
    minimum_group_size: int = 2

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.minimum_group_size < 2:
            raise ValueError("minimum_group_size must be at least two")


@dataclass(frozen=True, slots=True)
class EpisodeAdvantage:
    episode_id: str
    group_id: str
    reward: float
    group_mean: float
    group_std: float
    advantage: float


def episode_training_reward(episode: EpisodeRecord) -> float:
    """Return mixed reward when available, otherwise official outcome reward."""

    if episode.status is not EpisodeStatus.COMPLETED:
        raise ValueError(f"episode {episode.episode_id!r} is not completed")

    reward = (
        episode.reward.total
        if episode.reward.total is not None
        else episode.reward.outcome
    )
    if reward is None or not isfinite(reward):
        raise ValueError(
            f"episode {episode.episode_id!r} has no finite training reward"
        )
    return float(reward)


def compute_group_advantages(
    episodes: Sequence[EpisodeRecord],
    config: GroupAdvantageConfig | None = None,
) -> tuple[EpisodeAdvantage, ...]:
    """Standardize rewards independently inside each prompt/task group.

    Sample variance is used to match verl's native GRPO implementation.  A
    constant-reward group receives zero advantages rather than amplifying
    numerical noise through epsilon.
    """

    resolved = config or GroupAdvantageConfig()
    if not episodes:
        raise ValueError("episodes must not be empty")

    episode_ids = [episode.episode_id for episode in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode_id values must be unique")

    grouped: dict[str, list[tuple[EpisodeRecord, float]]] = {}
    for episode in episodes:
        grouped.setdefault(episode.group_id, []).append(
            (episode, episode_training_reward(episode))
        )

    by_episode_id: dict[str, EpisodeAdvantage] = {}
    for group_id, members in grouped.items():
        if len(members) < resolved.minimum_group_size:
            raise ValueError(
                f"group {group_id!r} has {len(members)} episodes; "
                f"at least {resolved.minimum_group_size} are required"
            )

        rewards = [reward for _, reward in members]
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
        std = sqrt(variance)

        for episode, reward in members:
            centered = reward - mean
            if not resolved.normalize_by_std:
                advantage = centered
            elif std <= resolved.epsilon:
                advantage = 0.0
            else:
                advantage = centered / (std + resolved.epsilon)

            by_episode_id[episode.episode_id] = EpisodeAdvantage(
                episode_id=episode.episode_id,
                group_id=group_id,
                reward=reward,
                group_mean=mean,
                group_std=std,
                advantage=advantage,
            )

    return tuple(by_episode_id[episode_id] for episode_id in episode_ids)
