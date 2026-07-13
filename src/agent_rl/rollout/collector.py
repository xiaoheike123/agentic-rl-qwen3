"""Synchronous, fixed-policy collection of complete rollout groups."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import monotonic
from typing import Sequence

from agent_rl.data.schemas import EpisodeRecord, EpisodeStatus
from agent_rl.envs.tau_env import TauEnvConfig
from agent_rl.rollout.episode_worker import EpisodeSpec, EpisodeWorker
from agent_rl.utils.jsonl import JsonlEpisodeStore


@dataclass(frozen=True, slots=True)
class SyncCollectorConfig:
    """Scheduling parameters for one fixed rollout batch."""

    group_size: int = 8
    max_concurrent_episodes: int = 8

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError("group_size must be at least two for GRPO")
        if self.max_concurrent_episodes <= 0:
            raise ValueError("max_concurrent_episodes must be greater than zero")


@dataclass(frozen=True, slots=True)
class RolloutGroupSpec:
    """One task whose policy response is sampled as a GRPO group."""

    group_id: str
    env_config: TauEnvConfig
    trial_id: int = 0
    seed: int | None = None
    policy_version: int | None = None
    attempt: int = 0

    def __post_init__(self) -> None:
        if not self.group_id.strip():
            raise ValueError("group_id must not be empty")
        if self.trial_id < 0:
            raise ValueError("trial_id must be non-negative")
        if self.attempt < 0:
            raise ValueError("attempt must be non-negative")

    @property
    def effective_group_id(self) -> str:
        return f"{self.group_id}::trial={self.trial_id}::attempt={self.attempt}"

    def episode_id(self, sample_index: int) -> str:
        return f"{self.effective_group_id}::sample={sample_index}"


@dataclass(frozen=True, slots=True)
class CollectedGroup:
    """All episode attempts associated with one GRPO group."""

    spec: RolloutGroupSpec
    episodes: tuple[EpisodeRecord, ...]
    expected_size: int

    def __post_init__(self) -> None:
        if self.expected_size < 2:
            raise ValueError("expected_size must be at least two for GRPO")

        if len(self.episodes) > self.expected_size:
            raise ValueError("a rollout group cannot exceed expected_size")

        sample_indices = [episode.sample_index for episode in self.episodes]
        if len(sample_indices) != len(set(sample_indices)):
            raise ValueError("a rollout group cannot contain duplicate sample indices")

        if len(self.completed) == self.expected_size:
            hashes = [
                episode.metadata.get("initial_db_hash")
                for episode in self.completed
            ]
            if not all(isinstance(value, str) and value for value in hashes):
                raise ValueError(
                    "every completed GRPO sample must record initial_db_hash"
                )
            if len(set(hashes)) != 1:
                raise ValueError(
                    "all samples in a GRPO group must start from the same DB state"
                )

    @property
    def completed(self) -> tuple[EpisodeRecord, ...]:
        return tuple(
            episode
            for episode in self.episodes
            if episode.status is EpisodeStatus.COMPLETED
        )

    @property
    def failed(self) -> tuple[EpisodeRecord, ...]:
        return tuple(
            episode
            for episode in self.episodes
            if episode.status is EpisodeStatus.FAILED
        )

    @property
    def ready_for_training(self) -> bool:
        return (
            len(self.episodes) == self.expected_size
            and len(self.completed) == self.expected_size
        )


@dataclass(frozen=True, slots=True)
class CollectionBatch:
    """One synchronous rollout batch returned after its barrier."""

    groups: tuple[CollectedGroup, ...]
    elapsed_seconds: float

    @property
    def episodes(self) -> tuple[EpisodeRecord, ...]:
        return tuple(episode for group in self.groups for episode in group.episodes)

    @property
    def training_groups(self) -> tuple[CollectedGroup, ...]:
        return tuple(group for group in self.groups if group.ready_for_training)

    @property
    def failed_episodes(self) -> tuple[EpisodeRecord, ...]:
        return tuple(episode for group in self.groups for episode in group.failed)


class SyncEpisodeCollector:
    """Run a fixed batch concurrently and wait for every episode."""

    def __init__(
        self,
        worker: EpisodeWorker,
        config: SyncCollectorConfig | None = None,
        store: JsonlEpisodeStore | None = None,
    ) -> None:
        self.worker = worker
        self.config = config or SyncCollectorConfig()
        self.store = store

    def collect(
        self,
        groups: Sequence[RolloutGroupSpec],
    ) -> CollectionBatch:
        if not groups:
            raise ValueError("groups must not be empty")

        self._validate_unique_groups(groups)
        started_at = monotonic()
        existing = self._load_existing_episodes()
        collected: list[list[EpisodeRecord]] = [[] for _ in groups]
        pending: list[tuple[int, EpisodeSpec]] = []

        for group_index, group in enumerate(groups):
            for sample_index in range(self.config.group_size):
                episode_spec = EpisodeSpec(
                    episode_id=group.episode_id(sample_index),
                    group_id=group.effective_group_id,
                    env_config=group.env_config,
                    sample_index=sample_index,
                    trial_id=group.trial_id,
                    seed=group.seed,
                    policy_version=group.policy_version,
                )
                previous = existing.get(episode_spec.episode_id)

                if previous is not None:
                    collected[group_index].append(previous)
                else:
                    pending.append((group_index, episode_spec))

        if pending:
            max_workers = min(
                self.config.max_concurrent_episodes,
                len(pending),
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.worker.run, episode_spec): group_index
                    for group_index, episode_spec in pending
                }

                for future in as_completed(futures):
                    group_index = futures[future]
                    episode = future.result()
                    collected[group_index].append(episode)

                    if self.store is not None:
                        self.store.append(episode)

        completed_groups = tuple(
            CollectedGroup(
                spec=group,
                episodes=tuple(
                    sorted(
                        collected[group_index],
                        key=lambda episode: episode.sample_index,
                    )
                ),
                expected_size=self.config.group_size,
            )
            for group_index, group in enumerate(groups)
        )

        return CollectionBatch(
            groups=completed_groups,
            elapsed_seconds=monotonic() - started_at,
        )

    def _load_existing_episodes(self) -> dict[str, EpisodeRecord]:
        if self.store is None:
            return {}

        return {episode.episode_id: episode for episode in self.store.iter_episodes()}

    @staticmethod
    def _validate_unique_groups(
        groups: Sequence[RolloutGroupSpec],
    ) -> None:
        identifiers = [group.effective_group_id for group in groups]

        if len(identifiers) != len(set(identifiers)):
            raise ValueError("groups must have unique effective_group_id values")

        policy_versions = {group.policy_version for group in groups}

        if len(policy_versions) > 1:
            raise ValueError(
                "all groups in one synchronous collection must use one "
                "frozen policy_version"
            )
