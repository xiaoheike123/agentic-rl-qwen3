"""Streaming episode scheduling under one frozen policy version."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from time import monotonic
from typing import Iterator, Sequence

from agent_rl.data.schemas import EpisodeRecord
from agent_rl.rollout.collector import (
    CollectedGroup,
    CollectionBatch,
    RolloutGroupSpec,
)
from agent_rl.rollout.episode_worker import EpisodeSpec, EpisodeWorker
from agent_rl.utils.jsonl import JsonlEpisodeStore


@dataclass(frozen=True, slots=True)
class AsyncCollectorConfig:
    """Worker-pool settings for streaming group completion."""

    group_size: int = 8
    max_concurrent_episodes: int = 8

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError("group_size must be at least two for GRPO")
        if self.max_concurrent_episodes <= 0:
            raise ValueError("max_concurrent_episodes must be greater than zero")


class AsyncEpisodeCollector:
    """Keep episode workers busy and emit each complete group immediately."""

    def __init__(
        self,
        worker: EpisodeWorker,
        config: AsyncCollectorConfig | None = None,
        store: JsonlEpisodeStore | None = None,
    ) -> None:
        self.worker = worker
        self.config = config or AsyncCollectorConfig()
        self.store = store

    def iter_ready_groups(
        self,
        groups: Sequence[RolloutGroupSpec],
    ) -> Iterator[CollectedGroup]:
        """Yield groups in completion order without updating the policy."""

        if not groups:
            raise ValueError("groups must not be empty")

        self._validate_groups(groups)
        existing = self._load_existing_episodes()
        collected: list[list[EpisodeRecord]] = [[] for _ in groups]
        pending_specs: list[tuple[int, EpisodeSpec]] = []

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
                    pending_specs.append((group_index, episode_spec))

        emitted: set[int] = set()

        for group_index in range(len(groups)):
            ready = self._build_ready_group(
                group_index,
                groups,
                collected,
            )
            if ready is not None:
                emitted.add(group_index)
                yield ready

        if not pending_specs:
            return

        pending_iter = iter(pending_specs)

        with ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_episodes
        ) as executor:
            active: dict[Future[EpisodeRecord], int] = {}

            def submit_next() -> bool:
                try:
                    group_index, episode_spec = next(pending_iter)
                except StopIteration:
                    return False

                future = executor.submit(self.worker.run, episode_spec)
                active[future] = group_index
                return True

            for _ in range(
                min(
                    self.config.max_concurrent_episodes,
                    len(pending_specs),
                )
            ):
                submit_next()

            while active:
                done, _ = wait(
                    active,
                    return_when=FIRST_COMPLETED,
                )
                ready_groups: list[CollectedGroup] = []

                for future in done:
                    group_index = active.pop(future)
                    episode = future.result()
                    collected[group_index].append(episode)

                    if self.store is not None:
                        self.store.append(episode)

                    if group_index not in emitted:
                        ready = self._build_ready_group(
                            group_index,
                            groups,
                            collected,
                        )
                        if ready is not None:
                            emitted.add(group_index)
                            ready_groups.append(ready)

                    submit_next()

                for ready in ready_groups:
                    yield ready

        if len(emitted) != len(groups):
            missing = [
                groups[index].effective_group_id
                for index in range(len(groups))
                if index not in emitted
            ]
            raise RuntimeError(
                f"collector ended before groups were complete: {missing}"
            )

    def collect(
        self,
        groups: Sequence[RolloutGroupSpec],
    ) -> CollectionBatch:
        """Collect all groups and return them in the requested order."""

        started_at = monotonic()
        by_id = {
            group.spec.effective_group_id: group
            for group in self.iter_ready_groups(groups)
        }
        ordered = tuple(by_id[group.effective_group_id] for group in groups)

        return CollectionBatch(
            groups=ordered,
            elapsed_seconds=monotonic() - started_at,
        )

    def _build_ready_group(
        self,
        group_index: int,
        groups: Sequence[RolloutGroupSpec],
        collected: list[list[EpisodeRecord]],
    ) -> CollectedGroup | None:
        episodes = collected[group_index]

        if len(episodes) < self.config.group_size:
            return None

        if len(episodes) > self.config.group_size:
            raise RuntimeError(
                f"group {groups[group_index].effective_group_id!r} "
                "contains too many episodes"
            )

        return CollectedGroup(
            spec=groups[group_index],
            episodes=tuple(
                sorted(
                    episodes,
                    key=lambda episode: episode.sample_index,
                )
            ),
            expected_size=self.config.group_size,
        )

    def _load_existing_episodes(self) -> dict[str, EpisodeRecord]:
        if self.store is None:
            return {}

        return {episode.episode_id: episode for episode in self.store.iter_episodes()}

    @staticmethod
    def _validate_groups(groups: Sequence[RolloutGroupSpec]) -> None:
        identifiers = [group.effective_group_id for group in groups]

        if len(identifiers) != len(set(identifiers)):
            raise ValueError("groups must have unique effective_group_id values")

        policy_versions = {group.policy_version for group in groups}

        if len(policy_versions) > 1:
            raise ValueError(
                "all groups in one async collection must use one frozen policy_version"
            )
