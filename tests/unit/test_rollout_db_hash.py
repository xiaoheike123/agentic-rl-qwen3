from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent_rl.data.schemas import EpisodeRecord, EpisodeStatus
from agent_rl.rollout.collector import CollectedGroup, RolloutGroupSpec
from agent_rl.envs.tau_env import TauEnvConfig


def _completed_episode(sample_index: int, initial_db_hash: str | None) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=f"episode-{sample_index}",
        group_id="group",
        domain="airline",
        task_id="synthetic-airline-test",
        model="Qwen3-8B",
        sample_index=sample_index,
        status=EpisodeStatus.COMPLETED,
        success=True,
        termination_reason="user_stop",
        finished_at=datetime.now(timezone.utc).isoformat(),
        metadata={"initial_db_hash": initial_db_hash},
    )


def _group(episodes: tuple[EpisodeRecord, ...]) -> CollectedGroup:
    return CollectedGroup(
        spec=RolloutGroupSpec(
            group_id="group",
            env_config=TauEnvConfig(domain="airline", task_id="task"),
        ),
        episodes=episodes,
        expected_size=2,
    )


def test_group_accepts_identical_initial_db_hashes() -> None:
    group = _group(
        (
            _completed_episode(0, "same-hash"),
            _completed_episode(1, "same-hash"),
        )
    )

    assert group.ready_for_training


def test_group_rejects_different_initial_db_hashes() -> None:
    with pytest.raises(ValueError, match="same DB state"):
        _group(
            (
                _completed_episode(0, "hash-a"),
                _completed_episode(1, "hash-b"),
            )
        )


def test_group_rejects_missing_initial_db_hash() -> None:
    with pytest.raises(ValueError, match="record initial_db_hash"):
        _group(
            (
                _completed_episode(0, "same-hash"),
                _completed_episode(1, None),
            )
        )
