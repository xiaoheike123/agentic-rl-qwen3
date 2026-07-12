from agent_rl.algorithms.advantage_utils import compute_group_advantages
from agent_rl.data.schemas import EpisodeRecord, RewardRecord, TurnRecord


def _episode(episode_id: str, reward: float) -> EpisodeRecord:
    episode = EpisodeRecord(
        episode_id=episode_id,
        group_id="group",
        domain="mock",
        task_id="task",
        model="model",
    )
    episode.append_turn(
        TurnRecord(
            turn_index=0,
            observation="user: request",
            prompt_messages=[{"role": "user", "content": "request"}],
            action="done",
            next_observation="user: done",
            terminated=True,
        )
    )
    episode.finish(
        reward=RewardRecord(outcome=reward, total=reward),
        success=reward > 0,
        termination_reason="user_stop",
    )
    return episode


def test_group_advantages_are_centered():
    results = compute_group_advantages((_episode("a", 0.0), _episode("b", 1.0)))
    assert abs(sum(item.advantage for item in results)) < 1e-6
    assert results[0].advantage < 0 < results[1].advantage


def test_constant_reward_group_has_zero_advantage():
    results = compute_group_advantages((_episode("a", 1.0), _episode("b", 1.0)))
    assert [item.advantage for item in results] == [0.0, 0.0]
