from agent_rl.credit.hindsight_credit import (
    HindsightCreditAssigner,
    compute_advantage_aligned_turn_weights,
)
from agent_rl.data.schemas import EpisodeRecord, RewardRecord, TurnRecord


def _episode() -> EpisodeRecord:
    episode = EpisodeRecord(
        episode_id="episode",
        group_id="group",
        domain="mock",
        task_id="task",
        model="model",
    )
    for index, process_reward in enumerate((-1.0, 0.5)):
        episode.append_turn(
            TurnRecord(
                turn_index=index,
                observation="request",
                prompt_messages=[{"role": "user", "content": "request"}],
                action="done",
                next_observation="next",
                process_reward=process_reward,
                terminated=index == 1,
            )
        )
    episode.finish(
        reward=RewardRecord(outcome=1.0, process=-0.5, total=0.85),
        success=True,
        termination_reason="user_stop",
    )
    return episode


def test_hindsight_extracts_signed_process_evidence():
    result = HindsightCreditAssigner().assign(_episode())
    assert result.turn_evidence == (-1.0, 0.5)


def test_positive_advantage_prefers_recovery_turn():
    weights = compute_advantage_aligned_turn_weights(
        (-1.0, 0.5),
        trajectory_advantage=1.0,
    )
    assert weights[1] > weights[0]
    assert abs(sum(weights) / 2 - 1.0) < 1e-9


def test_negative_advantage_prefers_error_turn():
    weights = compute_advantage_aligned_turn_weights(
        (-1.0, 0.5),
        trajectory_advantage=-1.0,
    )
    assert weights[0] > weights[1]
    assert abs(sum(weights) / 2 - 1.0) < 1e-9
