from agent_rl.data.schemas import EpisodeRecord, RewardRecord, TurnRecord
from agent_rl.robustness.metrics import compute_robustness_metrics


def _episode(episode_id: str, success: bool, turns: int, perturbation: str):
    episode = EpisodeRecord(
        episode_id=episode_id,
        group_id="group",
        domain="mock",
        task_id="task",
        model="model",
        metadata={"perturbation": perturbation},
    )
    for index in range(turns):
        episode.append_turn(
            TurnRecord(
                turn_index=index,
                observation="request",
                prompt_messages=[{"role": "user", "content": "request"}],
                action="done",
                next_observation="next",
                terminated=index == turns - 1,
            )
        )
    episode.finish(
        reward=RewardRecord(outcome=float(success), total=float(success)),
        success=success,
        termination_reason="user_stop",
    )
    return episode


def test_paired_success_drop_and_extra_steps():
    clean = _episode("clean", True, 1, "clean")
    perturbed = _episode("perturbed", False, 3, "paraphrase")
    metrics = compute_robustness_metrics((clean,), (perturbed,))
    assert metrics.success_drop == 1.0
    assert metrics.extra_steps_after_perturbation == 2
