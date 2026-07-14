from agent_rl.data.official_split import load_official_airline_split
from agent_rl.eval.official_airline import build_official_report
from agent_rl.eval.pass_hat_k import estimate_pass_hat_k


def test_locked_airline_split_has_expected_boundary() -> None:
    split = load_official_airline_split(validate_tau2=True)
    assert len(split.train_task_ids) == 30
    assert len(split.test_task_ids) == 20
    assert set(split.train_task_ids).isdisjoint(split.test_task_ids)


def test_pass_hat_k_is_not_pass_at_k() -> None:
    assert estimate_pass_hat_k(trials=4, successes=1, k=1) == 0.25
    assert estimate_pass_hat_k(trials=4, successes=1, k=4) == 0.0
    assert estimate_pass_hat_k(trials=4, successes=4, k=4) == 1.0


def test_official_report_requires_and_aggregates_20_by_4() -> None:
    split = load_official_airline_split(validate_tau2=False)
    trials = [
        {
            "task_id": task_id,
            "seed": seed,
            "success": seed == split.evaluation_seeds[0],
            "turns": 2,
            "prompt_tokens": 10,
            "response_tokens": 20,
            "tool_calls": 1,
            "tool_errors": 0,
            "invalid_actions": 0,
            "hit_max_turns": False,
            "termination_reason": "user_stop",
        }
        for task_id in split.test_task_ids
        for seed in split.evaluation_seeds
    ]
    report = build_official_report(trials)
    assert report["episodes"] == 80
    assert report["success_rate"] == 0.25
    assert report["pass^1"] == 0.25
    assert report["pass^2"] == 0.0
    assert report["pass^4"] == 0.0
