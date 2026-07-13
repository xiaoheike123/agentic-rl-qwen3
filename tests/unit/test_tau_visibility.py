from agent_rl.envs.tau_env import (
    EVALUATOR_TAU_INFO_KEYS,
    PUBLIC_TAU_INFO_KEYS,
    evaluator_tau_info,
    public_tau_info,
)


def test_public_tau_info_removes_private_task_fields() -> None:
    oracle_action = {
        "name": "cancel_reservation",
        "arguments": {"reservation_id": "secret-reservation"},
    }
    raw_info = {
        "task": {
            "initial_state": {"secret": True},
            "evaluation_criteria": {"actions": [oracle_action]},
        },
        "policy": "public domain policy",
        "tools": ["public tool"],
        "simulation_run": "{}",
        "reward_info": {"reward": 0.0},
        "future_upstream_private_field": oracle_action,
    }

    visible = public_tau_info(
        raw_info,
        domain="airline",
        task_id="synthetic-airline-1",
    )

    assert set(visible) == PUBLIC_TAU_INFO_KEYS | {"domain", "task_id"}
    assert "task" not in visible
    assert "future_upstream_private_field" not in visible
    assert visible["domain"] == "airline"
    assert visible["task_id"] == "synthetic-airline-1"

    evaluator = evaluator_tau_info(raw_info)
    assert set(evaluator) == EVALUATOR_TAU_INFO_KEYS
    assert "task" not in evaluator
    assert "policy" not in evaluator
    assert evaluator["reward_info"] == {"reward": 0.0}


def test_public_tau_info_does_not_mutate_upstream_info() -> None:
    raw_info = {
        "task": {"evaluation_criteria": {"actions": []}},
        "policy": "policy",
        "tools": [],
        "simulation_run": "{}",
    }

    public_tau_info(raw_info, domain="retail", task_id="synthetic-retail-1")

    assert "task" in raw_info
