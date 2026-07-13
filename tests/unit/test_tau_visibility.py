from typing import Any

from tau2.gym.gym_agent import AgentGymEnv

from agent_rl.envs.tau_env import (
    EVALUATOR_TAU_INFO_KEYS,
    PUBLIC_TAU_INFO_KEYS,
    _TransformableAgentGymEnv,
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


def test_official_evaluation_uses_the_upstream_database(
    monkeypatch: Any,
) -> None:
    expected = object()
    environment = _TransformableAgentGymEnv.__new__(_TransformableAgentGymEnv)
    environment._database_override = None
    environment._environment_transform = None
    monkeypatch.setattr(
        AgentGymEnv,
        "_get_environment",
        lambda _self: expected,
    )

    assert environment._get_environment() is expected


def test_synthetic_training_injects_a_private_database_copy(
    monkeypatch: Any,
) -> None:
    provided = {"customers": ["synthetic-customer"]}
    received: dict[str, Any] = {}
    environment = _TransformableAgentGymEnv.__new__(_TransformableAgentGymEnv)
    environment._database_override = provided
    environment._environment_transform = None
    environment.domain = "telecom"
    environment.solo_mode = False

    def constructor(*, db: Any, solo_mode: bool) -> dict[str, Any]:
        received["db"] = db
        received["solo_mode"] = solo_mode
        return {"database": db}

    monkeypatch.setattr(
        "agent_rl.envs.tau_env.registry.get_env_constructor",
        lambda domain: constructor if domain == "telecom" else None,
    )

    result = environment._get_environment()

    assert result["database"] == provided
    assert received["db"] is not provided
    assert received["solo_mode"] is False
