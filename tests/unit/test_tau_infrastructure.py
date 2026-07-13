import pickle

import pytest

import agent_rl.envs.tau_env as tau_env_module
from agent_rl.envs.tau_env import (
    TauEnv,
    TauEnvConfig,
    TauInfrastructureError,
)


def test_infrastructure_error_round_trips_through_pickle() -> None:
    original = TauInfrastructureError(
        stage="reset",
        domain="airline",
        task_id="task-1",
    )

    restored = pickle.loads(pickle.dumps(original))

    assert isinstance(restored, TauInfrastructureError)
    assert restored.stage == "reset"
    assert restored.domain == "airline"
    assert restored.task_id == "task-1"
    assert str(restored) == str(original)


def test_env_configures_deepseek_nl_evaluator(monkeypatch) -> None:
    monkeypatch.setattr(
        tau_env_module,
        "_TransformableAgentGymEnv",
        _FakeAgentGymEnv,
    )

    TauEnv(
        TauEnvConfig(
            domain="airline",
            task_id="task-1",
            evaluator_llm="deepseek/deepseek-v4-pro",
            evaluator_llm_args={"temperature": 0.0},
        )
    )

    assert (
        tau_env_module.evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS
        == "deepseek/deepseek-v4-pro"
    )
    assert (
        tau_env_module.evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS
        == {"temperature": 0.0}
    )


class _FakeAgentGymEnv:
    reset_result = (
        "user: hello",
        {
            "policy": "policy",
            "tools": [],
            "simulation_run": "{}",
        },
    )
    step_result = (
        "user: next",
        0.0,
        False,
        False,
        {
            "policy": "policy",
            "tools": [],
            "simulation_run": "{}",
        },
    )

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def reset(self, seed=None):
        return self.reset_result

    def step(self, action):
        return self.step_result

    def get_db_hash(self):
        return "db-hash"


def _make_env(monkeypatch, *, reset_result=None, step_result=None):
    class FakeAgentGymEnv(_FakeAgentGymEnv):
        pass

    if reset_result is not None:
        FakeAgentGymEnv.reset_result = reset_result
    if step_result is not None:
        FakeAgentGymEnv.step_result = step_result
    monkeypatch.setattr(
        tau_env_module,
        "_TransformableAgentGymEnv",
        FakeAgentGymEnv,
    )
    return TauEnv(TauEnvConfig(domain="airline", task_id="task-1"))


def _info(simulation_run="{}"):
    return {
        "policy": "policy",
        "tools": [],
        "simulation_run": simulation_run,
    }


def test_reset_rejects_empty_initial_observation(monkeypatch):
    env = _make_env(monkeypatch, reset_result=("", _info()))

    with pytest.raises(TauInfrastructureError, match="during reset"):
        env.reset()


def test_step_rejects_empty_nonterminal_observation(monkeypatch):
    env = _make_env(
        monkeypatch,
        step_result=("", 0.0, False, False, _info()),
    )
    env.reset()

    with pytest.raises(TauInfrastructureError, match="step_observation"):
        env.step("hello")


def test_step_rejects_termination_without_simulation_run(monkeypatch):
    env = _make_env(
        monkeypatch,
        step_result=("user: final", 0.0, True, False, _info()),
    )
    env.reset()

    with pytest.raises(TauInfrastructureError, match="step_termination"):
        env.step("hello")


def test_step_accepts_empty_observation_after_valid_termination(monkeypatch):
    env = _make_env(
        monkeypatch,
        step_result=(
            "",
            1.0,
            True,
            False,
            _info('{"messages": []}'),
        ),
    )
    env.reset()

    transition = env.step("hello")

    assert transition.done is True
    assert transition.observation == ""
    assert transition.reward == 1.0


def test_step_wraps_evaluator_api_failure_for_episode_retry(monkeypatch):
    class FailingAgentGymEnv(_FakeAgentGymEnv):
        def step(self, action):
            raise RuntimeError("judge API unavailable")

    monkeypatch.setattr(
        tau_env_module,
        "_TransformableAgentGymEnv",
        FailingAgentGymEnv,
    )
    env = TauEnv(TauEnvConfig(domain="airline", task_id="task-1"))
    env.reset()

    with pytest.raises(TauInfrastructureError, match="during step") as caught:
        env.step("hello")

    assert caught.value.__cause__ is not None
    assert str(caught.value.__cause__) == "judge API unavailable"
