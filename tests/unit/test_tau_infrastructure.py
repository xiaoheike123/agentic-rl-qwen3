import pytest

import agent_rl.envs.tau_env as tau_env_module
from agent_rl.envs.tau_env import (
    TauEnv,
    TauEnvConfig,
    TauInfrastructureError,
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
