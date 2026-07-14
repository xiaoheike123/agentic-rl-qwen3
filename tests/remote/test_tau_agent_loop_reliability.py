import asyncio
from types import MethodType

import pytest

pytest.importorskip("verl.experimental.agent_loop.agent_loop")

import agent_rl.rollout.tau_agent_loop as loop_module
from agent_rl.envs.tau_env import TauInfrastructureError
from agent_rl.rollout.tau_agent_loop import (
    TauAgentLoop,
    TauAgentLoopSettings,
)
from agent_rl.trainer.tau_agent_loop_manager import TauAgentLoopWorker


@pytest.fixture(autouse=True)
def _reset_worker_semaphore(monkeypatch):
    monkeypatch.setattr(loop_module, "_worker_episode_semaphore", None)
    monkeypatch.setattr(loop_module, "_worker_episode_limit", None)


def _make_loop(operation, **settings):
    loop = object.__new__(TauAgentLoop)
    loop.settings = TauAgentLoopSettings(
        retry_backoff_seconds=0.0,
        **settings,
    )
    loop._run_episode = MethodType(operation, loop)
    return loop


def _failure(stage="reset"):
    return TauInfrastructureError(
        stage=stage,
        domain="airline",
        task_id="task-1",
    )


def test_episode_retries_from_a_fresh_attempt():
    attempts = 0
    expected = object()

    async def operation(self, sampling_params, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise _failure()
        return expected

    loop = _make_loop(operation, max_episode_attempts=3)
    result = asyncio.run(loop.run({}))

    assert result is expected
    assert attempts == 3


def test_episode_raises_after_retry_budget_is_exhausted():
    attempts = 0

    async def operation(self, sampling_params, **kwargs):
        nonlocal attempts
        attempts += 1
        raise _failure("step_termination")

    loop = _make_loop(operation, max_episode_attempts=3)

    with pytest.raises(TauInfrastructureError, match="step_termination"):
        asyncio.run(loop.run({}))

    assert attempts == 3


def test_worker_semaphore_limits_concurrent_episodes():
    active = 0
    maximum_active = 0

    async def operation(self, sampling_params, **kwargs):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return object()

    first = _make_loop(
        operation,
        max_episode_attempts=1,
        max_concurrent_episodes_per_worker=1,
    )
    second = _make_loop(
        operation,
        max_episode_attempts=1,
        max_concurrent_episodes_per_worker=1,
    )

    async def run_both():
        await asyncio.gather(first.run({}), second.run({}))

    asyncio.run(run_both())

    assert maximum_active == 1


def test_tau_worker_forwards_rollout_index(monkeypatch):
    captured = {}

    async def parent_run(
        self,
        sampling_params,
        trajectory,
        *,
        agent_name,
        trace=True,
        **kwargs,
    ):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        loop_module,
        "_worker_episode_semaphore",
        None,
    )
    monkeypatch.setattr(
        "verl.experimental.agent_loop.agent_loop.AgentLoopWorker._run_agent_loop",
        parent_run,
    )
    worker = object.__new__(TauAgentLoopWorker)
    asyncio.run(
        worker._run_agent_loop(
            {},
            {"rollout_n": 3, "step": 7, "validate": False},
            agent_name="tau_agent",
        )
    )
    assert captured["rollout_n"] == 3
    assert captured["trajectory_step"] == 7
    assert captured["trajectory_validate"] is False
