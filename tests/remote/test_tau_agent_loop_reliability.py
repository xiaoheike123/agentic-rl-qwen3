import asyncio
from types import MethodType

import pytest

pytest.importorskip("verl.experimental.agent_loop.agent_loop")

import agent_rl.rollout.tau_agent_loop as loop_module
from agent_rl.data.schemas import (
    EpisodeRecord,
    RewardRecord,
    TokenTrace,
    TurnRecord,
)
from agent_rl.envs.tau_env import TauInfrastructureError
from agent_rl.rewards.process_reward import EnvironmentProcessReward
from agent_rl.rewards.reward_mixer import RewardMixer, RewardMixerConfig
from agent_rl.rollout.tau_agent_loop import (
    TauAgentLoop,
    TauAgentLoopSettings,
)
from agent_rl.trainer.tau_agent_loop_manager import TauAgentLoopWorker
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics


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


def test_invalid_policy_output_uses_e5_process_and_credit_pipeline():
    loop = object.__new__(TauAgentLoop)
    loop.settings = TauAgentLoopSettings(
        process_weight=0.1,
        enable_hindsight_credit=True,
    )
    loop.process_reward = EnvironmentProcessReward()
    loop.reward_mixer = RewardMixer(
        RewardMixerConfig(outcome_weight=1.0, process_weight=0.1)
    )
    loop.rollout_config = type("RolloutConfig", (), {"response_length": 8})()

    episode = EpisodeRecord(
        episode_id="episode",
        group_id="group",
        domain="airline",
        task_id="task",
        model="model",
    )
    episode.append_turn(
        TurnRecord(
            turn_index=0,
            observation="request",
            prompt_messages=[{"role": "user", "content": "request"}],
            action="<invalid_action>",
            next_observation="request",
            token_trace=TokenTrace(
                response_token_ids=[1, 2],
                response_loss_mask=[1, 1],
            ),
            terminated=True,
            info={
                "action_parse_valid": False,
                "action_parse_error": "invalid syntax",
            },
        )
    )
    episode.finish(
        reward=RewardRecord(outcome=0.0, total=0.0),
        success=False,
        termination_reason="invalid_action",
    )

    result = loop._build_policy_failure_output(
        episode=episode,
        prompt_ids=[10],
        response_ids=[1, 2],
        response_mask=[1, 1],
        response_logprobs=[0.0, 0.0],
        metrics=AgentLoopMetrics(),
        group_id="group",
        sample_index=0,
        episode_seed=42,
        domain="airline",
        task_id="task",
        database_source="training",
        context_rotations=0,
        attempted_turns=1,
        turn_token_spans=[(0, 0, 2)],
        invalid_action_error="invalid syntax",
        invalid_token_ids=[1, 2],
        invalid_decoded_output="<invalid_action>",
    )

    info = result.extra_fields["reward_extra_info"]
    assert result.reward_score == pytest.approx(-0.1)
    assert info["tau_process_reward"] == -1.0
    assert info["tau_invalid_action_count"] == 1
    assert result.extra_fields["tau_hindsight_evidence"] == [-1.0, -1.0]
