import pytest

from agent_rl.data.schemas import EpisodeRecord, ToolCallRecord, TurnRecord
from agent_rl.rewards.environment_checks import collect_tool_executions
from agent_rl.rollout.episode_worker import EpisodeDataError, _hydrate_tool_results


def _episode_with_call(*, name: str, is_control: bool) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id="episode-1",
        group_id="group-1",
        domain="airline",
        task_id="task-1",
        model="Qwen3-8B",
        turns=[
            TurnRecord(
                turn_index=0,
                observation="observation",
                prompt_messages=[{"role": "user", "content": "observation"}],
                action=name,
                next_observation="",
                tool_calls=[
                    ToolCallRecord(
                        call_id="call_9",
                        name=name,
                        is_control=is_control,
                    )
                ],
                terminated=True,
            )
        ],
    )


def test_done_control_call_is_acknowledged_by_agent_stop() -> None:
    episode = _episode_with_call(name="done", is_control=True)

    missing = _hydrate_tool_results(
        episode,
        {"termination_reason": "agent_stop", "messages": []},
        strict=True,
    )

    call = episode.turns[0].tool_calls[0]
    assert missing == []
    assert call.result_received is True
    assert call.result == {"termination_reason": "agent_stop"}
    assert collect_tool_executions(episode) == ()


def test_done_control_call_requires_agent_stop() -> None:
    episode = _episode_with_call(name="done", is_control=True)

    with pytest.raises(EpisodeDataError, match="call_9"):
        _hydrate_tool_results(
            episode,
            {"termination_reason": "max_steps", "messages": []},
            strict=True,
        )


def test_missing_environment_tool_result_remains_an_error() -> None:
    episode = _episode_with_call(name="search", is_control=False)

    with pytest.raises(EpisodeDataError, match="call_9"):
        _hydrate_tool_results(
            episode,
            {"termination_reason": "agent_stop", "messages": []},
            strict=True,
        )
