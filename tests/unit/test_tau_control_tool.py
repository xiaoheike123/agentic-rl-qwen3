import pytest

from agent_rl.data.schemas import EpisodeRecord, ToolCallRecord, TurnRecord
from agent_rl.rewards.environment_checks import collect_tool_executions
from agent_rl.rewards.process_reward import EnvironmentProcessReward
from agent_rl.rollout.episode_worker import (
    EpisodeDataError,
    _collect_tool_results,
    _hydrate_tool_results,
)


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

    with pytest.raises(
        EpisodeDataError,
        match="termination_reason='agent_stop'",
    ):
        _hydrate_tool_results(
            episode,
            {"termination_reason": "agent_stop", "messages": []},
            strict=True,
        )


def test_agent_error_preserves_final_recorded_unexecuted_tool_call() -> None:
    episode = _episode_with_call(name="create_task", is_control=False)
    episode.turns[0].tool_calls[0].call_id = "call_31"

    missing = _hydrate_tool_results(
        episode,
        {
            "termination_reason": "agent_error",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_31",
                            "name": "create_task",
                            "arguments": {
                                "title": "Important Meeting",
                                "user_id": "user_1",
                            },
                        }
                    ],
                }
            ],
        },
        strict=True,
    )

    turn = episode.turns[0]
    call = turn.tool_calls[0]

    assert missing == ["call_31"]
    assert call.result_received is False
    assert call.result is None
    assert call.error is None
    assert turn.truncated is False
    assert turn.info["terminal_unexecuted_tool_call_ids"] == ["call_31"]
    assert turn.info["terminal_unexecuted_reason"] == "agent_error"

    executions = collect_tool_executions(episode)
    assert len(executions) == 1
    assert executions[0].call_id == "call_31"
    assert executions[0].result_received is False

    process = EnvironmentProcessReward().evaluate(episode)
    missing_checks = [
        check for check in process.checks
        if check.name == "tool_result_received"
    ]
    assert process.total < 0.0
    assert len(missing_checks) == 1
    assert missing_checks[0].passed is False
    assert missing_checks[0].evidence["call_id"] == "call_31"


def test_agent_error_requires_request_to_exist_in_tau_history() -> None:
    episode = _episode_with_call(name="create_task", is_control=False)
    episode.turns[0].tool_calls[0].call_id = "call_31"

    with pytest.raises(
        EpisodeDataError,
        match=r"assistant_request_ids=\[\]",
    ):
        _hydrate_tool_results(
            episode,
            {"termination_reason": "agent_error", "messages": []},
            strict=True,
        )


def test_user_error_does_not_allow_missing_assistant_tool_result() -> None:
    episode = _episode_with_call(name="create_task", is_control=False)
    episode.turns[0].tool_calls[0].call_id = "call_31"

    with pytest.raises(
        EpisodeDataError,
        match="termination_reason='user_error'",
    ):
        _hydrate_tool_results(
            episode,
            {
                "termination_reason": "user_error",
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_31",
                                "name": "create_task",
                                "arguments": {},
                            }
                        ],
                    }
                ],
            },
            strict=True,
        )


def test_agent_error_does_not_allow_missing_result_from_earlier_turn() -> None:
    episode = EpisodeRecord(
        episode_id="episode-1",
        group_id="group-1",
        domain="airline",
        task_id="task-1",
        model="Qwen3-8B",
        turns=[
            TurnRecord(
                turn_index=0,
                observation="first observation",
                prompt_messages=[
                    {"role": "user", "content": "first observation"}
                ],
                action="first_search",
                next_observation="second observation",
                tool_calls=[
                    ToolCallRecord(
                        call_id="call_1",
                        name="first_search",
                        is_control=False,
                    )
                ],
            ),
            TurnRecord(
                turn_index=1,
                observation="second observation",
                prompt_messages=[
                    {"role": "user", "content": "second observation"}
                ],
                action="final answer",
                next_observation="",
                terminated=True,
            ),
        ],
    )

    with pytest.raises(EpisodeDataError, match="call_1"):
        _hydrate_tool_results(
            episode,
            {
                "termination_reason": "agent_error",
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "name": "first_search",
                                "arguments": {},
                            }
                        ],
                    }
                ],
            },
            strict=True,
        )


def test_max_steps_allows_missing_result_only_on_final_environment_turn() -> None:
    episode = _episode_with_call(name="search", is_control=False)

    missing = _hydrate_tool_results(
        episode,
        {"termination_reason": "max_steps", "messages": []},
        strict=True,
    )

    turn = episode.turns[0]
    call = turn.tool_calls[0]

    assert missing == ["call_9"]
    assert turn.truncated is True
    assert turn.info["max_steps_truncated_tool_call_ids"] == ["call_9"]
    assert call.result_received is False
    assert call.result is None
    assert call.error is None


def test_max_steps_rejects_submitted_call_without_result() -> None:
    episode = _episode_with_call(name="search", is_control=False)

    with pytest.raises(EpisodeDataError, match="call_9"):
        _hydrate_tool_results(
            episode,
            {
                "termination_reason": "max_steps",
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_9",
                                "name": "search",
                                "arguments": {},
                            }
                        ],
                    }
                ],
            },
            strict=True,
        )


def test_max_steps_does_not_allow_missing_result_from_earlier_turn() -> None:
    episode = EpisodeRecord(
        episode_id="episode-1",
        group_id="group-1",
        domain="airline",
        task_id="task-1",
        model="Qwen3-8B",
        turns=[
            TurnRecord(
                turn_index=0,
                observation="first observation",
                prompt_messages=[
                    {"role": "user", "content": "first observation"}
                ],
                action="first_search",
                next_observation="second observation",
                tool_calls=[
                    ToolCallRecord(
                        call_id="call_1",
                        name="first_search",
                        is_control=False,
                    )
                ],
            ),
            TurnRecord(
                turn_index=1,
                observation="second observation",
                prompt_messages=[
                    {"role": "user", "content": "second observation"}
                ],
                action="second_search",
                next_observation="",
                tool_calls=[
                    ToolCallRecord(
                        call_id="call_2",
                        name="second_search",
                        is_control=False,
                    )
                ],
                terminated=True,
            ),
        ],
    )

    with pytest.raises(EpisodeDataError, match="call_1"):
        _hydrate_tool_results(
            episode,
            {
                "termination_reason": "max_steps",
                "messages": [
                    {
                        "role": "tool",
                        "requestor": "assistant",
                        "id": "call_2",
                        "content": {"status": "ok"},
                        "error": False,
                    }
                ],
            },
            strict=True,
        )


def test_max_steps_preserves_uncommitted_tool_call_at_recorded_tail() -> None:
    episode = EpisodeRecord(
        episode_id="episode-1",
        group_id="group-1",
        domain="airline",
        task_id="task-1",
        model="Qwen3-8B",
        turns=[
            TurnRecord(
                turn_index=0,
                observation="first observation",
                prompt_messages=[
                    {"role": "user", "content": "first observation"}
                ],
                action="resolved_search",
                next_observation="second observation",
                tool_calls=[
                    ToolCallRecord(
                        call_id="call_30",
                        name="resolved_search",
                        is_control=False,
                    )
                ],
            ),
            TurnRecord(
                turn_index=1,
                observation="second observation",
                prompt_messages=[
                    {"role": "user", "content": "second observation"}
                ],
                action="boundary_search",
                next_observation="terminal observation",
                tool_calls=[
                    ToolCallRecord(
                        call_id="call_31",
                        name="boundary_search",
                        is_control=False,
                    )
                ],
            ),
            TurnRecord(
                turn_index=2,
                observation="terminal observation",
                prompt_messages=[
                    {"role": "user", "content": "terminal observation"}
                ],
                action="terminal text",
                next_observation="",
                truncated=True,
            ),
        ],
    )

    missing = _hydrate_tool_results(
        episode,
        {
            "termination_reason": "max_steps",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_30",
                            "name": "resolved_search",
                            "arguments": {},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "requestor": "assistant",
                    "id": "call_30",
                    "content": {"status": "ok"},
                    "error": False,
                },
            ],
        },
        strict=True,
    )

    resolved_call = episode.turns[0].tool_calls[0]
    boundary_turn = episode.turns[1]
    boundary_call = boundary_turn.tool_calls[0]

    assert missing == ["call_31"]
    assert resolved_call.result_received is True
    assert boundary_call.result_received is False
    assert boundary_turn.truncated is False
    assert boundary_turn.info["max_steps_truncated_tool_call_ids"] == [
        "call_31"
    ]
    assert boundary_turn.info["tool_call_commit_status"] == "uncommitted"


def test_call_31_with_matching_tau_result_is_hydrated_normally() -> None:
    episode = _episode_with_call(name="create_task", is_control=False)
    episode.turns[0].tool_calls[0].call_id = "call_31"

    missing = _hydrate_tool_results(
        episode,
        {
            "termination_reason": "max_steps",
            "messages": [
                {
                    "role": "tool",
                    "requestor": "assistant",
                    "id": "call_31",
                    "content": {
                        "task_id": "task_2",
                        "title": "Important Meeting",
                    },
                    "error": False,
                }
            ],
        },
        strict=True,
    )

    call = episode.turns[0].tool_calls[0]

    assert missing == []
    assert call.result_received is True
    assert call.result == {
        "task_id": "task_2",
        "title": "Important Meeting",
    }
    assert call.error is None
    assert episode.turns[0].truncated is False

def test_tau_multi_tool_message_is_flattened_and_user_results_are_ignored() -> None:
    results = _collect_tool_results(
        {
            "messages": [
                {
                    "role": "tool",
                    "tool_messages": [
                        {
                            "role": "tool",
                            "requestor": "assistant",
                            "id": "call_15",
                            "content": {"status": "ok"},
                            "error": False,
                        },
                        {
                            "role": "tool",
                            "requestor": "user",
                            "id": "user_call_3",
                            "content": {"status": "private"},
                            "error": False,
                        },
                    ],
                },
            ]
        }
    )

    assert results == {
        "call_15": {
            "result": {"status": "ok"},
            "error": None,
        }
    }


def test_assistant_tool_message_without_call_id_is_ignored() -> None:
    results = _collect_tool_results(
        {
            "messages": [
                {
                    "role": "tool",
                    "tool_messages": [
                        {
                            "role": "tool",
                            "requestor": "assistant",
                            "content": {"status": "orphan"},
                            "error": False,
                        },
                        {
                            "role": "tool",
                            "requestor": "assistant",
                            "id": "call_15",
                            "content": {"status": "ok"},
                            "error": False,
                        },
                    ],
                }
            ]
        }
    )

    assert results == {
        "call_15": {
            "result": {"status": "ok"},
            "error": None,
        }
    }


def test_identical_duplicate_tool_results_are_deduplicated() -> None:
    message = {
        "role": "tool",
        "requestor": "assistant",
        "id": "call_15",
        "content": {"status": "ok"},
        "error": False,
    }

    results = _collect_tool_results(
        {"messages": [message, dict(message)]}
    )

    assert results == {
        "call_15": {
            "result": {"status": "ok"},
            "error": None,
        }
    }


def test_conflicting_duplicate_tool_results_keep_first_success() -> None:
    simulation_run = {
        "messages": [
            {
                "role": "tool",
                "requestor": "assistant",
                "id": "call_15",
                "content": {"status": "ok"},
                "error": False,
            },
            {
                "role": "tool",
                "requestor": "assistant",
                "id": "call_15",
                "content": {"status": "different"},
                "error": False,
            },
        ]
    }

    assert _collect_tool_results(simulation_run) == {
        "call_15": {
            "result": {"status": "ok"},
            "error": None,
        }
    }


def test_duplicate_tool_result_prefers_success_over_error() -> None:
    simulation_run = {
        "messages": [
            {
                "role": "tool",
                "requestor": "assistant",
                "id": "call_15",
                "content": "temporary failure",
                "error": True,
            },
            {
                "role": "tool",
                "requestor": "assistant",
                "id": "call_15",
                "content": {"status": "ok"},
                "error": False,
            },
        ]
    }

    assert _collect_tool_results(simulation_run) == {
        "call_15": {
            "result": {"status": "ok"},
            "error": None,
        }
    }
