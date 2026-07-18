from agent_rl.data.schemas import (
    EpisodeRecord,
    RewardRecord,
    ToolCallRecord,
    TurnRecord,
)
from agent_rl.rewards.process_reward import EnvironmentProcessReward


def test_process_reward_records_error_and_recovery():
    episode = EpisodeRecord(
        episode_id="episode",
        group_id="group",
        domain="mock",
        task_id="task",
        model="model",
    )
    episode.append_turn(
        TurnRecord(
            turn_index=0,
            observation="request",
            prompt_messages=[{"role": "user", "content": "request"}],
            action="tool()",
            next_observation="temporary error",
            tool_calls=[
                ToolCallRecord(
                    call_id="one",
                    name="tool",
                    arguments={"id": 1},
                    error="temporary error",
                    result_received=True,
                )
            ],
        )
    )
    episode.append_turn(
        TurnRecord(
            turn_index=1,
            observation="temporary error",
            prompt_messages=[{"role": "user", "content": "temporary error"}],
            action="tool()",
            next_observation="ok",
            tool_calls=[
                ToolCallRecord(
                    call_id="two",
                    name="tool",
                    arguments={"id": 1},
                    result="ok",
                    result_received=True,
                )
            ],
            terminated=True,
        )
    )
    result = EnvironmentProcessReward().evaluate(episode)
    assert any(
        check.name == "error_recovery" and check.passed for check in result.checks
    )
    assert episode.turns[0].process_reward is not None
    assert episode.turns[1].process_reward is not None
    assert result.normalization_count == 1
    assert result.turn_scores == (-0.25, 0.25)
    assert result.total == 0.0


def test_corrected_same_tool_call_counts_as_recovery():
    episode = EpisodeRecord(
        episode_id="episode",
        group_id="group",
        domain="mock",
        task_id="task",
        model="model",
    )
    episode.append_turn(
        TurnRecord(
            turn_index=0,
            observation="request",
            prompt_messages=[{"role": "user", "content": "request"}],
            action="tool()",
            next_observation="bad identifier",
            tool_calls=[
                ToolCallRecord(
                    call_id="one",
                    name="tool",
                    arguments={"id": "wrong"},
                    error="not found",
                    result_received=True,
                )
            ],
        )
    )
    episode.append_turn(
        TurnRecord(
            turn_index=1,
            observation="bad identifier",
            prompt_messages=[{"role": "user", "content": "bad identifier"}],
            action="tool()",
            next_observation="ok",
            tool_calls=[
                ToolCallRecord(
                    call_id="two",
                    name="tool",
                    arguments={"id": "correct"},
                    result="ok",
                    result_received=True,
                )
            ],
            terminated=True,
        )
    )

    result = EnvironmentProcessReward().evaluate(episode)

    recovery = next(
        check for check in result.checks if check.name == "error_recovery"
    )
    assert recovery.passed is True
    assert recovery.evidence["match_kind"] == "corrected_retry"
    assert result.turn_scores == (-0.25, 0.25)
    assert result.total == 0.0


def test_successful_calls_do_not_dilute_an_error_penalty():
    episode = EpisodeRecord(
        episode_id="episode",
        group_id="group",
        domain="mock",
        task_id="task",
        model="model",
    )
    episode.append_turn(
        TurnRecord(
            turn_index=0,
            observation="request",
            prompt_messages=[{"role": "user", "content": "request"}],
            action="tool()",
            next_observation="error",
            tool_calls=[
                ToolCallRecord(
                    call_id="error",
                    name="tool",
                    arguments={"id": 1},
                    error="permanent error",
                    result_received=True,
                )
            ],
        )
    )
    for index in range(1, 5):
        episode.append_turn(
            TurnRecord(
                turn_index=index,
                observation="continue",
                prompt_messages=[{"role": "user", "content": "continue"}],
                action="other_tool()",
                next_observation="ok",
                tool_calls=[
                    ToolCallRecord(
                        call_id=f"success-{index}",
                        name=f"other_tool_{index}",
                        arguments={"id": index},
                        result="ok",
                        result_received=True,
                    )
                ],
            )
        )

    result = EnvironmentProcessReward().evaluate(episode)

    assert result.total == -0.25
    assert result.turn_scores[0] == -0.25
    assert sum(result.turn_scores[1:]) == 0.0


def test_multiple_unresolved_errors_accumulate_before_clipping():
    episode = EpisodeRecord(
        episode_id="episode",
        group_id="group",
        domain="mock",
        task_id="task",
        model="model",
    )
    for index in range(5):
        episode.append_turn(
            TurnRecord(
                turn_index=index,
                observation="request",
                prompt_messages=[{"role": "user", "content": "request"}],
                action="tool()",
                next_observation="error",
                tool_calls=[
                    ToolCallRecord(
                        call_id=f"error-{index}",
                        name=f"tool_{index}",
                        arguments={"id": index},
                        error="error",
                        result_received=True,
                    )
                ],
            )
        )

    result = EnvironmentProcessReward().evaluate(episode)

    assert result.total == -1.0
    assert result.turn_scores == (-0.2, -0.2, -0.2, -0.2, -0.2)
    unresolved = [
        check
        for check in result.checks
        if check.name == "error_recovery" and not check.passed
    ]
    assert len(unresolved) == 5
    assert all(check.score == 0.0 for check in unresolved)


def test_repeated_identical_calls_are_diagnostic_only():
    episode = EpisodeRecord(
        episode_id="episode",
        group_id="group",
        domain="mock",
        task_id="task",
        model="model",
    )
    for index in range(4):
        episode.append_turn(
            TurnRecord(
                turn_index=index,
                observation="query",
                prompt_messages=[{"role": "user", "content": "query"}],
                action="get_user()",
                next_observation="same user",
                tool_calls=[
                    ToolCallRecord(
                        call_id=f"call-{index}",
                        name="get_user",
                        arguments={"user_id": "user_1"},
                        result={"id": "user_1"},
                        result_received=True,
                    )
                ],
            )
        )

    result = EnvironmentProcessReward().evaluate(episode)

    assert result.total == 0.0
    assert result.turn_scores == (0.0, 0.0, 0.0, 0.0)
    diagnostics = [
        check
        for check in result.checks
        if check.name == "repeated_identical_call_observed"
    ]
    assert len(diagnostics) == 2
    assert all(check.score == 0.0 for check in diagnostics)
    assert all(
        check.evidence["training_penalty_applied"] is False
        for check in diagnostics
    )


def test_invalid_action_receives_process_penalty():
    episode = EpisodeRecord(
        episode_id="episode",
        group_id="group",
        domain="mock",
        task_id="task",
        model="model",
    )
    episode.append_turn(
        TurnRecord(
            turn_index=0,
            observation="request",
            prompt_messages=[{"role": "user", "content": "request"}],
            action="malformed action",
            next_observation="invalid",
            info={
                "action_parse_valid": False,
                "action_parse_error": "invalid syntax",
            },
        )
    )

    result = EnvironmentProcessReward().evaluate(episode)

    assert result.total == -1.0
    assert result.turn_scores == (-1.0,)
    assert any(check.name == "action_parse" for check in result.checks)


def test_max_steps_receives_one_final_turn_penalty():
    episode = EpisodeRecord(
        episode_id="episode",
        group_id="group",
        domain="mock",
        task_id="task",
        model="model",
    )
    episode.append_turn(
        TurnRecord(
            turn_index=0,
            observation="request",
            prompt_messages=[{"role": "user", "content": "request"}],
            action="still working",
            next_observation="continue",
            terminated=True,
        )
    )
    episode.finish(
        reward=RewardRecord(outcome=0.0, total=0.0),
        success=False,
        termination_reason="max_steps",
    )

    result = EnvironmentProcessReward().evaluate(episode)

    assert result.total == -1.0
    assert result.turn_scores == (-1.0,)
    assert sum(
        check.name == "abnormal_truncation" for check in result.checks
    ) == 1
