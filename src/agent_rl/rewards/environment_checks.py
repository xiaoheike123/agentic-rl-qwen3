"""Deterministic evidence extracted from tau2 tool transitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator

from agent_rl.data.schemas import EpisodeRecord, ToolCallRecord


@dataclass(frozen=True, slots=True)
class ToolExecutionEvidence:
    """One model tool call paired with its authoritative tau2 result."""

    turn_index: int
    call_index: int
    call_id: str
    name: str
    arguments: dict[str, Any]
    result_received: bool
    error: str | None

    @property
    def succeeded(self) -> bool:
        return self.result_received and self.error is None

    @property
    def fingerprint(self) -> str:
        arguments = json.dumps(
            self.arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return f"{self.name}:{arguments}"


@dataclass(frozen=True, slots=True)
class RecoveryEvidence:
    """Whether a later tool call recovered from an environment error."""

    error_turn_index: int
    recovery_turn_index: int | None

    @property
    def recovered(self) -> bool:
        return self.recovery_turn_index is not None


def iter_tool_executions(
    episode: EpisodeRecord,
) -> Iterator[ToolExecutionEvidence]:
    for turn in episode.turns:
        for call_index, call in enumerate(turn.tool_calls):
            yield _to_execution_evidence(
                turn_index=turn.turn_index,
                call_index=call_index,
                call=call,
            )


def collect_tool_executions(
    episode: EpisodeRecord,
) -> tuple[ToolExecutionEvidence, ...]:
    return tuple(iter_tool_executions(episode))


def find_error_recoveries(
    executions: tuple[ToolExecutionEvidence, ...],
) -> tuple[RecoveryEvidence, ...]:
    """Match each error to the first later successful tool execution."""

    recoveries: list[RecoveryEvidence] = []

    for index, execution in enumerate(executions):
        if execution.error is None:
            continue

        recovery_turn_index: int | None = None

        for candidate in executions[index + 1 :]:
            if candidate.succeeded and candidate.fingerprint == execution.fingerprint:
                recovery_turn_index = candidate.turn_index
                break

        recoveries.append(
            RecoveryEvidence(
                error_turn_index=execution.turn_index,
                recovery_turn_index=recovery_turn_index,
            )
        )

    return tuple(recoveries)


def count_excess_identical_calls(
    executions: tuple[ToolExecutionEvidence, ...],
    *,
    allowed_occurrences: int,
) -> dict[int, int]:
    """Count repeated identical calls beyond an explicit allowance."""

    if allowed_occurrences <= 0:
        raise ValueError("allowed_occurrences must be greater than zero")

    counts: dict[str, int] = {}
    excess_by_turn: dict[int, int] = {}

    for execution in executions:
        count = counts.get(execution.fingerprint, 0) + 1
        counts[execution.fingerprint] = count

        if count > allowed_occurrences:
            excess_by_turn[execution.turn_index] = (
                excess_by_turn.get(execution.turn_index, 0) + 1
            )

    return excess_by_turn


def _to_execution_evidence(
    *,
    turn_index: int,
    call_index: int,
    call: ToolCallRecord,
) -> ToolExecutionEvidence:
    return ToolExecutionEvidence(
        turn_index=turn_index,
        call_index=call_index,
        call_id=call.call_id,
        name=call.name,
        arguments=dict(call.arguments),
        result_received=call.result_received,
        error=call.error,
    )
