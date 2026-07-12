"""Environment-verifiable process reward for tool-agent trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_rl.data.schemas import EpisodeRecord
from agent_rl.rewards.environment_checks import (
    collect_tool_executions,
    count_excess_identical_calls,
    find_error_recoveries,
)


@dataclass(frozen=True, slots=True)
class ProcessRewardConfig:
    """Weights for deterministic process evidence."""

    tool_success_reward: float = 0.0
    invalid_action_penalty: float = 1.0
    tool_error_penalty: float = 1.0
    missing_result_penalty: float = 1.0
    recovery_bonus: float = 0.5
    unresolved_error_penalty: float = 0.5
    abnormal_truncation_penalty: float = 1.0
    repeated_call_diagnostic_after: int = 2
    minimum_score: float = -1.0
    maximum_score: float = 1.0

    def __post_init__(self) -> None:
        non_negative = (
            "tool_success_reward",
            "invalid_action_penalty",
            "tool_error_penalty",
            "missing_result_penalty",
            "recovery_bonus",
            "unresolved_error_penalty",
            "abnormal_truncation_penalty",
        )

        for name in non_negative:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

        if self.repeated_call_diagnostic_after <= 0:
            raise ValueError(
                "repeated_call_diagnostic_after must be greater than zero"
            )

        if self.minimum_score >= self.maximum_score:
            raise ValueError("minimum_score must be less than maximum_score")


@dataclass(frozen=True, slots=True)
class ProcessCheck:
    """One auditable contribution to process reward."""

    name: str
    turn_index: int
    score: float
    passed: bool
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProcessRewardResult:
    """Normalized episode score and token-alignable turn scores."""

    total: float
    turn_scores: tuple[float, ...]
    checks: tuple[ProcessCheck, ...]
    normalization_count: int


class EnvironmentProcessReward:
    """Score only behavior that can be verified from tau2 transitions."""

    def __init__(
        self,
        config: ProcessRewardConfig | None = None,
    ) -> None:
        self.config = config or ProcessRewardConfig()

    def evaluate(self, episode: EpisodeRecord) -> ProcessRewardResult:
        turn_scores = [0.0 for _ in episode.turns]
        checks: list[ProcessCheck] = []
        executions = collect_tool_executions(episode)

        for turn in episode.turns:
            if turn.info.get("action_parse_valid", True) is False:
                score = -self.config.invalid_action_penalty
                turn_scores[turn.turn_index] += score
                checks.append(
                    ProcessCheck(
                        name="action_parse",
                        turn_index=turn.turn_index,
                        score=score,
                        passed=False,
                        evidence={"error": turn.info.get("action_parse_error")},
                    )
                )

        for execution in executions:
            if not execution.result_received:
                score = -self.config.missing_result_penalty
                check_name = "tool_result_received"
                passed = False
            elif execution.error is not None:
                score = -self.config.tool_error_penalty
                check_name = "tool_execution"
                passed = False
            else:
                score = self.config.tool_success_reward
                check_name = "tool_execution"
                passed = True

            turn_scores[execution.turn_index] += score
            checks.append(
                ProcessCheck(
                    name=check_name,
                    turn_index=execution.turn_index,
                    score=score,
                    passed=passed,
                    evidence={
                        "call_id": execution.call_id,
                        "tool": execution.name,
                        "error": execution.error,
                    },
                )
            )

        excess_by_turn = count_excess_identical_calls(
            executions,
            allowed_occurrences=self.config.repeated_call_diagnostic_after,
        )

        for turn_index, excess_count in excess_by_turn.items():
            checks.append(
                ProcessCheck(
                    name="repeated_identical_call_observed",
                    turn_index=turn_index,
                    score=0.0,
                    passed=False,
                    evidence={
                        "excess_count": excess_count,
                        "training_penalty_applied": False,
                    },
                )
            )

        rewarded_recovery_turns: set[int] = set()

        for recovery in find_error_recoveries(executions):
            if recovery.recovered:
                recovery_turn = recovery.recovery_turn_index
                assert recovery_turn is not None
                score = 0.0

                if recovery_turn not in rewarded_recovery_turns:
                    score = self.config.recovery_bonus
                    rewarded_recovery_turns.add(recovery_turn)
                    turn_scores[recovery_turn] += score

                checks.append(
                    ProcessCheck(
                        name="error_recovery",
                        turn_index=recovery_turn,
                        score=score,
                        passed=True,
                        evidence={
                            "error_turn_index": recovery.error_turn_index,
                        },
                    )
                )
            else:
                score = -self.config.unresolved_error_penalty
                turn_scores[recovery.error_turn_index] += score
                checks.append(
                    ProcessCheck(
                        name="error_recovery",
                        turn_index=recovery.error_turn_index,
                        score=score,
                        passed=False,
                        evidence={"recovery_turn_index": None},
                    )
                )

        hit_max_steps = episode.termination_reason == "max_steps"
        was_truncated = any(turn.truncated for turn in episode.turns)
        if hit_max_steps or was_truncated:
            final_turn_index = len(episode.turns) - 1
            score = -self.config.abnormal_truncation_penalty
            turn_scores[final_turn_index] += score
            checks.append(
                ProcessCheck(
                    name="abnormal_truncation",
                    turn_index=final_turn_index,
                    score=score,
                    passed=False,
                    evidence={
                        "termination_reason": episode.termination_reason,
                        "turn_truncated": was_truncated,
                    },
                )
            )

        normalization_count = 1
        final_turn_scores = tuple(turn_scores)
        raw_total = sum(final_turn_scores)
        total = min(
            self.config.maximum_score,
            max(self.config.minimum_score, raw_total),
        )

        if raw_total != 0.0 and total != raw_total:
            scale = total / raw_total
            final_turn_scores = tuple(score * scale for score in final_turn_scores)

        for turn, score in zip(episode.turns, final_turn_scores):
            turn.process_reward = score

        return ProcessRewardResult(
            total=total,
            turn_scores=final_turn_scores,
            checks=tuple(checks),
            normalization_count=normalization_count,
        )
