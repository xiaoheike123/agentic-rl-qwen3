"""Serializable trajectory records shared by rollout, reward, and training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


SCHEMA_VERSION = 1


class EpisodeStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ToolCallRecord:
    """One tool call and the environment or control result it produced."""

    call_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    result_received: bool = False
    duration_ms: float | None = None
    is_control: bool = False

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("call_id must not be empty")

        if not self.name.strip():
            raise ValueError("tool name must not be empty")

        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")

        if self.error is not None and not self.result_received:
            raise ValueError("a tool error requires result_received=True")

    @property
    def succeeded(self) -> bool:
        return self.result_received and self.error is None


@dataclass(slots=True)
class TokenTrace:
    """Token-level policy data required by on-policy training."""

    prompt_token_ids: list[int] = field(default_factory=list)
    response_token_ids: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    response_loss_mask: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        response_length = len(self.response_token_ids)

        if self.response_logprobs and len(self.response_logprobs) != response_length:
            raise ValueError("response_logprobs must align with response_token_ids")

        if self.response_loss_mask and len(self.response_loss_mask) != response_length:
            raise ValueError("response_loss_mask must align with response_token_ids")

        if any(value not in (0, 1) for value in self.response_loss_mask):
            raise ValueError("response_loss_mask values must be zero or one")

    @property
    def is_populated(self) -> bool:
        return bool(self.response_token_ids)


@dataclass(slots=True)
class TurnRecord:
    """One policy decision followed by one environment transition."""

    turn_index: int
    observation: str
    prompt_messages: list[dict[str, Any]]
    action: str
    next_observation: str
    assistant_message: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    token_trace: TokenTrace = field(default_factory=TokenTrace)
    environment_reward: float = 0.0
    process_reward: float | None = None
    hindsight_credit: float | None = None
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative")

        if not isinstance(self.observation, str):
            raise TypeError("observation must be a string")

        if not isinstance(self.next_observation, str):
            raise TypeError("next_observation must be a string")

        if not self.prompt_messages:
            raise ValueError("prompt_messages must not be empty")

        if not all(isinstance(message, dict) for message in self.prompt_messages):
            raise TypeError("every prompt message must be a dictionary")

        if not self.action.strip():
            raise ValueError("action must not be empty")

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurnRecord:
        values = dict(data)
        values["tool_calls"] = [
            ToolCallRecord(**item) for item in values.get("tool_calls", [])
        ]
        values["token_trace"] = TokenTrace(**values.get("token_trace", {}))
        return cls(**values)


@dataclass(slots=True)
class RewardRecord:
    """Episode-level evaluator outputs and auditable reward components."""

    outcome: float | None = None
    process: float | None = None
    total: float | None = None
    components: dict[str, float] = field(default_factory=dict)
    evaluator_info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeRecord:
    """One complete tau2 trajectory and its training metadata."""

    episode_id: str
    group_id: str
    domain: str
    task_id: str
    model: str
    sample_index: int = 0
    trial_id: int = 0
    seed: int | None = None
    policy_version: int | None = None
    user_model: str | None = None
    turns: list[TurnRecord] = field(default_factory=list)
    reward: RewardRecord = field(default_factory=RewardRecord)
    success: bool | None = None
    status: EpisodeStatus = EpisodeStatus.RUNNING
    error: str | None = None
    termination_reason: str | None = None
    started_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )

        for name, value in (
            ("episode_id", self.episode_id),
            ("group_id", self.group_id),
            ("domain", self.domain),
            ("task_id", self.task_id),
            ("model", self.model),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")

        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")

        if self.trial_id < 0:
            raise ValueError("trial_id must be non-negative")

        self._validate_turn_order()
        self._validate_lifecycle()

    def _validate_turn_order(self) -> None:
        expected = list(range(len(self.turns)))
        actual = [turn.turn_index for turn in self.turns]

        if actual != expected:
            raise ValueError(
                f"turn indices must be contiguous; expected {expected}, got {actual}"
            )

    def _validate_lifecycle(self) -> None:
        if self.status is EpisodeStatus.RUNNING:
            if self.finished_at is not None:
                raise ValueError("a running episode cannot have finished_at")
            if self.error is not None:
                raise ValueError("a running episode cannot have an error")
            return

        if self.finished_at is None:
            raise ValueError("a finalized episode must have finished_at")

        if self.status is EpisodeStatus.COMPLETED:
            if self.success is None:
                raise ValueError("a completed episode must define success")
            if self.error is not None:
                raise ValueError("a completed episode cannot have an error")
            return

        if self.status is EpisodeStatus.FAILED:
            if self.success is not False:
                raise ValueError("a failed episode must have success=False")
            if self.error is None or not self.error.strip():
                raise ValueError("a failed episode must contain an error")

    def append_turn(self, turn: TurnRecord) -> None:
        if self.status is not EpisodeStatus.RUNNING:
            raise RuntimeError("cannot append a turn after the episode is finalized")

        expected_index = len(self.turns)

        if turn.turn_index != expected_index:
            raise ValueError(
                f"expected turn_index {expected_index}, got {turn.turn_index}"
            )

        if self.done:
            raise RuntimeError("cannot append a turn after the episode has ended")

        self.turns.append(turn)

    @property
    def done(self) -> bool:
        return bool(self.turns and self.turns[-1].done)

    def finish(
        self,
        *,
        reward: RewardRecord,
        success: bool,
        termination_reason: str,
    ) -> None:
        if self.status is not EpisodeStatus.RUNNING:
            raise RuntimeError("episode has already been finalized")

        if not self.done:
            raise RuntimeError("cannot finish an episode before its final turn")

        if not termination_reason.strip():
            raise ValueError("termination_reason must not be empty")

        self.reward = reward
        self.success = success
        self.termination_reason = termination_reason
        self.status = EpisodeStatus.COMPLETED
        self.finished_at = utc_now()

    def fail(self, *, error: str, termination_reason: str) -> None:
        if self.status is not EpisodeStatus.RUNNING:
            raise RuntimeError("episode has already been finalized")

        if not error.strip():
            raise ValueError("error must not be empty")

        if not termination_reason.strip():
            raise ValueError("termination_reason must not be empty")

        self.status = EpisodeStatus.FAILED
        self.error = error
        self.success = False
        self.termination_reason = termination_reason
        self.finished_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeRecord:
        values = dict(data)
        values["status"] = EpisodeStatus(values.get("status", EpisodeStatus.RUNNING))
        values["turns"] = [
            TurnRecord.from_dict(item) for item in values.get("turns", [])
        ]
        values["reward"] = RewardRecord(**values.get("reward", {}))
        return cls(**values)
