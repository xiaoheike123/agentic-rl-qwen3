"""Deterministic injection of recoverable failures before tool execution."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable

from tau2.data_model.message import ToolMessage


@dataclass(frozen=True, slots=True)
class ToolFailurePlan:
    tool_name: str | None = None
    failure_message: str = "Temporary service unavailable. Please retry."
    failures: int = 1
    requestor: str = "assistant"

    def __post_init__(self) -> None:
        if self.tool_name is not None and not self.tool_name.strip():
            raise ValueError("tool_name must not be blank")
        if not self.failure_message.strip():
            raise ValueError("failure_message must not be empty")
        if self.failures <= 0:
            raise ValueError("failures must be greater than zero")
        if self.requestor not in {"assistant", "user"}:
            raise ValueError("requestor must be 'assistant' or 'user'")


class RecoverableToolFailureInjector:
    """Wrap a tau2 Environment and fail the first matching calls safely."""

    def __init__(self, plan: ToolFailurePlan) -> None:
        self.plan = plan
        self._remaining = plan.failures
        self._injected = 0
        self._lock = Lock()

    @property
    def injected_count(self) -> int:
        with self._lock:
            return self._injected

    def transform(self, environment: Any) -> Any:
        original_get_response: Callable[[Any], ToolMessage] = environment.get_response

        def get_response(message: Any) -> ToolMessage:
            should_fail = getattr(
                message, "requestor", None
            ) == self.plan.requestor and (
                self.plan.tool_name is None
                or getattr(message, "name", None) == self.plan.tool_name
            )

            with self._lock:
                inject = should_fail and self._remaining > 0
                if inject:
                    self._remaining -= 1
                    self._injected += 1

            if not inject:
                return original_get_response(message)

            return ToolMessage(
                id=message.id,
                content=self.plan.failure_message,
                requestor=message.requestor,
                role="tool",
                error=True,
            )

        environment.get_response = get_response
        return environment
