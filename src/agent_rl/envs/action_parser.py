"""Parse model outputs into tau2-compatible actions."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence


class ActionFormatError(ValueError):
    """Raised when a model output cannot become one valid tau2 action."""


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    name: str
    arguments: dict[str, Any]
    tool_call_id: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ActionFormatError("Tool name must not be empty")

        if not isinstance(self.arguments, dict):
            raise ActionFormatError("Tool arguments must be a dictionary")


def to_tau_action(
    *,
    content: str | None,
    tool_calls: Sequence[ModelToolCall] | None = None,
) -> str:
    calls = tuple(tool_calls or ())
    has_content = content is not None and bool(content.strip())

    if has_content and calls:
        raise ActionFormatError(
            "A tau2 action cannot contain both text and tool calls"
        )

    if len(calls) > 1:
        raise ActionFormatError(
            "A tau2 action can contain at most one tool call"
        )

    if calls:
        call = calls[0]
        payload = {
            "id": call.tool_call_id,
            "name": call.name,
            "arguments": call.arguments,
            "requestor": "assistant",
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    if has_content:
        return content.strip()

    raise ActionFormatError(
        "Model output must contain text or one tool call"
    )
