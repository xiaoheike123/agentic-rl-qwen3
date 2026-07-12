"""Stateless lightweight compression for long tau2 observations."""

from __future__ import annotations

import re
from dataclasses import dataclass


_ROLE_PATTERN = re.compile(
    r"^(user|assistant|tool|system):(?:\s|$)",
    flags=re.IGNORECASE,
)

_DEFAULT_ERROR_KEYWORDS = (
    "error",
    "failed",
    "failure",
    "invalid",
    "timeout",
    "rate limit",
    "unavailable",
    "denied",
    "not found",
)


@dataclass(frozen=True, slots=True)
class ContextCompressionConfig:
    """Character-budget policy for deterministic context compression."""

    max_chars: int = 60_000
    recent_events: int = 12
    max_event_chars: int = 12_000
    first_user_budget: int = 8_000
    minimum_event_budget: int = 256
    error_keywords: tuple[str, ...] = _DEFAULT_ERROR_KEYWORDS

    def __post_init__(self) -> None:
        if self.max_chars < 512:
            raise ValueError("max_chars must be at least 512")

        if self.recent_events <= 0:
            raise ValueError("recent_events must be greater than zero")

        if self.max_event_chars <= 0:
            raise ValueError("max_event_chars must be greater than zero")

        if self.first_user_budget <= 0:
            raise ValueError("first_user_budget must be greater than zero")

        if self.minimum_event_budget <= 0:
            raise ValueError("minimum_event_budget must be greater than zero")

        if self.max_event_chars > self.max_chars:
            raise ValueError("max_event_chars cannot exceed max_chars")

        if self.first_user_budget > self.max_chars:
            raise ValueError("first_user_budget cannot exceed max_chars")

        if self.minimum_event_budget > self.max_chars:
            raise ValueError("minimum_event_budget cannot exceed max_chars")

        if not self.error_keywords:
            raise ValueError("error_keywords must not be empty")

        if any(not keyword.strip() for keyword in self.error_keywords):
            raise ValueError("error keywords must not be empty")


@dataclass(frozen=True, slots=True)
class ContextCompressionResult:
    """Compressed text and auditable compression statistics."""

    text: str
    applied: bool
    original_chars: int
    compressed_chars: int
    original_events: int
    retained_events: int
    dropped_events: int

    @property
    def compression_ratio(self) -> float:
        if self.original_chars == 0:
            return 1.0

        return self.compressed_chars / self.original_chars


@dataclass(frozen=True, slots=True)
class _ContextEvent:
    index: int
    role: str
    text: str


class LightweightContextCompressor:
    """Compress a full observation without persistent memory or LLM calls."""

    _HEADER_RESERVE = 200

    def __init__(
        self,
        config: ContextCompressionConfig | None = None,
    ) -> None:
        self.config = config or ContextCompressionConfig()

    def compress(self, observation: str) -> ContextCompressionResult:
        if not isinstance(observation, str):
            raise TypeError("observation must be a string")

        events = self._split_events(observation)
        original_chars = len(observation)

        if original_chars <= self.config.max_chars:
            return ContextCompressionResult(
                text=observation,
                applied=False,
                original_chars=original_chars,
                compressed_chars=original_chars,
                original_events=len(events),
                retained_events=len(events),
                dropped_events=0,
            )

        selected = self._select_events(events)
        selected_indices = sorted(selected)
        retained_events = len(selected_indices)
        dropped_events = len(events) - retained_events

        header = (
            "[Context compressed deterministically: "
            f"retained {retained_events} of {len(events)} events; "
            f"omitted {dropped_events} older events.]"
        )

        parts = [header]
        parts.extend(selected[index] for index in selected_indices)
        compressed = "\n".join(parts)

        if len(compressed) > self.config.max_chars:
            raise RuntimeError("context compressor exceeded its configured budget")

        return ContextCompressionResult(
            text=compressed,
            applied=True,
            original_chars=original_chars,
            compressed_chars=len(compressed),
            original_events=len(events),
            retained_events=retained_events,
            dropped_events=dropped_events,
        )

    def _select_events(
        self,
        events: list[_ContextEvent],
    ) -> dict[int, str]:
        available = self.config.max_chars - self._HEADER_RESERVE

        if available <= 0:
            raise RuntimeError("max_chars is too small for compression metadata")

        selected: dict[int, str] = {}

        first_user = next(
            (event for event in events if event.role == "user"),
            None,
        )

        if first_user is not None:
            available = self._try_select(
                event=first_user,
                selected=selected,
                available=available,
                preferred_limit=self.config.first_user_budget,
                required=True,
            )

        recent_start = max(
            0,
            len(events) - self.config.recent_events,
        )

        for event in reversed(events[recent_start:]):
            available = self._try_select(
                event=event,
                selected=selected,
                available=available,
                preferred_limit=self.config.max_event_chars,
                required=event.index == len(events) - 1,
            )

        important_events = [
            event
            for event in events[:recent_start]
            if event.index not in selected and self._is_important(event)
        ]

        for event in reversed(important_events):
            available = self._try_select(
                event=event,
                selected=selected,
                available=available,
                preferred_limit=self.config.max_event_chars,
                required=False,
            )

        if not selected and events:
            self._try_select(
                event=events[-1],
                selected=selected,
                available=available,
                preferred_limit=self.config.max_event_chars,
                required=True,
            )

        return selected

    def _try_select(
        self,
        *,
        event: _ContextEvent,
        selected: dict[int, str],
        available: int,
        preferred_limit: int,
        required: bool,
    ) -> int:
        if event.index in selected:
            return available

        separator_cost = 1 if selected else 0
        usable = available - separator_cost

        if usable <= 0:
            return available

        if not required and usable < self.config.minimum_event_budget:
            return available

        limit = min(
            preferred_limit,
            self.config.max_event_chars,
            usable,
        )

        if limit <= 0:
            return available

        selected[event.index] = self._clip_text(
            event.text,
            limit=limit,
        )

        return available - len(selected[event.index]) - separator_cost

    def _is_important(self, event: _ContextEvent) -> bool:
        if event.role == "tool":
            return True

        lowered = event.text.casefold()

        return any(
            keyword.casefold() in lowered for keyword in self.config.error_keywords
        )

    @staticmethod
    def _split_events(observation: str) -> list[_ContextEvent]:
        if not observation:
            return []

        events: list[_ContextEvent] = []
        current_role = "unknown"
        current_lines: list[str] = []

        def flush() -> None:
            if not current_lines:
                return

            events.append(
                _ContextEvent(
                    index=len(events),
                    role=current_role,
                    text="\n".join(current_lines),
                )
            )

        for line in observation.splitlines():
            match = _ROLE_PATTERN.match(line)

            if match is not None:
                flush()
                current_lines = [line]
                current_role = match.group(1).lower()
            else:
                current_lines.append(line)

        flush()

        if not events:
            events.append(
                _ContextEvent(
                    index=0,
                    role="unknown",
                    text=observation,
                )
            )

        return events

    @staticmethod
    def _clip_text(text: str, *, limit: int) -> str:
        if len(text) <= limit:
            return text

        marker = "\n[... event content clipped ...]\n"

        if limit <= len(marker):
            return text[:limit]

        remaining = limit - len(marker)
        head_size = (remaining * 2) // 3
        tail_size = remaining - head_size

        return text[:head_size] + marker + text[-tail_size:]
