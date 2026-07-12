"""Deterministic failure categories for completed rollout records."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from agent_rl.data.schemas import EpisodeRecord, EpisodeStatus


@dataclass(frozen=True, slots=True)
class ErrorSummary:
    total: int
    successful: int
    failed: int
    categories: dict[str, int]


def classify_episode(episode: EpisodeRecord) -> str:
    if episode.status is EpisodeStatus.FAILED:
        return episode.termination_reason or "rollout_error"
    if episode.success:
        return "success"
    if any(call.error for turn in episode.turns for call in turn.tool_calls):
        return "unrecovered_tool_error"
    if episode.termination_reason in {"max_steps", "response_length"}:
        return episode.termination_reason
    if episode.reward.outcome == 0:
        return "evaluator_failure"
    return "unsuccessful"


def summarize_errors(episodes: Sequence[EpisodeRecord]) -> ErrorSummary:
    counts = Counter(classify_episode(episode) for episode in episodes)
    successful = counts.get("success", 0)
    return ErrorSummary(
        total=len(episodes),
        successful=successful,
        failed=len(episodes) - successful,
        categories=dict(sorted(counts.items())),
    )
