"""Project trajectory advantages onto turns and generated tokens."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from agent_rl.data.schemas import EpisodeRecord
from agent_rl.credit.hindsight_credit import (
    HindsightCreditResult,
    compute_advantage_aligned_turn_weights,
)


@dataclass(frozen=True, slots=True)
class TurnAdvantageResult:
    trajectory_advantage: float
    turn_advantages: tuple[float, ...]
    token_advantages: tuple[tuple[float, ...], ...]

    @property
    def flat_token_advantages(self) -> tuple[float, ...]:
        return tuple(
            value for turn_values in self.token_advantages for value in turn_values
        )


def project_turn_advantages(
    episode: EpisodeRecord,
    *,
    trajectory_advantage: float,
    credit: HindsightCreditResult | None = None,
    require_token_traces: bool = True,
) -> TurnAdvantageResult:
    if not isfinite(trajectory_advantage):
        raise ValueError("trajectory_advantage must be finite")

    if not episode.turns:
        raise ValueError("episode must contain at least one turn")

    if credit is None:
        turn_weights = tuple(1.0 for _ in episode.turns)
    else:
        turn_weights = compute_advantage_aligned_turn_weights(
            credit.turn_evidence,
            trajectory_advantage=trajectory_advantage,
        )

    if len(turn_weights) != len(episode.turns):
        raise ValueError("credit weights must align with episode turns")

    turn_advantages = tuple(trajectory_advantage * weight for weight in turn_weights)
    token_advantages: list[tuple[float, ...]] = []

    for turn, turn_advantage in zip(
        episode.turns,
        turn_advantages,
    ):
        trace = turn.token_trace

        if not trace.response_token_ids:
            if require_token_traces:
                raise ValueError(f"turn {turn.turn_index} has no response token IDs")

            token_advantages.append(())
            continue

        mask = (
            trace.response_loss_mask
            if trace.response_loss_mask
            else [1 for _ in trace.response_token_ids]
        )
        token_advantages.append(
            tuple(turn_advantage if include else 0.0 for include in mask)
        )

    return TurnAdvantageResult(
        trajectory_advantage=trajectory_advantage,
        turn_advantages=turn_advantages,
        token_advantages=tuple(token_advantages),
    )
