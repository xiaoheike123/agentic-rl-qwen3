"""Paired clean/perturbed robustness metrics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from agent_rl.data.schemas import EpisodeRecord


@dataclass(frozen=True, slots=True)
class RobustnessMetrics:
    pairs: int
    clean_success_rate: float
    perturbed_success_rate: float
    success_drop: float
    recovery_success_rate: float | None
    extra_steps_after_perturbation: float


def _pair_key(episode: EpisodeRecord) -> tuple[str, int, int]:
    return episode.task_id, episode.trial_id, episode.sample_index


def compute_robustness_metrics(
    clean: Sequence[EpisodeRecord],
    perturbed: Sequence[EpisodeRecord],
    *,
    tool_failure: bool = False,
) -> RobustnessMetrics:
    if not clean or not perturbed:
        raise ValueError("clean and perturbed episodes must not be empty")

    clean_by_key = {_pair_key(episode): episode for episode in clean}
    perturbed_by_key = {_pair_key(episode): episode for episode in perturbed}
    if len(clean_by_key) != len(clean) or len(perturbed_by_key) != len(perturbed):
        raise ValueError("duplicate task/trial/sample keys are not allowed")
    if clean_by_key.keys() != perturbed_by_key.keys():
        missing_clean = sorted(perturbed_by_key.keys() - clean_by_key.keys())
        missing_perturbed = sorted(clean_by_key.keys() - perturbed_by_key.keys())
        raise ValueError(
            "clean and perturbed episodes are not paired; "
            f"missing_clean={missing_clean}, "
            f"missing_perturbed={missing_perturbed}"
        )

    ordered_keys = sorted(clean_by_key)
    clean_success = [float(clean_by_key[key].success is True) for key in ordered_keys]
    perturbed_success = [
        float(perturbed_by_key[key].success is True) for key in ordered_keys
    ]
    extra_steps = [
        len(perturbed_by_key[key].turns) - len(clean_by_key[key].turns)
        for key in ordered_keys
    ]

    clean_rate = mean(clean_success)
    perturbed_rate = mean(perturbed_success)
    recovery_rate = None
    if tool_failure:
        injected_keys = [
            key
            for key in ordered_keys
            if int(
                perturbed_by_key[key].metadata.get(
                    "injected_tool_failures",
                    0,
                )
            )
            > 0
        ]
        recovery_rate = (
            mean(perturbed_by_key[key].success is True for key in injected_keys)
            if injected_keys
            else None
        )

    return RobustnessMetrics(
        pairs=len(ordered_keys),
        clean_success_rate=clean_rate,
        perturbed_success_rate=perturbed_rate,
        success_drop=clean_rate - perturbed_rate,
        recovery_success_rate=recovery_rate,
        extra_steps_after_perturbation=mean(extra_steps),
    )
