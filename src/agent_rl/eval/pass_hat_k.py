"""Official tau2 pass^k metric for repeated trials."""

from __future__ import annotations

from math import comb
from typing import Iterable


def estimate_pass_hat_k(*, trials: int, successes: int, k: int) -> float:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between zero and trials")
    if not 1 <= k <= trials:
        raise ValueError("k must be between one and trials")
    return comb(successes, k) / comb(trials, k)


def pass_hat_k_from_outcomes(outcomes: Iterable[bool], k: int) -> float:
    values = tuple(bool(value) for value in outcomes)
    return estimate_pass_hat_k(
        trials=len(values),
        successes=sum(values),
        k=k,
    )
