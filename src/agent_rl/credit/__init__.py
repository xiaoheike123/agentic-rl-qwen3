"""Turn-level credit assignment independent of reward generation."""

from .hindsight_credit import (
    HindsightCreditAssigner,
    HindsightCreditConfig,
    HindsightCreditResult,
    compute_advantage_aligned_turn_weights,
)
from .turn_advantage import (
    TurnAdvantageResult,
    project_turn_advantages,
)

__all__ = [
    "HindsightCreditAssigner",
    "HindsightCreditConfig",
    "HindsightCreditResult",
    "compute_advantage_aligned_turn_weights",
    "TurnAdvantageResult",
    "project_turn_advantages",
]
