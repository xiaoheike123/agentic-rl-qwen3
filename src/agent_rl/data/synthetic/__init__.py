"""Benchmark-safe synthetic task generation for tau2 training."""

from agent_rl.data.synthetic.schema import (
    GenerationMetadata,
    OverlapMetadata,
    SyntheticSplit,
    SyntheticTaskRecord,
    VerificationMetadata,
)

__all__ = [
    "GenerationMetadata",
    "OverlapMetadata",
    "SyntheticSplit",
    "SyntheticTaskRecord",
    "VerificationMetadata",
]
