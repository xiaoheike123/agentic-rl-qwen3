"""Benchmark-safe synthetic task generation for tau2 training."""

from agent_rl.data.synthetic.schema import (
    GenerationMetadata,
    SyntheticSplit,
    SyntheticTaskRecord,
    VerificationMetadata,
)

__all__ = [
    "GenerationMetadata",
    "SyntheticSplit",
    "SyntheticTaskRecord",
    "VerificationMetadata",
]
