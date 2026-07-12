"""verl integration layer."""

from agent_rl.trainer.aggregation_adapter import (
    PreparedAdvantageBatch,
    PreparedEpisodeAdvantages,
    prepare_advantage_batch,
)

__all__ = [
    "PreparedAdvantageBatch",
    "PreparedEpisodeAdvantages",
    "prepare_advantage_batch",
]
