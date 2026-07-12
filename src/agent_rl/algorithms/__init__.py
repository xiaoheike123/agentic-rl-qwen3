"""Advantage and grouping algorithms."""

from agent_rl.algorithms.aggregation import (
    AggregationExample,
    AggregationMode,
    AggregationWeights,
    aggregate_scalar_losses,
    compute_aggregation_weights,
)
from agent_rl.algorithms.advantage_utils import (
    EpisodeAdvantage,
    GroupAdvantageConfig,
    compute_group_advantages,
    episode_training_reward,
)

__all__ = [
    "AggregationExample",
    "AggregationMode",
    "AggregationWeights",
    "aggregate_scalar_losses",
    "compute_aggregation_weights",
    "EpisodeAdvantage",
    "GroupAdvantageConfig",
    "compute_group_advantages",
    "episode_training_reward",
]
