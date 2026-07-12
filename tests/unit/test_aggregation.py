from agent_rl.algorithms.aggregation import (
    AggregationExample,
    aggregate_scalar_losses,
    compute_aggregation_weights,
)


def _examples():
    return (
        AggregationExample("positive", 1.0, 2),
        AggregationExample("negative", -1.0, 4),
    )


def test_token_aggregation_is_global_token_mean():
    value = aggregate_scalar_losses(
        ((1.0, 1.0), (3.0, 3.0, 3.0, 3.0)), _examples(), "token"
    )
    assert value == 14.0 / 6.0


def test_sequence_aggregation_gives_episodes_equal_weight():
    value = aggregate_scalar_losses(
        ((1.0, 1.0), (3.0, 3.0, 3.0, 3.0)), _examples(), "sequence"
    )
    assert value == 2.0


def test_balanced_weights_are_sign_count_weighted():
    weights = compute_aggregation_weights(_examples(), "balanced")
    assert weights.positive_episodes == 1
    assert weights.negative_episodes == 1
    assert weights.episode_weights == (1.5, 0.75)
