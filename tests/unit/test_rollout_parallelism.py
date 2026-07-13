from types import SimpleNamespace

import pytest

from agent_rl.trainer.verl_entry import _rollout_parallelism


def _config(*, max_concurrent, workers):
    return SimpleNamespace(
        rollout={"max_concurrent_episodes": max_concurrent},
        runtime={"agent_loop_workers": workers},
    )


def test_parallelism_caps_workers_and_computes_per_worker_limit():
    assert _rollout_parallelism(_config(max_concurrent=8, workers=8)) == (8, 1)
    assert _rollout_parallelism(_config(max_concurrent=4, workers=8)) == (4, 1)
    assert _rollout_parallelism(_config(max_concurrent=8, workers=2)) == (2, 4)


@pytest.mark.parametrize(
    ("max_concurrent", "workers"),
    [(0, 8), (8, 0)],
)
def test_parallelism_rejects_nonpositive_values(max_concurrent, workers):
    with pytest.raises(ValueError):
        _rollout_parallelism(
            _config(max_concurrent=max_concurrent, workers=workers)
        )
