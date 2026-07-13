from __future__ import annotations

from collections import Counter

from agent_rl.data.synthetic.generators.retail import generate_retail_candidates
from agent_rl.data.synthetic.policy_validation import validate_candidate_policy


def test_retail_generator_has_balanced_long_horizon_families() -> None:
    candidates = generate_retail_candidates(42)
    templates = Counter(item.generation.template for item in candidates)
    action_counts = Counter(
        len(item.task.evaluation_criteria.actions or []) for item in candidates
    )
    difficulties = Counter(
        item.metadata["policy"]["difficulty"] for item in candidates
    )

    assert len(templates) >= 16
    assert max(templates.values()) / len(candidates) < 0.07
    assert action_counts[2] / len(candidates) >= 0.35
    assert (difficulties["complex"] + difficulties["hard"]) / len(candidates) >= 0.65


def test_all_retail_candidates_pass_independent_policy_validation() -> None:
    candidates = generate_retail_candidates(42)

    for candidate in candidates:
        validate_candidate_policy(candidate)


def test_retail_private_policy_metadata_is_not_embedded_in_tau_task() -> None:
    candidate = generate_retail_candidates(42)[0]
    task_payload = candidate.task.model_dump(mode="json")

    assert "policy" not in task_payload
    assert "required_reads" not in task_payload
    assert "expected_writes" not in task_payload
