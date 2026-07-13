from __future__ import annotations

from collections import Counter

from agent_rl.data.synthetic.generators.telecom import generate_telecom_candidates
from agent_rl.data.synthetic.policy_validation import validate_candidate_policy


def test_telecom_generator_has_real_support_and_multi_line_diversity() -> None:
    candidates = generate_telecom_candidates(2_000_049)
    templates = Counter(item.generation.template for item in candidates)
    action_counts = Counter(
        len(item.task.evaluation_criteria.actions or []) for item in candidates
    )
    difficulties = Counter(
        item.metadata["policy"]["difficulty"] for item in candidates
    )

    assert len(templates) >= 25
    assert max(templates.values()) / len(candidates) < 0.20
    assert sum(count for size, count in action_counts.items() if size > 1) / len(candidates) >= 0.54
    assert (difficulties["complex"] + difficulties["hard"]) / len(candidates) >= 0.74


def test_all_telecom_candidates_pass_independent_policy_validation() -> None:
    candidates = generate_telecom_candidates(2_000_049)

    for candidate in candidates:
        validate_candidate_policy(candidate)


def test_telecom_support_targets_user_actions_without_leaking_policy_metadata() -> None:
    candidate = next(
        item
        for item in generate_telecom_candidates(2_000_049)
        if item.generation.template == "support_airplane_mode_on"
    )
    actions = candidate.task.evaluation_criteria.actions or []
    task_payload = candidate.task.model_dump(mode="json")

    assert actions[0].requestor == "user"
    assert candidate.task.initial_state is not None
    assert "policy" not in task_payload
    assert "required_reads" not in task_payload
