"""Deterministic task fingerprints used for deduplication and overlap audits."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any

from tau2.data_model.tasks import Task


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def task_payload(task: Task) -> dict[str, Any]:
    """Return behavior-defining task fields without IDs or author notes."""

    value = task.model_dump(mode="json")
    return {
        "user_scenario": value.get("user_scenario"),
        "initial_state": value.get("initial_state"),
        "evaluation_criteria": value.get("evaluation_criteria"),
        "user_tools": value.get("user_tools"),
    }


def exact_fingerprint(task: Task) -> str:
    payload = canonical_json(task_payload(task)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def semantic_fingerprint(domain: str, task: Task) -> str:
    payload = {
        "domain": domain,
        "actions": action_signature(task, include_arguments=True),
        "tokens": sorted(task_tokens(task)),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def action_signature(
    task: Task,
    *,
    include_arguments: bool,
) -> tuple[str, ...]:
    criteria = task.evaluation_criteria
    actions = criteria.actions if criteria and criteria.actions else []
    if include_arguments:
        return tuple(
            f"{action.requestor}:{action.name}:{canonical_json(action.arguments)}"
            for action in actions
        )
    return tuple(f"{action.requestor}:{action.name}" for action in actions)


def task_tokens(task: Task) -> frozenset[str]:
    scenario = task.user_scenario.model_dump(mode="json")
    criteria = (
        task.evaluation_criteria.model_dump(mode="json")
        if task.evaluation_criteria is not None
        else {}
    )
    text = canonical_json({"scenario": scenario, "criteria": criteria}).lower()
    return frozenset(TOKEN_PATTERN.findall(text))


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0
