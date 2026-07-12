"""Load public tau2 task identifiers without exposing hidden task answers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from tau2.registry import registry


@dataclass(frozen=True, slots=True)
class TauTaskRef:
    domain: str
    task_id: str
    split: str


def stable_task_split(task_id: str) -> str:
    """Deterministic 80/10/10 split based only on the public task ID."""

    bucket = int(hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "validation"
    return "test"


def load_tau_task_refs(
    domain: str,
    *,
    split: str = "all",
    task_ids: Iterable[str] | None = None,
) -> tuple[TauTaskRef, ...]:
    if not domain.strip():
        raise ValueError("domain must not be empty")
    if split not in {"all", "train", "validation", "test"}:
        raise ValueError("split must be all, train, validation, or test")

    requested = set(task_ids) if task_ids is not None else None
    tasks = registry.get_tasks_loader(domain)()
    refs = []
    found: set[str] = set()
    for task in tasks:
        task_split = stable_task_split(task.id)
        if requested is not None and task.id not in requested:
            continue
        if split != "all" and task_split != split:
            continue
        refs.append(TauTaskRef(domain=domain, task_id=task.id, split=task_split))
        found.add(task.id)

    if requested is not None:
        missing = requested - found
        if missing:
            raise ValueError(f"unknown or excluded task IDs: {sorted(missing)}")
    if not refs:
        raise ValueError(f"no tasks found for domain={domain!r}, split={split!r}")
    return tuple(sorted(refs, key=lambda item: item.task_id))
