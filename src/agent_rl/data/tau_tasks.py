"""Explicit loaders for untouched official tau2 evaluation splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from tau2.registry import registry


@dataclass(frozen=True, slots=True)
class TauTaskRef:
    domain: str
    task_id: str
    split: str


def load_tau_task_refs(
    domain: str,
    *,
    split: str = "base",
    task_ids: Iterable[str] | None = None,
) -> tuple[TauTaskRef, ...]:
    """Load official IDs without inventing a project-specific train/test split."""

    if not domain.strip():
        raise ValueError("domain must not be empty")
    available = registry.get_task_splits_loader(domain)()
    if split not in available:
        raise ValueError(
            f"unknown official split {split!r} for {domain!r}; "
            f"available={sorted(available)}"
        )

    requested = set(task_ids) if task_ids is not None else None
    tasks = registry.get_tasks_loader(domain)(task_split_name=split)
    known = {task.id for task in tasks}
    if requested is not None:
        missing = requested - known
        if missing:
            raise ValueError(f"unknown task IDs in {domain}/{split}: {sorted(missing)}")
        known &= requested
    return tuple(
        TauTaskRef(domain=domain, task_id=task_id, split=split)
        for task_id in sorted(known)
    )
