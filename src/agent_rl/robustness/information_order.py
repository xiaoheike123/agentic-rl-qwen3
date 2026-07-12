"""Pre-authored information-order perturbations for tau2 user scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class InformationOrderVariant:
    """A meaning-preserving instruction rendering with reordered facts."""

    task_id: str
    ordered_instructions: str

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if not self.ordered_instructions.strip():
            raise ValueError("ordered_instructions must not be empty")


def make_information_order_transform(variant: InformationOrderVariant):
    """Replace only user-simulator instructions; keep task answers unchanged."""

    def transform(task: Any) -> Any:
        if getattr(task, "id", None) != variant.task_id:
            raise ValueError(
                f"information-order variant for {variant.task_id!r} cannot "
                f"transform task {getattr(task, 'id', None)!r}"
            )

        task.user_scenario.instructions = variant.ordered_instructions
        return task

    return transform
