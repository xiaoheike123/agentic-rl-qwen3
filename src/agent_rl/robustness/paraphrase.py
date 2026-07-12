"""Auditable, pre-authored user-request paraphrase perturbations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ParaphraseVariant:
    """Replacement text reviewed to preserve one task's original intent."""

    task_id: str
    instructions: str
    reason_for_call: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if not self.instructions.strip():
            raise ValueError("instructions must not be empty")


def make_paraphrase_transform(variant: ParaphraseVariant):
    """Build a tau2 task transform without touching evaluation criteria."""

    def transform(task: Any) -> Any:
        if getattr(task, "id", None) != variant.task_id:
            raise ValueError(
                f"paraphrase for {variant.task_id!r} cannot transform "
                f"task {getattr(task, 'id', None)!r}"
            )

        scenario = task.user_scenario
        instructions = scenario.instructions

        if isinstance(instructions, str):
            scenario.instructions = variant.instructions
        else:
            instructions.task_instructions = variant.instructions
            if variant.reason_for_call is not None:
                instructions.reason_for_call = variant.reason_for_call

        return task

    return transform
