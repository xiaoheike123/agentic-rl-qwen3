"""Shared task construction helpers for procedural domain generators."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from tau2.data_model.tasks import (
    Action,
    Description,
    EvaluationCriteria,
    RewardType,
    StructuredUserInstructions,
    Task,
    UserScenario,
)

from agent_rl.data.synthetic.schema import GenerationMetadata


GENERATOR_VERSION = "2.0.0"


@dataclass(frozen=True, slots=True)
class OracleActionSpec:
    """One ordered, environment-executable action in a synthetic task target."""

    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Oracle action name must not be empty")
        if not isinstance(self.arguments, dict):
            raise TypeError("Oracle action arguments must be a dictionary")


@dataclass(frozen=True, slots=True)
class GeneratedCandidate:
    domain: str
    task: Task
    generation: GenerationMetadata


def make_candidate(
    *,
    domain: str,
    template: str,
    seed: int,
    entities: tuple[str, ...],
    reason_for_call: str,
    known_info: str,
    task_instructions: str,
    actions: tuple[OracleActionSpec, ...],
    communicate_info: list[str],
    purpose: str,
) -> GeneratedCandidate:
    if not actions:
        raise ValueError("A synthetic task requires at least one Oracle action")

    identity = json.dumps(
        {
            "domain": domain,
            "template": template,
            "seed": seed,
            "entities": entities,
            "actions": [
                {"name": action.name, "arguments": action.arguments}
                for action in actions
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    task_id = f"synthetic-{domain}-{suffix}"
    task = Task(
        id=task_id,
        description=Description(
            purpose=purpose,
            relevant_policies=None,
            notes="Procedurally generated; no official task text was used.",
        ),
        user_scenario=UserScenario(
            persona=None,
            instructions=StructuredUserInstructions(
                domain=domain,
                reason_for_call=reason_for_call,
                known_info=known_info,
                unknown_info=None,
                task_instructions=task_instructions,
            ),
        ),
        ticket=None,
        initial_state=None,
        evaluation_criteria=EvaluationCriteria(
            actions=[
                Action(
                    action_id=f"oracle_{action.name}_{index}",
                    requestor="assistant",
                    name=action.name,
                    arguments=action.arguments,
                )
                for index, action in enumerate(actions)
            ],
            communicate_info=communicate_info,
            reward_basis=[RewardType.DB, RewardType.COMMUNICATE],
        ),
        issues=None,
        required_documents=None,
        user_tools=None,
    )
    return GeneratedCandidate(
        domain=domain,
        task=task,
        generation=GenerationMetadata(
            generator=f"agent_rl.synthetic.{domain}",
            generator_version=GENERATOR_VERSION,
            seed=seed,
            template=template,
            source_entities=entities,
        ),
    )
