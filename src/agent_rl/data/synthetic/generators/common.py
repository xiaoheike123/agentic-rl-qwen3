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


GENERATOR_VERSION = "1.0.0"


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
    action_name: str,
    action_arguments: dict[str, Any],
    communicate_info: list[str],
    purpose: str,
) -> GeneratedCandidate:
    identity = json.dumps(
        {
            "domain": domain,
            "template": template,
            "seed": seed,
            "entities": entities,
            "action": action_name,
            "arguments": action_arguments,
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
                    action_id=f"oracle_{action_name}_0",
                    requestor="assistant",
                    name=action_name,
                    arguments=action_arguments,
                )
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
