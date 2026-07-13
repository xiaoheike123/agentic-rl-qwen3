"""Environment-backed verification for generated oracle trajectories."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any

from tau2.data_model.tasks import Task
from tau2.registry import registry

from agent_rl.data.synthetic.schema import VerificationMetadata


def _environment_hash(environment) -> str:
    """Hash both assistant-side and user-side state when available."""

    agent_hash = environment.get_db_hash() or ""
    user_hash = environment.get_user_db_hash() or ""
    return hashlib.sha256(f"{agent_hash}:{user_hash}".encode("utf-8")).hexdigest()


def verify_oracle_task(
    domain: str,
    task: Task,
    database: Any | None = None,
) -> VerificationMetadata:
    criteria = task.evaluation_criteria
    actions = criteria.actions if criteria and criteria.actions else []
    if not actions:
        return VerificationMetadata(
            oracle_verified=False,
            database_changed=False,
            action_count=0,
            error="task has no oracle actions",
        )

    constructor_kwargs: dict[str, Any] = {"solo_mode": False}
    if database is not None:
        constructor_kwargs["db"] = deepcopy(database)
    environment = registry.get_env_constructor(domain)(**constructor_kwargs)
    initial_state = task.initial_state
    environment.set_state(
        initialization_data=(
            initial_state.initialization_data if initial_state is not None else None
        ),
        initialization_actions=(
            initial_state.initialization_actions if initial_state is not None else None
        ),
        message_history=(
            list(initial_state.message_history or [])
            if initial_state is not None
            else []
        ),
    )
    initial_hash = _environment_hash(environment)
    try:
        for action in actions:
            environment.make_tool_call(
                tool_name=action.name,
                requestor=action.requestor,
                **action.arguments,
            )
            environment.sync_tools()
    except Exception as error:
        return VerificationMetadata(
            oracle_verified=False,
            database_changed=False,
            action_count=len(actions),
            initial_db_hash=initial_hash,
            target_db_hash=_environment_hash(environment),
            error=f"{type(error).__name__}: {error}",
        )

    target_hash = _environment_hash(environment)
    changed = initial_hash != target_hash
    return VerificationMetadata(
        oracle_verified=changed,
        database_changed=changed,
        action_count=len(actions),
        initial_db_hash=initial_hash,
        target_db_hash=target_hash,
        error=None if changed else "oracle actions did not change the database",
    )
