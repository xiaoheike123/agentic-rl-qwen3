"""Environment-backed verification for generated oracle trajectories."""

from __future__ import annotations

from tau2.data_model.tasks import Task
from tau2.registry import registry

from agent_rl.data.synthetic.schema import VerificationMetadata


def verify_oracle_task(domain: str, task: Task) -> VerificationMetadata:
    criteria = task.evaluation_criteria
    actions = criteria.actions if criteria and criteria.actions else []
    if not actions:
        return VerificationMetadata(
            oracle_verified=False,
            database_changed=False,
            action_count=0,
            error="task has no oracle actions",
        )

    environment = registry.get_env_constructor(domain)(solo_mode=False)
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
    initial_hash = environment.get_db_hash()
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
            target_db_hash=environment.get_db_hash(),
            error=f"{type(error).__name__}: {error}",
        )

    target_hash = environment.get_db_hash()
    changed = initial_hash != target_hash
    return VerificationMetadata(
        oracle_verified=changed,
        database_changed=changed,
        action_count=len(actions),
        initial_db_hash=initial_hash,
        target_db_hash=target_hash,
        error=None if changed else "oracle actions did not change the database",
    )
