from tau2.data_model.tasks import (
    Action,
    EvaluationCriteria,
    RewardType,
    Task,
    UserScenario,
)

from agent_rl.data.synthetic.fingerprint import exact_fingerprint
from agent_rl.data.synthetic.schema import (
    GenerationMetadata,
    SyntheticSplit,
    SyntheticTaskRecord,
    VerificationMetadata,
)


def _task(task_id: str = "synthetic-airline-abc") -> Task:
    return Task(
        id=task_id,
        user_scenario=UserScenario(instructions="Cancel reservation SYN123."),
        evaluation_criteria=EvaluationCriteria(
            actions=[
                Action(
                    action_id="cancel_0",
                    name="cancel_reservation",
                    arguments={"reservation_id": "SYN123"},
                )
            ],
            communicate_info=["SYN123", "cancelled"],
            reward_basis=[RewardType.DB, RewardType.COMMUNICATE],
        ),
    )


def test_synthetic_record_round_trip() -> None:
    task = _task()
    record = SyntheticTaskRecord(
        domain="airline",
        split=SyntheticSplit.TRAIN,
        task=task.model_dump(mode="json"),
        semantic_fingerprint=exact_fingerprint(task),
        generation=GenerationMetadata(
            generator="test",
            generator_version="1",
            seed=1,
            template="cancel",
            source_entities=("user", "SYN123"),
        ),
        verification=VerificationMetadata(
            oracle_verified=True,
            database_changed=True,
            action_count=1,
            initial_db_hash="before",
            target_db_hash="after",
        ),
    )

    restored = SyntheticTaskRecord.from_dict(record.to_dict())

    assert restored == record
    assert restored.to_tau_task() == task
