from agent_rl.data.synthetic.sampler import balanced_records
from agent_rl.data.synthetic.schema import (
    GenerationMetadata,
    OverlapMetadata,
    SyntheticSplit,
    SyntheticTaskRecord,
    VerificationMetadata,
)


def _record(domain: str, index: int) -> SyntheticTaskRecord:
    task_id = f"synthetic-{domain}-{index}"
    return SyntheticTaskRecord(
        domain=domain,
        split=SyntheticSplit.TRAIN,
        task={
            "id": task_id,
            "user_scenario": {"instructions": "Perform the requested update."},
            "evaluation_criteria": {
                "actions": [
                    {
                        "action_id": "write_0",
                        "requestor": "assistant",
                        "name": "write_tool",
                        "arguments": {"value": index},
                    }
                ],
                "communicate_info": [],
                "reward_basis": ["DB", "COMMUNICATE"],
            },
        },
        semantic_fingerprint=f"fp-{domain}-{index}",
        generation=GenerationMetadata(
            generator="test",
            generator_version="1",
            seed=index,
            template="write",
            source_entities=(f"entity-{index}",),
        ),
        verification=VerificationMetadata(True, True, 1, "a", "b"),
        overlap=OverlapMetadata(True, False, False, None, 0.0, 0.82),
    )


def test_balanced_records_use_equal_domain_counts() -> None:
    records = [
        *[_record("airline", index) for index in range(5)],
        *[_record("retail", index) for index in range(4)],
        *[_record("telecom", index) for index in range(3)],
    ]

    selected = balanced_records(
        records,
        split=SyntheticSplit.TRAIN,
        seed=42,
    )

    assert len(selected) == 9
    assert {domain: [item.domain for item in selected].count(domain) for domain in {
        "airline", "retail", "telecom"
    }} == {"airline": 3, "retail": 3, "telecom": 3}
