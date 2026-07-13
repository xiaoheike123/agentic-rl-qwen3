import json
from pathlib import Path

from agent_rl.data.synthetic.sampler import (
    balanced_records,
    build_balanced_verl_dataset,
)
from agent_rl.data.synthetic.schema import (
    GenerationMetadata,
    SyntheticSplit,
    SyntheticTaskRecord,
    VerificationMetadata,
)
from agent_rl.data.synthetic.storage import write_records


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


def test_verl_export_carries_training_database_provenance(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    fingerprint = "f" * 64
    for domain in ("airline", "retail", "telecom"):
        write_records(
            corpus / domain / "train.jsonl",
            [_record(domain, 1)],
        )
    (corpus / "manifest.json").write_text(
        json.dumps(
            {"config": {"training_database_fingerprint": fingerprint}}
        ),
        encoding="utf-8",
    )
    output = tmp_path / "train.jsonl"

    count = build_balanced_verl_dataset(
        output,
        corpus_root=corpus,
        split=SyntheticSplit.TRAIN,
        seed=42,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert count == 3
    assert all(
        row["extra_info"]["database_source"]
        == "pseudonymized_training"
        for row in rows
    )
    assert all(
        row["extra_info"]["training_database_fingerprint"]
        == fingerprint
        for row in rows
    )
