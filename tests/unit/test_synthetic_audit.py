from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from agent_rl.data.synthetic.audit import (
    AuditStatus,
    SyntheticAuditConfig,
    _near_duplicates,
    audit_synthetic_corpus,
)
from agent_rl.data.synthetic.fingerprint import semantic_fingerprint
from agent_rl.data.synthetic.generators.common import (
    OracleActionSpec,
    make_candidate,
)
from agent_rl.data.synthetic.schema import (
    SyntheticSplit,
    SyntheticTaskRecord,
    VerificationMetadata,
)
from agent_rl.data.synthetic.storage import write_records


def _record(
    *,
    index: int,
    split: SyntheticSplit,
    template: str,
    entity: str,
) -> SyntheticTaskRecord:
    candidate = make_candidate(
        domain="airline",
        template=template,
        seed=index,
        entities=(entity, f"reservation-{index}"),
        reason_for_call=f"Handle reservation {index}.",
        known_info=f"The reservation is reservation-{index}.",
        task_instructions=f"Request operation {index} and confirm it.",
        actions=(
            OracleActionSpec(
                name="cancel_reservation",
                arguments={"reservation_id": f"reservation-{index}"},
            ),
        ),
        communicate_info=[f"reservation-{index}", "cancelled"],
        purpose="Test one synthetic audit record.",
    )
    return SyntheticTaskRecord(
        domain="airline",
        split=split,
        task=candidate.task.model_dump(mode="json"),
        semantic_fingerprint=semantic_fingerprint("airline", candidate.task),
        generation=candidate.generation,
        verification=VerificationMetadata(
            oracle_verified=True,
            database_changed=True,
            action_count=1,
            initial_db_hash=f"before-{index}",
            target_db_hash=f"after-{index}",
        ),
        metadata={
            "policy": {"difficulty": "simple"},
            "policy_validation": {"passed": True},
        },
    )


def _write_corpus(root: Path, records: list[SyntheticTaskRecord]) -> None:
    for split in SyntheticSplit:
        selected = [record for record in records if record.split is split]
        write_records(root / "airline" / f"{split.value}.jsonl", selected)
    manifest = {
        "domains": {
            "airline": {
                "accepted_train": sum(
                    record.split is SyntheticSplit.TRAIN for record in records
                ),
                "accepted_validation": sum(
                    record.split is SyntheticSplit.VALIDATION for record in records
                ),
            }
        }
    }
    with (root / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream)


def test_audit_rejects_corpus_without_long_horizon_coverage(
    tmp_path: Path,
) -> None:
    records = [
        _record(
            index=index,
            split=(SyntheticSplit.TRAIN if index < 2 else SyntheticSplit.VALIDATION),
            template=f"template-{index % 2}",
            entity=f"entity-{index}",
        )
        for index in range(4)
    ]
    _write_corpus(tmp_path, records)

    report = audit_synthetic_corpus(
        SyntheticAuditConfig(
            corpus_root=tmp_path,
            domains=("airline",),
            min_split_size=2,
            min_templates_per_domain=2,
            max_near_duplicate_record_fraction=1.0,
        )
    )

    assert report.status is AuditStatus.FAIL
    assert report.domains["airline"].entity_leakage == []
    failed_codes = {
        finding.code
        for finding in report.findings
        if finding.status is AuditStatus.FAIL
    }
    assert "multi_action_coverage" in failed_codes
    assert "difficulty_coverage" in failed_codes


def test_audit_fails_entity_leakage_and_template_concentration(
    tmp_path: Path,
) -> None:
    records = [
        _record(
            index=index,
            split=(SyntheticSplit.TRAIN if index < 2 else SyntheticSplit.VALIDATION),
            template="one-template",
            entity="shared-entity" if index in {0, 2} else f"entity-{index}",
        )
        for index in range(4)
    ]
    _write_corpus(tmp_path, records)

    report = audit_synthetic_corpus(
        SyntheticAuditConfig(
            corpus_root=tmp_path,
            domains=("airline",),
            min_split_size=2,
            max_near_duplicate_record_fraction=1.0,
        )
    )

    assert report.status is AuditStatus.FAIL
    failed_codes = {
        finding.code
        for finding in report.findings
        if finding.status is AuditStatus.FAIL
    }
    assert "entity_isolation" in failed_codes
    assert "template_coverage" in failed_codes
    assert "template_concentration" in failed_codes


def test_near_duplicate_gate_ignores_same_template_across_entities() -> None:
    left = _record(
        index=1,
        split=SyntheticSplit.TRAIN,
        template="shared-template",
        entity="entity-1",
    )
    right = _record(
        index=2,
        split=SyntheticSplit.VALIDATION,
        template="shared-template",
        entity="entity-2",
    )

    pair_count, record_fraction, examples = _near_duplicates(
        [left, right],
        threshold=0.0,
    )

    assert pair_count == 0
    assert record_fraction == 0.0
    assert examples == []


def test_near_duplicate_gate_ignores_shared_support_component() -> None:
    left = _record(
        index=3,
        split=SyntheticSplit.TRAIN,
        template="single-fault",
        entity="entity-3",
    )
    right = _record(
        index=4,
        split=SyntheticSplit.VALIDATION,
        template="composed-fault",
        entity="entity-4",
    )
    left = replace(
        left,
        metadata={
            **left.metadata,
            "policy": {
                **left.metadata["policy"],
                "support_cases": ["broken_apn"],
            },
        },
    )
    right = replace(
        right,
        metadata={
            **right.metadata,
            "policy": {
                **right.metadata["policy"],
                "support_cases": ["broken_apn", "slow_vpn"],
            },
        },
    )

    pair_count, record_fraction, _ = _near_duplicates(
        [left, right],
        threshold=0.0,
    )

    assert pair_count == 0
    assert record_fraction == 0.0


def test_near_duplicate_gate_still_compares_unrelated_templates() -> None:
    left = _record(
        index=5,
        split=SyntheticSplit.TRAIN,
        template="template-a",
        entity="entity-5",
    )
    right = _record(
        index=6,
        split=SyntheticSplit.VALIDATION,
        template="template-b",
        entity="entity-6",
    )

    pair_count, record_fraction, examples = _near_duplicates(
        [left, right],
        threshold=0.0,
    )

    assert pair_count == 1
    assert record_fraction == 1.0
    assert len(examples) == 1
