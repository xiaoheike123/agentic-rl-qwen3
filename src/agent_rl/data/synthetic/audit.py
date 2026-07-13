"""Audit a generated synthetic corpus before it is admitted to RL training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from agent_rl.data.synthetic.fingerprint import jaccard, task_tokens
from agent_rl.data.synthetic.schema import (
    SUPPORTED_DOMAINS,
    SyntheticSplit,
    SyntheticTaskRecord,
)
from agent_rl.data.synthetic.storage import load_records


class AuditStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


STATUS_RANK = {
    AuditStatus.PASS: 0,
    AuditStatus.WARN: 1,
    AuditStatus.FAIL: 2,
}


@dataclass(frozen=True, slots=True)
class SyntheticAuditConfig:
    corpus_root: Path
    domains: tuple[str, ...] = ("airline", "retail", "telecom")
    min_split_size: int = 20
    min_templates_per_domain: int = 6
    max_dominant_template_fraction: float = 0.35
    max_parameter_variant_fraction: float = 0.50
    min_multi_action_fraction: float = 0.35
    min_complex_or_hard_fraction: float = 0.30
    near_duplicate_threshold: float = 0.92
    max_near_duplicate_record_fraction: float = 0.20
    human_sample_per_domain: int = 20

    def __post_init__(self) -> None:
        invalid = set(self.domains) - SUPPORTED_DOMAINS
        if invalid:
            raise ValueError(f"unsupported domains: {sorted(invalid)}")
        if not self.domains:
            raise ValueError("at least one domain is required")
        if self.min_split_size <= 0:
            raise ValueError("min_split_size must be positive")
        if self.min_templates_per_domain <= 0:
            raise ValueError("min_templates_per_domain must be positive")
        for name, value in (
            (
                "max_dominant_template_fraction",
                self.max_dominant_template_fraction,
            ),
            ("max_parameter_variant_fraction", self.max_parameter_variant_fraction),
            ("near_duplicate_threshold", self.near_duplicate_threshold),
            ("min_multi_action_fraction", self.min_multi_action_fraction),
            ("min_complex_or_hard_fraction", self.min_complex_or_hard_fraction),
            (
                "max_near_duplicate_record_fraction",
                self.max_near_duplicate_record_fraction,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.human_sample_per_domain <= 0:
            raise ValueError("human_sample_per_domain must be positive")


@dataclass(frozen=True, slots=True)
class AuditFinding:
    code: str
    status: AuditStatus
    message: str
    domain: str | None = None
    value: Any = None
    threshold: Any = None


@dataclass(slots=True)
class DomainAudit:
    train_count: int
    validation_count: int
    template_counts: dict[str, int]
    action_counts: dict[str, int]
    action_count_distribution: dict[str, int]
    difficulty_counts: dict[str, int]
    dominant_template_fraction: float
    normalized_template_entropy: float
    parameter_variant_record_fraction: float
    max_variants_per_entity_template: int
    multi_action_fraction: float
    complex_or_hard_fraction: float
    policy_metadata_coverage: float
    communication_coverage: float
    verified_fraction: float
    train_entity_count: int
    validation_entity_count: int
    entity_leakage: list[str]
    duplicate_fingerprints: list[str]
    near_duplicate_pair_count: int
    near_duplicate_record_fraction: float
    near_duplicate_examples: list[dict[str, Any]] = field(default_factory=list)
    human_sample_task_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SyntheticAuditReport:
    status: AuditStatus
    generated_at: str
    corpus_root: str
    config: dict[str, Any]
    domains: dict[str, DomainAudit]
    findings: list[AuditFinding]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "generated_at": self.generated_at,
            "corpus_root": self.corpus_root,
            "config": self.config,
            "domains": {
                domain: asdict(audit) for domain, audit in self.domains.items()
            },
            "findings": [
                {**asdict(finding), "status": finding.status.value}
                for finding in self.findings
            ],
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _overall_status(findings: Iterable[AuditFinding]) -> AuditStatus:
    return max(
        (finding.status for finding in findings),
        key=STATUS_RANK.__getitem__,
        default=AuditStatus.PASS,
    )


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _normalized_entropy(counts: Counter[str]) -> float:
    if len(counts) <= 1:
        return 0.0
    total = sum(counts.values())
    entropy = -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
        if count
    )
    return entropy / math.log(len(counts))


def _deterministic_sample(
    records: list[SyntheticTaskRecord],
    *,
    count: int,
) -> list[str]:
    ranked = sorted(
        records,
        key=lambda record: hashlib.sha256(record.task_id.encode("utf-8")).hexdigest(),
    )
    return [record.task_id for record in ranked[:count]]


def _near_duplicates(
    records: list[SyntheticTaskRecord],
    *,
    threshold: float,
) -> tuple[int, float, list[dict[str, Any]]]:
    token_sets = [task_tokens(record.to_tau_task()) for record in records]
    involved: set[str] = set()
    examples: list[dict[str, Any]] = []
    pair_count = 0
    for left_index, left in enumerate(records):
        for right_index in range(left_index + 1, len(records)):
            right = records[right_index]
            # Reusing one task family across isolated entities is intentional.
            # Same-entity variants and split leakage have separate quality gates.
            if left.generation.template == right.generation.template:
                continue
            left_policy = left.metadata.get("policy", {})
            right_policy = right.metadata.get("policy", {})
            left_cases = set(left_policy.get("support_cases", []))
            right_cases = set(right_policy.get("support_cases", []))
            # Single-fault and composed-fault tasks deliberately share a
            # curriculum component; unrelated support cases remain compared.
            if left_cases and right_cases and left_cases & right_cases:
                continue
            similarity = jaccard(token_sets[left_index], token_sets[right_index])
            if similarity < threshold:
                continue
            pair_count += 1
            involved.update((left.task_id, right.task_id))
            if len(examples) < 20:
                examples.append(
                    {
                        "left_task_id": left.task_id,
                        "right_task_id": right.task_id,
                        "similarity": round(similarity, 6),
                        "left_template": left.generation.template,
                        "right_template": right.generation.template,
                    }
                )
    return pair_count, _fraction(len(involved), len(records)), examples


def _audit_domain(
    domain: str,
    records: list[SyntheticTaskRecord],
    config: SyntheticAuditConfig,
) -> DomainAudit:
    by_split = {
        split: [record for record in records if record.split is split]
        for split in SyntheticSplit
    }
    templates = Counter(record.generation.template for record in records)
    actions: Counter[str] = Counter()
    action_lengths: Counter[str] = Counter()
    communication_present = 0
    multi_action = 0
    policy_metadata_present = 0
    difficulties: Counter[str] = Counter()
    variant_groups: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    fingerprint_counts = Counter(record.semantic_fingerprint for record in records)

    for record in records:
        task = record.to_tau_task()
        criteria = task.evaluation_criteria
        oracle_actions = list(criteria.actions or []) if criteria else []
        action_lengths[str(len(oracle_actions))] += 1
        actions.update(action.name for action in oracle_actions)
        multi_action += int(len(oracle_actions) > 1)
        policy = record.metadata.get("policy")
        validation = record.metadata.get("policy_validation")
        if isinstance(policy, dict):
            difficulties[str(policy.get("difficulty", "missing"))] += 1
        policy_metadata_present += int(
            isinstance(policy, dict)
            and isinstance(validation, dict)
            and validation.get("passed") is True
        )
        communication_present += int(bool(criteria and criteria.communicate_info))
        variant_groups[
            (record.generation.template, record.generation.source_entities)
        ].append(record.task_id)

    variant_records = sum(
        len(task_ids) for task_ids in variant_groups.values() if len(task_ids) > 1
    )
    max_variants = max((len(task_ids) for task_ids in variant_groups.values()), default=0)
    train_entities = {
        entity
        for record in by_split[SyntheticSplit.TRAIN]
        for entity in record.generation.source_entities
    }
    validation_entities = {
        entity
        for record in by_split[SyntheticSplit.VALIDATION]
        for entity in record.generation.source_entities
    }
    near_pairs, near_fraction, near_examples = _near_duplicates(
        records,
        threshold=config.near_duplicate_threshold,
    )
    total = len(records)
    return DomainAudit(
        train_count=len(by_split[SyntheticSplit.TRAIN]),
        validation_count=len(by_split[SyntheticSplit.VALIDATION]),
        template_counts=dict(sorted(templates.items())),
        action_counts=dict(sorted(actions.items())),
        action_count_distribution=dict(sorted(action_lengths.items())),
        difficulty_counts=dict(sorted(difficulties.items())),
        dominant_template_fraction=(
            max(templates.values()) / total if templates and total else 0.0
        ),
        normalized_template_entropy=_normalized_entropy(templates),
        parameter_variant_record_fraction=_fraction(variant_records, total),
        max_variants_per_entity_template=max_variants,
        multi_action_fraction=_fraction(multi_action, total),
        complex_or_hard_fraction=_fraction(
            difficulties["complex"] + difficulties["hard"], total
        ),
        policy_metadata_coverage=_fraction(policy_metadata_present, total),
        communication_coverage=_fraction(communication_present, total),
        verified_fraction=_fraction(
            sum(record.verification.oracle_verified for record in records), total
        ),
        train_entity_count=len(train_entities),
        validation_entity_count=len(validation_entities),
        entity_leakage=sorted(train_entities & validation_entities),
        duplicate_fingerprints=sorted(
            fingerprint
            for fingerprint, count in fingerprint_counts.items()
            if count > 1
        ),
        near_duplicate_pair_count=near_pairs,
        near_duplicate_record_fraction=near_fraction,
        near_duplicate_examples=near_examples,
        human_sample_task_ids=_deterministic_sample(
            records,
            count=min(config.human_sample_per_domain, total),
        ),
    )


def _domain_findings(
    domain: str,
    audit: DomainAudit,
    config: SyntheticAuditConfig,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []

    def add(
        code: str,
        status: AuditStatus,
        message: str,
        value: Any,
        threshold: Any,
    ) -> None:
        findings.append(
            AuditFinding(code, status, message, domain, value, threshold)
        )

    for split, count in (
        ("train", audit.train_count),
        ("validation", audit.validation_count),
    ):
        add(
            f"{split}_size",
            AuditStatus.PASS if count >= config.min_split_size else AuditStatus.FAIL,
            f"{split} split size",
            count,
            f">={config.min_split_size}",
        )
    add(
        "template_coverage",
        (
            AuditStatus.PASS
            if len(audit.template_counts) >= config.min_templates_per_domain
            else AuditStatus.FAIL
        ),
        "distinct task templates",
        len(audit.template_counts),
        f">={config.min_templates_per_domain}",
    )
    add(
        "template_concentration",
        (
            AuditStatus.PASS
            if audit.dominant_template_fraction
            <= config.max_dominant_template_fraction
            else AuditStatus.FAIL
        ),
        "fraction occupied by the dominant template",
        round(audit.dominant_template_fraction, 6),
        f"<={config.max_dominant_template_fraction}",
    )
    add(
        "parameter_variants",
        (
            AuditStatus.PASS
            if audit.parameter_variant_record_fraction
            <= config.max_parameter_variant_fraction
            else AuditStatus.FAIL
        ),
        "records created as variants of the same template and entity group",
        round(audit.parameter_variant_record_fraction, 6),
        f"<={config.max_parameter_variant_fraction}",
    )
    add(
        "near_duplicates",
        (
            AuditStatus.PASS
            if audit.near_duplicate_record_fraction
            <= config.max_near_duplicate_record_fraction
            else AuditStatus.FAIL
        ),
        "records involved in a high-similarity corpus pair",
        round(audit.near_duplicate_record_fraction, 6),
        f"<={config.max_near_duplicate_record_fraction}",
    )
    add(
        "entity_isolation",
        AuditStatus.PASS if not audit.entity_leakage else AuditStatus.FAIL,
        "entities shared by train and validation",
        len(audit.entity_leakage),
        "0",
    )
    add(
        "fingerprint_uniqueness",
        AuditStatus.PASS if not audit.duplicate_fingerprints else AuditStatus.FAIL,
        "duplicate semantic fingerprints",
        len(audit.duplicate_fingerprints),
        "0",
    )
    add(
        "oracle_verification",
        AuditStatus.PASS if audit.verified_fraction == 1.0 else AuditStatus.FAIL,
        "Oracle-verified record fraction",
        round(audit.verified_fraction, 6),
        "1.0",
    )
    add(
        "policy_metadata",
        AuditStatus.PASS if audit.policy_metadata_coverage == 1.0 else AuditStatus.FAIL,
        "records with independently validated private policy metadata",
        round(audit.policy_metadata_coverage, 6),
        "1.0",
    )
    add(
        "communication_coverage",
        AuditStatus.PASS if audit.communication_coverage == 1.0 else AuditStatus.FAIL,
        "records with an explicit communication requirement",
        round(audit.communication_coverage, 6),
        "1.0",
    )
    add(
        "multi_action_coverage",
        (
            AuditStatus.PASS
            if audit.multi_action_fraction >= config.min_multi_action_fraction
            else AuditStatus.FAIL
        ),
        "records requiring more than one Oracle action",
        round(audit.multi_action_fraction, 6),
        f">={config.min_multi_action_fraction}",
    )
    add(
        "difficulty_coverage",
        (
            AuditStatus.PASS
            if audit.complex_or_hard_fraction
            >= config.min_complex_or_hard_fraction
            else AuditStatus.FAIL
        ),
        "records labeled complex or hard",
        round(audit.complex_or_hard_fraction, 6),
        f">={config.min_complex_or_hard_fraction}",
    )
    return findings


def audit_synthetic_corpus(config: SyntheticAuditConfig) -> SyntheticAuditReport:
    manifest_path = config.corpus_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    domain_audits: dict[str, DomainAudit] = {}
    findings: list[AuditFinding] = []
    for domain in config.domains:
        records: list[SyntheticTaskRecord] = []
        for split in SyntheticSplit:
            path = config.corpus_root / domain / f"{split.value}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            records.extend(load_records(path))
        audit = _audit_domain(domain, records, config)
        domain_audits[domain] = audit
        findings.extend(_domain_findings(domain, audit, config))

        manifest_stats = (manifest.get("domains") or {}).get(domain) or {}
        for split, actual in (
            ("train", audit.train_count),
            ("validation", audit.validation_count),
        ):
            expected = manifest_stats.get(f"accepted_{split}")
            findings.append(
                AuditFinding(
                    code=f"manifest_{split}_count",
                    status=(
                        AuditStatus.PASS
                        if expected == actual
                        else AuditStatus.FAIL
                    ),
                    message=f"manifest and {split} JSONL counts agree",
                    domain=domain,
                    value=actual,
                    threshold=expected,
                )
            )

    return SyntheticAuditReport(
        status=_overall_status(findings),
        generated_at=_utc_now(),
        corpus_root=str(config.corpus_root),
        config={**asdict(config), "corpus_root": str(config.corpus_root)},
        domains=domain_audits,
        findings=findings,
    )


def _markdown(report: SyntheticAuditReport) -> str:
    lines = [
        "# Synthetic Corpus Quality Audit",
        "",
        f"Overall status: **{report.status.value}**",
        "",
        "| Domain | Train | Validation | Templates | Dominant | Param variants | Near duplicates | Multi-action |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for domain, audit in report.domains.items():
        lines.append(
            "| "
            + " | ".join(
                (
                    domain,
                    str(audit.train_count),
                    str(audit.validation_count),
                    str(len(audit.template_counts)),
                    f"{audit.dominant_template_fraction:.1%}",
                    f"{audit.parameter_variant_record_fraction:.1%}",
                    f"{audit.near_duplicate_record_fraction:.1%}",
                    f"{audit.multi_action_fraction:.1%}",
                )
            )
            + " |"
        )
    lines.extend(("", "## Findings", ""))
    for finding in report.findings:
        prefix = f"[{finding.status.value}]"
        domain = f" {finding.domain}:" if finding.domain else ""
        lines.append(
            f"- {prefix}{domain} {finding.message}; value={finding.value}, "
            f"threshold={finding.threshold}"
        )
    lines.extend(("", "## Human Review Samples", ""))
    for domain, audit in report.domains.items():
        lines.append(f"### {domain}")
        lines.append("")
        lines.extend(f"- `{task_id}`" for task_id in audit.human_sample_task_ids)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_audit_report(
    report: SyntheticAuditReport,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    with json_output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report.to_dict(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    with markdown_output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(_markdown(report))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=sorted(SUPPORTED_DOMAINS),
        default=["airline", "retail", "telecom"],
    )
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    parser.add_argument("--fail-on-quality-gate", action="store_true")
    args = parser.parse_args()
    root = Path(args.corpus_root)
    report = audit_synthetic_corpus(
        SyntheticAuditConfig(corpus_root=root, domains=tuple(args.domains))
    )
    json_path = Path(args.json_output) if args.json_output else root / "audit.json"
    markdown_path = (
        Path(args.markdown_output)
        if args.markdown_output
        else root / "audit.md"
    )
    write_audit_report(
        report,
        json_path=json_path,
        markdown_path=markdown_path,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    if args.fail_on_quality_gate and report.status is AuditStatus.FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
