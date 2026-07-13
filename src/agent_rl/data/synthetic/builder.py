"""Build a verified, clean-room three-domain synthetic corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agent_rl.data.synthetic.fingerprint import semantic_fingerprint
from agent_rl.data.synthetic.generators import GENERATORS
from agent_rl.data.synthetic.generators.common import GENERATOR_VERSION
from agent_rl.data.synthetic.policy_validation import validate_candidate_policy
from agent_rl.data.synthetic.schema import (
    SUPPORTED_DOMAINS,
    SyntheticSplit,
    SyntheticTaskRecord,
)
from agent_rl.data.synthetic.storage import write_records
from agent_rl.data.synthetic.training_db import (
    TRAINING_DB_VERSION,
    TrainingDatabaseConfig,
    build_training_databases,
    load_training_database,
    training_database_fingerprint,
    validate_training_databases,
)
from agent_rl.data.synthetic.verifier import verify_oracle_task


@dataclass(frozen=True, slots=True)
class SyntheticBuildConfig:
    output_root: Path
    domains: tuple[str, ...] = ("airline", "retail", "telecom")
    seed: int = 42
    validation_fraction: float = 0.15
    max_train_per_domain: int | None = None
    max_validation_per_domain: int | None = None
    training_database_root: Path | None = None
    telecom_clone_factor: int = 16

    def __post_init__(self) -> None:
        invalid = set(self.domains) - SUPPORTED_DOMAINS
        if invalid:
            raise ValueError(f"unsupported domains: {sorted(invalid)}")
        if not self.domains:
            raise ValueError("at least one domain is required")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        for name, value in (
            ("max_train_per_domain", self.max_train_per_domain),
            ("max_validation_per_domain", self.max_validation_per_domain),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.telecom_clone_factor <= 0:
            raise ValueError("telecom_clone_factor must be positive")

    @property
    def resolved_training_database_root(self) -> Path:
        return self.training_database_root or self.output_root.parent / "training_db"

    def limit_for(self, split: SyntheticSplit) -> int | None:
        if split is SyntheticSplit.TRAIN:
            return self.max_train_per_domain
        return self.max_validation_per_domain


@dataclass(slots=True)
class DomainBuildStats:
    generated: int = 0
    accepted_train: int = 0
    accepted_validation: int = 0
    rejected_policy: int = 0
    rejected_oracle: int = 0
    rejected_duplicate: int = 0


@dataclass(slots=True)
class SyntheticBuildReport:
    domains: dict[str, DomainBuildStats] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {"domains": {key: asdict(value) for key, value in self.domains.items()}}


def validate_corpus_manifest(config: SyntheticBuildConfig) -> None:
    manifest_path = config.output_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    training_config = TrainingDatabaseConfig(
        output_root=config.resolved_training_database_root,
        seed=config.seed,
        telecom_clone_factor=config.telecom_clone_factor,
    )
    training_manifest = validate_training_databases(training_config)
    actual = manifest.get("config") or {}
    expected = {
        "generator_version": GENERATOR_VERSION,
        "training_database_version": TRAINING_DB_VERSION,
        "training_database_fingerprint": training_database_fingerprint(
            training_manifest
        ),
        "domains": list(config.domains),
        "seed": config.seed,
        "validation_fraction": config.validation_fraction,
        "max_train_per_domain": config.max_train_per_domain,
        "max_validation_per_domain": config.max_validation_per_domain,
        "telecom_clone_factor": config.telecom_clone_factor,
    }
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "synthetic corpus manifest does not match runtime config; "
            f"rebuild the corpus. mismatches={mismatches}"
        )
    for domain in config.domains:
        for split in SyntheticSplit:
            path = config.output_root / domain / f"{split.value}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)


def _entity_split_map(
    domain: str,
    entity_groups: set[str],
    *,
    seed: int,
    validation_fraction: float,
) -> dict[str, SyntheticSplit]:
    if len(entity_groups) < 2:
        raise ValueError(f"{domain} needs at least two independent entity groups")
    groups = sorted(
        entity_groups,
        key=lambda value: hashlib.sha256(
            f"{seed}:{domain}:{value}".encode("utf-8")
        ).hexdigest(),
    )
    validation_count = max(1, round(len(groups) * validation_fraction))
    validation_count = min(validation_count, len(groups) - 1)
    validation_groups = set(groups[:validation_count])
    return {
        group: (
            SyntheticSplit.VALIDATION
            if group in validation_groups
            else SyntheticSplit.TRAIN
        )
        for group in groups
    }


def build_synthetic_corpus(config: SyntheticBuildConfig) -> SyntheticBuildReport:
    config.output_root.mkdir(parents=True, exist_ok=True)
    training_config = TrainingDatabaseConfig(
        output_root=config.resolved_training_database_root,
        seed=config.seed,
        telecom_clone_factor=config.telecom_clone_factor,
    )
    if (training_config.output_root / "manifest.json").is_file():
        training_manifest = validate_training_databases(training_config)
    else:
        training_manifest = build_training_databases(training_config)
    report = SyntheticBuildReport()
    rejected_rows: list[dict[str, object]] = []

    for domain_index, domain in enumerate(config.domains):
        stats = DomainBuildStats()
        report.domains[domain] = stats
        domain_seed = config.seed + domain_index * 1_000_003
        database = load_training_database(training_config.output_root, domain)
        candidates = GENERATORS[domain](domain_seed, database)
        random.Random(domain_seed).shuffle(candidates)
        stats.generated = len(candidates)
        split_map = _entity_split_map(
            domain,
            {
                candidate.generation.source_entities[0]
                for candidate in candidates
                if candidate.generation.source_entities
            },
            seed=config.seed,
            validation_fraction=config.validation_fraction,
        )
        by_split: dict[SyntheticSplit, list[SyntheticTaskRecord]] = {
            SyntheticSplit.TRAIN: [],
            SyntheticSplit.VALIDATION: [],
        }
        seen_fingerprints: set[str] = set()

        for candidate in candidates:
            primary_entity = candidate.generation.source_entities[0]
            split = split_map[primary_entity]
            split_limit = config.limit_for(split)
            if split_limit is not None and len(by_split[split]) >= split_limit:
                continue
            try:
                validate_candidate_policy(candidate, database)
            except ValueError as error:
                stats.rejected_policy += 1
                rejected_rows.append(
                    {
                        "domain": domain,
                        "task_id": candidate.task.id,
                        "reason": "policy",
                        "detail": str(error),
                    }
                )
                continue
            verification = verify_oracle_task(domain, candidate.task, database)
            if not verification.oracle_verified:
                stats.rejected_oracle += 1
                rejected_rows.append(
                    {
                        "domain": domain,
                        "task_id": candidate.task.id,
                        "reason": "oracle",
                        "detail": verification.error,
                    }
                )
                continue

            fingerprint = semantic_fingerprint(domain, candidate.task)
            if fingerprint in seen_fingerprints:
                stats.rejected_duplicate += 1
                continue
            seen_fingerprints.add(fingerprint)
            record = SyntheticTaskRecord(
                domain=domain,
                split=split,
                task=candidate.task.model_dump(mode="json"),
                semantic_fingerprint=fingerprint,
                generation=candidate.generation,
                verification=verification,
                metadata={
                    **candidate.metadata,
                    "policy_validation": {"passed": True},
                },
            )
            by_split[split].append(record)

        for split, records in by_split.items():
            records.sort(key=lambda item: item.task_id)
            write_records(
                config.output_root / domain / f"{split.value}.jsonl",
                records,
            )
            if split is SyntheticSplit.TRAIN:
                stats.accepted_train = len(records)
            else:
                stats.accepted_validation = len(records)

        if stats.accepted_train == 0 or stats.accepted_validation == 0:
            raise RuntimeError(
                f"{domain} produced an empty split: "
                f"train={stats.accepted_train}, validation={stats.accepted_validation}"
            )

    with (config.output_root / "rejections.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for row in rejected_rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    with (config.output_root / "manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(
            {
                "config": {
                    "generator_version": GENERATOR_VERSION,
                    "training_database_version": TRAINING_DB_VERSION,
                    "training_database_fingerprint": (
                        training_database_fingerprint(training_manifest)
                    ),
                    "output_root": str(config.output_root),
                    "domains": list(config.domains),
                    "seed": config.seed,
                    "validation_fraction": config.validation_fraction,
                    "max_train_per_domain": config.max_train_per_domain,
                    "max_validation_per_domain": config.max_validation_per_domain,
                    "telecom_clone_factor": config.telecom_clone_factor,
                },
                **report.to_dict(),
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=sorted(SUPPORTED_DOMAINS),
        default=["airline", "retail", "telecom"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-train-per-domain", type=int)
    parser.add_argument("--max-validation-per-domain", type=int)
    parser.add_argument("--training-database-root")
    parser.add_argument("--telecom-clone-factor", type=int, default=16)
    args = parser.parse_args()
    report = build_synthetic_corpus(
        SyntheticBuildConfig(
            output_root=Path(args.output_root),
            domains=tuple(args.domains),
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            max_train_per_domain=args.max_train_per_domain,
            max_validation_per_domain=args.max_validation_per_domain,
            training_database_root=(
                Path(args.training_database_root)
                if args.training_database_root
                else None
            ),
            telecom_clone_factor=args.telecom_clone_factor,
        )
    )
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
