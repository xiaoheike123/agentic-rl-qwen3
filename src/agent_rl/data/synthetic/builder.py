"""Build a verified, benchmark-filtered three-domain synthetic corpus."""

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
from agent_rl.data.synthetic.overlap import BenchmarkOverlapGuard
from agent_rl.data.synthetic.schema import (
    SUPPORTED_DOMAINS,
    SyntheticSplit,
    SyntheticTaskRecord,
)
from agent_rl.data.synthetic.storage import write_records
from agent_rl.data.synthetic.verifier import verify_oracle_task


@dataclass(frozen=True, slots=True)
class SyntheticBuildConfig:
    output_root: Path
    domains: tuple[str, ...] = ("airline", "retail", "telecom")
    seed: int = 42
    validation_fraction: float = 0.15
    similarity_threshold: float = 0.82
    max_per_split_per_domain: int | None = None

    def __post_init__(self) -> None:
        invalid = set(self.domains) - SUPPORTED_DOMAINS
        if invalid:
            raise ValueError(f"unsupported domains: {sorted(invalid)}")
        if not self.domains:
            raise ValueError("at least one domain is required")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        if self.max_per_split_per_domain is not None:
            if self.max_per_split_per_domain <= 0:
                raise ValueError("max_per_split_per_domain must be positive")


@dataclass(slots=True)
class DomainBuildStats:
    generated: int = 0
    accepted_train: int = 0
    accepted_validation: int = 0
    rejected_oracle: int = 0
    rejected_overlap: int = 0
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
    actual = manifest.get("config") or {}
    expected = {
        "generator_version": GENERATOR_VERSION,
        "domains": list(config.domains),
        "seed": config.seed,
        "validation_fraction": config.validation_fraction,
        "similarity_threshold": config.similarity_threshold,
        "max_per_split_per_domain": config.max_per_split_per_domain,
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
    report = SyntheticBuildReport()
    rejected_rows: list[dict[str, object]] = []

    for domain_index, domain in enumerate(config.domains):
        stats = DomainBuildStats()
        report.domains[domain] = stats
        domain_seed = config.seed + domain_index * 1_000_003
        candidates = GENERATORS[domain](domain_seed)
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
        guard = BenchmarkOverlapGuard(
            domain,
            similarity_threshold=config.similarity_threshold,
        )
        by_split: dict[SyntheticSplit, list[SyntheticTaskRecord]] = {
            SyntheticSplit.TRAIN: [],
            SyntheticSplit.VALIDATION: [],
        }
        seen_fingerprints: set[str] = set()

        for candidate in candidates:
            primary_entity = candidate.generation.source_entities[0]
            split = split_map[primary_entity]
            if (
                config.max_per_split_per_domain is not None
                and len(by_split[split]) >= config.max_per_split_per_domain
            ):
                continue
            verification = verify_oracle_task(domain, candidate.task)
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

            overlap = guard.inspect(candidate.task)
            if not overlap.passed:
                stats.rejected_overlap += 1
                rejected_rows.append(
                    {
                        "domain": domain,
                        "task_id": candidate.task.id,
                        "reason": "benchmark_overlap",
                        "nearest_task_id": overlap.nearest_task_id,
                        "similarity": overlap.nearest_similarity,
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
                overlap=overlap,
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
                    **asdict(config),
                    "output_root": str(config.output_root),
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
    parser.add_argument("--similarity-threshold", type=float, default=0.82)
    parser.add_argument("--max-per-split-per-domain", type=int)
    args = parser.parse_args()
    report = build_synthetic_corpus(
        SyntheticBuildConfig(
            output_root=Path(args.output_root),
            domains=tuple(args.domains),
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            similarity_threshold=args.similarity_threshold,
            max_per_split_per_domain=args.max_per_split_per_domain,
        )
    )
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
