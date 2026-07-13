"""Deterministic domain-balanced export for verl agent-loop datasets."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from agent_rl.data.synthetic.schema import SyntheticSplit, SyntheticTaskRecord
from agent_rl.data.synthetic.storage import load_records


def balanced_records(
    records: Iterable[SyntheticTaskRecord],
    *,
    split: SyntheticSplit,
    seed: int,
    per_domain_limit: int | None = None,
) -> list[SyntheticTaskRecord]:
    by_domain: dict[str, list[SyntheticTaskRecord]] = {}
    for record in records:
        if record.split is split:
            by_domain.setdefault(record.domain, []).append(record)
    if not by_domain:
        raise ValueError(f"no synthetic records found for split={split.value!r}")

    expected = {"airline", "retail", "telecom"}
    missing = expected - set(by_domain)
    if missing:
        raise ValueError(f"missing synthetic domains: {sorted(missing)}")

    rng = random.Random(seed)
    for domain_records in by_domain.values():
        domain_records.sort(key=lambda item: item.task_id)
        rng.shuffle(domain_records)

    count = min(len(values) for values in by_domain.values())
    if per_domain_limit is not None:
        if per_domain_limit <= 0:
            raise ValueError("per_domain_limit must be positive")
        count = min(count, per_domain_limit)
    if count == 0:
        raise ValueError("at least one domain has no usable synthetic records")

    output: list[SyntheticTaskRecord] = []
    domains = sorted(by_domain)
    for index in range(count):
        rotated = domains[index % len(domains) :] + domains[: index % len(domains)]
        output.extend(by_domain[domain][index] for domain in rotated)
    return output


def build_balanced_verl_dataset(
    output_path: str | Path,
    *,
    corpus_root: str | Path,
    split: SyntheticSplit,
    seed: int,
    per_domain_limit: int | None = None,
) -> int:
    root = Path(corpus_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_config = manifest.get("config") or {}
    training_database_fingerprint = manifest_config.get(
        "training_database_fingerprint"
    )
    if not isinstance(training_database_fingerprint, str) or not (
        training_database_fingerprint.strip()
    ):
        raise ValueError(
            "synthetic corpus manifest has no training database fingerprint"
        )

    records = []
    for domain in ("airline", "retail", "telecom"):
        records.extend(load_records(root / domain / f"{split.value}.jsonl"))
    selected = balanced_records(
        records,
        split=split,
        seed=seed,
        per_domain_limit=per_domain_limit,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for index, record in enumerate(selected):
            row = {
                "data_source": "tau2_synthetic",
                "prompt": [
                    {
                        "role": "user",
                        "content": (
                            f"Run synthetic tau2 task {record.task_id} in "
                            f"{record.domain}."
                        ),
                    }
                ],
                "domain": record.domain,
                "task_id": record.task_id,
                "synthetic_task": record.task,
                "seed": seed + index,
                "extra_info": {
                    "index": index,
                    "split": split.value,
                    "semantic_fingerprint": record.semantic_fingerprint,
                    "generator_version": record.generation.generator_version,
                    "database_source": "pseudonymized_training",
                    "training_database_fingerprint": (
                        training_database_fingerprint
                    ),
                },
            }
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            )
            stream.write("\n")
    temporary.replace(output)
    return len(selected)
