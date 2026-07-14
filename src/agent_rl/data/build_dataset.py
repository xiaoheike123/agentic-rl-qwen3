"""Build benchmark-safe verl datasets from locked official tau2 task IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from agent_rl.data.official_split import load_official_airline_split


def _write_rows(path: Path, rows: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
            count += 1
    return count


def _official_row(
    *,
    split: str,
    task_id: str,
    seed: int,
    row_index: int,
    evaluation_only: bool,
) -> dict[str, object]:
    return {
        "uid": f"airline:{split}:{task_id}:{seed}" if evaluation_only else f"airline:{split}:{task_id}",
        "data_source": f"tau2_official_{split}",
        "prompt": [
            {
                "role": "user",
                "content": f"Run tau2 task {task_id} in airline.",
            }
        ],
        "domain": "airline",
        "task_id": task_id,
        "seed": seed,
        "extra_info": {
            "index": row_index,
            "official_split": split,
            "evaluation_only": evaluation_only,
            "evaluation_seed": seed if evaluation_only else None,
        },
    }


def build_official_train_dataset(
    output_path: str | Path,
    *,
    seed: int = 42,
    task_limit: int | None = None,
) -> int:
    """Write each of the 30 locked airline train tasks exactly once."""

    manifest = load_official_airline_split()
    task_ids = manifest.train_task_ids
    if task_limit is not None:
        if not 1 <= task_limit <= len(task_ids):
            raise ValueError("task_limit must be between 1 and 30")
        task_ids = task_ids[:task_limit]
    rows = (
        _official_row(
            split="train",
            task_id=task_id,
            seed=seed,
            row_index=index,
            evaluation_only=False,
        )
        for index, task_id in enumerate(task_ids)
    )
    return _write_rows(Path(output_path), rows)


def build_official_test_dataset(
    output_path: str | Path,
) -> int:
    """Write the 20 test tasks at four fixed seeds, producing 80 rows."""

    manifest = load_official_airline_split()
    rows = []
    for task_id in manifest.test_task_ids:
        for seed in manifest.evaluation_seeds:
            rows.append(
                _official_row(
                    split="test",
                    task_id=task_id,
                    seed=seed,
                    row_index=len(rows),
                    evaluation_only=True,
                )
            )
    return _write_rows(Path(output_path), rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.split == "train":
        count = build_official_train_dataset(args.output, seed=args.seed)
    else:
        count = build_official_test_dataset(args.output)
    print(f"wrote {count} locked official airline rows to {args.output}")


if __name__ == "__main__":
    main()
