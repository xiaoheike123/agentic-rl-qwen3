"""Build minimal verl datasets containing only public tau2 task identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_rl.data.tau_tasks import load_tau_task_refs


def build_tau_dataset(
    output_path: str | Path,
    *,
    domain: str,
    split: str,
    seed: int = 42,
    limit: int | None = None,
) -> int:
    refs = list(load_tau_task_refs(domain, split=split))
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        refs = refs[:limit]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, ref in enumerate(refs):
            row = {
                "data_source": "tau2",
                "prompt": [
                    {
                        "role": "user",
                        "content": f"Run tau2 task {ref.task_id} in {ref.domain}.",
                    }
                ],
                "domain": ref.domain,
                "task_id": ref.task_id,
                "seed": seed + index,
                "extra_info": {"index": index},
            }
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    return len(refs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument(
        "--split", choices=("all", "train", "validation", "test"), required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    count = build_tau_dataset(
        args.output,
        domain=args.domain,
        split=args.split,
        seed=args.seed,
        limit=args.limit,
    )
    print(f"wrote {count} tau2 task rows to {args.output}")


if __name__ == "__main__":
    main()
