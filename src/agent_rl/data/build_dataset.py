"""Export official evaluation or benchmark-safe synthetic verl datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_rl.data.synthetic.sampler import build_balanced_verl_dataset
from agent_rl.data.synthetic.schema import SyntheticSplit
from agent_rl.data.tau_tasks import load_tau_task_refs


def build_official_eval_dataset(
    output_path: str | Path,
    *,
    domains: tuple[str, ...],
    split: str = "base",
    seed: int = 42,
) -> int:
    rows = []
    for domain in domains:
        rows.extend(load_tau_task_refs(domain, split=split))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, ref in enumerate(rows):
            row = {
                "data_source": "tau2_official_eval",
                "prompt": [
                    {
                        "role": "user",
                        "content": f"Run tau2 task {ref.task_id} in {ref.domain}.",
                    }
                ],
                "domain": ref.domain,
                "task_id": ref.task_id,
                "seed": seed + index,
                "extra_info": {
                    "index": index,
                    "official_split": split,
                    "evaluation_only": True,
                },
            }
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    synthetic = subparsers.add_parser("synthetic")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--corpus-root", required=True)
    synthetic.add_argument(
        "--split", choices=("train", "validation"), required=True
    )
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument("--per-domain-limit", type=int)

    official = subparsers.add_parser("official-eval")
    official.add_argument("--output", required=True)
    official.add_argument(
        "--domains", nargs="+", default=["airline", "retail", "telecom"]
    )
    official.add_argument("--split", default="base")
    official.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "synthetic":
        count = build_balanced_verl_dataset(
            args.output,
            corpus_root=args.corpus_root,
            split=SyntheticSplit(args.split),
            seed=args.seed,
            per_domain_limit=args.per_domain_limit,
        )
    else:
        count = build_official_eval_dataset(
            args.output,
            domains=tuple(args.domains),
            split=args.split,
            seed=args.seed,
        )
    print(f"wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
