"""Diagnose GRPO group diversity before committing to a full run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from agent_rl.utils.jsonl import iter_json_objects


def _field(row: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in row:
        return row[key]
    for container_key in ("reward_extra_info", "extra_fields"):
        container = row.get(container_key)
        if not isinstance(container, dict):
            continue
        if key in container:
            return container[key]
        nested = container.get("reward_extra_info")
        if isinstance(nested, dict) and key in nested:
            return nested[key]
    return default


def _response_fingerprint(row: dict[str, Any]) -> str | None:
    for key in ("response_ids", "response", "responses", "output"):
        value = row.get(key)
        if value not in (None, "", []):
            payload = json.dumps(value, sort_keys=True, ensure_ascii=False)
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return None


def build_group_report(rows: Iterable[dict[str, Any]], *, expected_group_size: int = 4) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_id = str(_field(row, "tau_group_id", row.get("uid", "")))
        if not group_id:
            raise ValueError("rollout row has no tau_group_id or uid")
        groups[group_id].append(row)
    if not groups:
        raise ValueError("no rollout rows were found")

    diagnostics = []
    for group_id, group_rows in sorted(groups.items()):
        if len(group_rows) != expected_group_size:
            raise ValueError(
                f"group {group_id!r} contains {len(group_rows)} rows; "
                f"expected {expected_group_size}"
            )
        rewards = [
            float(
                _field(
                    row,
                    "tau_outcome_reward",
                    _field(row, "reward", 0.0),
                )
                or 0.0
            )
            for row in group_rows
        ]
        fingerprints = [
            fingerprint
            for fingerprint in (_response_fingerprint(row) for row in group_rows)
            if fingerprint is not None
        ]
        diagnostics.append(
            {
                "group_id": group_id,
                "rewards": rewards,
                "nonzero_advantage": max(rewards) - min(rewards) > 1e-12,
                "duplicate_responses": len(fingerprints) - len(set(fingerprints)),
                "tool_errors": sum(
                    int(_field(row, "tau_tool_error_count", 0) or 0)
                    for row in group_rows
                ),
                "invalid_actions": sum(
                    int(_field(row, "tau_invalid_action_count", 0) or 0)
                    for row in group_rows
                ),
            }
        )

    coverage = mean(item["nonzero_advantage"] for item in diagnostics)
    if coverage >= 0.5:
        recommendation = "KEEP_G4"
    elif coverage >= 0.3:
        recommendation = "TUNE_TEMPERATURE_OR_PROMPT_THEN_REPEAT_G4"
    else:
        recommendation = "CONSIDER_G6_OR_G8_AFTER_DIAGNOSIS"
    return {
        "groups": len(diagnostics),
        "trajectories": len(diagnostics) * expected_group_size,
        "group_size": expected_group_size,
        "nonzero_advantage_groups": sum(
            item["nonzero_advantage"] for item in diagnostics
        ),
        "nonzero_advantage_fraction": coverage,
        "duplicate_responses": sum(
            item["duplicate_responses"] for item in diagnostics
        ),
        "tool_errors": sum(item["tool_errors"] for item in diagnostics),
        "invalid_actions": sum(item["invalid_actions"] for item in diagnostics),
        "recommendation": recommendation,
        "group_results": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--group-size", type=int, default=4)
    args = parser.parse_args()
    rows = [row for path in args.inputs for row in iter_json_objects(path)]
    report = build_group_report(rows, expected_group_size=args.group_size)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
