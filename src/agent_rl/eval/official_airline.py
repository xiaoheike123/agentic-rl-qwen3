"""Audit and summarize the locked 20-task, four-seed airline evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from agent_rl.data.official_split import load_official_airline_split
from agent_rl.eval.pass_hat_k import pass_hat_k_from_outcomes
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


def _normalize_trial(row: dict[str, Any]) -> dict[str, Any]:
    task_id = str(_field(row, "tau_task_id", row.get("task_id", "")))
    seed = _field(row, "tau_seed", row.get("seed"))
    success = _field(row, "tau_success")
    if success is None:
        reward = _field(row, "tau_outcome_reward", row.get("reward", 0.0))
        success = float(reward or 0.0) >= 1.0
    return {
        "task_id": task_id,
        "seed": int(seed) if seed is not None else None,
        "success": bool(success),
        "turns": int(_field(row, "tau_total_turns", 0) or 0),
        "prompt_tokens": int(_field(row, "tau_prompt_tokens", 0) or 0),
        "response_tokens": int(_field(row, "tau_response_tokens", 0) or 0),
        "tool_calls": int(_field(row, "tau_tool_call_count", 0) or 0),
        "tool_errors": int(_field(row, "tau_tool_error_count", 0) or 0),
        "invalid_actions": int(_field(row, "tau_invalid_action_count", 0) or 0),
        "hit_max_turns": bool(_field(row, "tau_hit_max_turns", False)),
        "termination_reason": _field(row, "tau_termination_reason"),
    }


def load_trials(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    trials = []
    for path in paths:
        trials.extend(_normalize_trial(row) for row in iter_json_objects(path))
    return trials


def build_official_report(trials: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = load_official_airline_split()
    expected_tasks = set(manifest.test_task_ids)
    expected_seeds = set(manifest.evaluation_seeds)
    if len(trials) != 80:
        raise ValueError(f"final airline evaluation requires 80 trials, got {len(trials)}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for trial in trials:
        task_id = trial["task_id"]
        seed = trial["seed"]
        if task_id not in expected_tasks:
            raise ValueError(f"unexpected official test task: {task_id!r}")
        if seed not in expected_seeds:
            raise ValueError(f"unexpected evaluation seed for task {task_id}: {seed}")
        identity = (task_id, seed)
        if identity in seen:
            raise ValueError(f"duplicate evaluation trial: task={task_id}, seed={seed}")
        seen.add(identity)
        grouped[task_id].append(trial)

    task_reports: dict[str, dict[str, Any]] = {}
    for task_id in sorted(expected_tasks, key=int):
        task_trials = sorted(grouped[task_id], key=lambda item: item["seed"])
        seeds = {trial["seed"] for trial in task_trials}
        if seeds != expected_seeds:
            raise ValueError(
                f"task {task_id} does not contain exactly the four locked seeds"
            )
        outcomes = [trial["success"] for trial in task_trials]
        task_reports[task_id] = {
            "successes": sum(outcomes),
            "success_rate": mean(outcomes),
            **{
                f"pass^{k}": pass_hat_k_from_outcomes(outcomes, k)
                for k in range(1, 5)
            },
            "mean_turns": mean(trial["turns"] for trial in task_trials),
        }

    return {
        "protocol": "tau2_official_airline_test_20_tasks_x_4_seeds",
        "episodes": len(trials),
        "tasks": len(task_reports),
        "seeds": list(manifest.evaluation_seeds),
        "success_rate": mean(trial["success"] for trial in trials),
        **{
            f"pass^{k}": mean(report[f"pass^{k}"] for report in task_reports.values())
            for k in range(1, 5)
        },
        "mean_turns": mean(trial["turns"] for trial in trials),
        "mean_response_tokens": mean(trial["response_tokens"] for trial in trials),
        "mean_tool_calls": mean(trial["tool_calls"] for trial in trials),
        "tool_error_rate": mean(trial["tool_errors"] > 0 for trial in trials),
        "invalid_action_rate": mean(
            trial["invalid_actions"] > 0 for trial in trials
        ),
        "max_turn_rate": mean(trial["hit_max_turns"] for trial in trials),
        "task_results": task_reports,
    }


def write_report(
    output_dir: str | Path,
    trials: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (target / "trials.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trials[0]))
        writer.writeheader()
        writer.writerows(trials)
    task_rows = [
        {"task_id": task_id, **values}
        for task_id, values in report["task_results"].items()
    ]
    with (target / "tasks.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    trials = load_trials(args.inputs)
    report = build_official_report(trials)
    write_report(args.output_dir, trials, report)
    print(json.dumps({key: value for key, value in report.items() if key != "task_results"}, indent=2))


if __name__ == "__main__":
    main()
