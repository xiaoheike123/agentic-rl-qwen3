"""Export trajectory results to compact leaderboard JSON."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Sequence

from agent_rl.data.schemas import EpisodeRecord
from agent_rl.eval.pass_at_k import pass_at_k_from_outcomes


def build_leaderboard_report(
    episodes: Sequence[EpisodeRecord],
    *,
    pass_k: int = 1,
) -> dict:
    if not episodes:
        raise ValueError("episodes must not be empty")
    by_task: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for episode in episodes:
        by_task[episode.task_id].append(episode)

    tasks = {}
    for task_id, task_episodes in sorted(by_task.items()):
        outcomes = [episode.success is True for episode in task_episodes]
        if pass_k > len(outcomes):
            raise ValueError(f"task {task_id!r} has fewer than pass_k={pass_k} trials")
        tasks[task_id] = {
            "trials": len(outcomes),
            "success_rate": mean(outcomes),
            "pass_at_k": pass_at_k_from_outcomes(outcomes, pass_k),
            "mean_turns": mean(len(episode.turns) for episode in task_episodes),
        }

    return {
        "domain": episodes[0].domain,
        "model": episodes[0].model,
        "episodes": len(episodes),
        "tasks": len(tasks),
        "success_rate": mean(episode.success is True for episode in episodes),
        "pass_k": pass_k,
        "mean_task_pass_at_k": mean(item["pass_at_k"] for item in tasks.values()),
        "task_results": tasks,
    }


def write_leaderboard_report(path: str | Path, report: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
