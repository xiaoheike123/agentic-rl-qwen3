"""Reject generated tasks that duplicate public tau2 leaderboard tasks."""

from __future__ import annotations

from dataclasses import dataclass

from tau2.data_model.tasks import Task
from tau2.registry import registry

from agent_rl.data.synthetic.fingerprint import (
    action_signature,
    exact_fingerprint,
    jaccard,
    task_tokens,
)
from agent_rl.data.synthetic.schema import OverlapMetadata


@dataclass(frozen=True, slots=True)
class _BenchmarkTask:
    task_id: str
    exact: str
    action_arguments: tuple[str, ...]
    action_names: tuple[str, ...]
    tokens: frozenset[str]


class BenchmarkOverlapGuard:
    """In-memory index of the official `base` split for one domain."""

    def __init__(self, domain: str, *, similarity_threshold: float = 0.82) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between zero and one")
        self.domain = domain
        self.similarity_threshold = similarity_threshold
        official_tasks = registry.get_tasks_loader(domain)(task_split_name="base")
        self._tasks = tuple(self._index(task) for task in official_tasks)
        if not self._tasks:
            raise ValueError(f"official base split is empty for {domain!r}")

    @staticmethod
    def _index(task: Task) -> _BenchmarkTask:
        return _BenchmarkTask(
            task_id=task.id,
            exact=exact_fingerprint(task),
            action_arguments=action_signature(task, include_arguments=True),
            action_names=action_signature(task, include_arguments=False),
            tokens=task_tokens(task),
        )

    def inspect(self, task: Task) -> OverlapMetadata:
        candidate_exact = exact_fingerprint(task)
        candidate_arguments = action_signature(task, include_arguments=True)
        candidate_names = action_signature(task, include_arguments=False)
        candidate_tokens = task_tokens(task)

        exact_match = False
        same_action_arguments = False
        nearest_task_id: str | None = None
        nearest_similarity = 0.0
        for benchmark in self._tasks:
            if candidate_exact == benchmark.exact:
                exact_match = True
            if candidate_arguments and candidate_arguments == benchmark.action_arguments:
                same_action_arguments = True
            similarity = jaccard(candidate_tokens, benchmark.tokens)
            if similarity > nearest_similarity:
                nearest_similarity = similarity
                nearest_task_id = benchmark.task_id

        similar_same_skill = (
            nearest_similarity >= self.similarity_threshold
            and any(
                candidate_names == benchmark.action_names
                and jaccard(candidate_tokens, benchmark.tokens)
                >= self.similarity_threshold
                for benchmark in self._tasks
            )
        )
        passed = not (exact_match or same_action_arguments or similar_same_skill)
        return OverlapMetadata(
            passed=passed,
            exact_match=exact_match,
            same_action_arguments=same_action_arguments,
            nearest_task_id=nearest_task_id,
            nearest_similarity=nearest_similarity,
            threshold=self.similarity_threshold,
        )
