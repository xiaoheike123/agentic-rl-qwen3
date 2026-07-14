"""Immutable project manifest for the official tau2 airline split."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agent_rl.data.tau_tasks import load_tau_task_refs


DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "data"
    / "airline_official.json"
)


@dataclass(frozen=True, slots=True)
class OfficialAirlineSplit:
    domain: str
    tau2_commit: str
    train_task_ids: tuple[str, ...]
    test_task_ids: tuple[str, ...]
    evaluation_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.domain != "airline":
            raise ValueError("the formal project domain must be airline")
        if len(self.train_task_ids) != 30:
            raise ValueError("the official airline train split must contain 30 tasks")
        if len(self.test_task_ids) != 20:
            raise ValueError("the official airline test split must contain 20 tasks")
        if set(self.train_task_ids) & set(self.test_task_ids):
            raise ValueError("official train and test task IDs must be disjoint")
        if len(set(self.evaluation_seeds)) != 4:
            raise ValueError("final evaluation requires four distinct seeds")

    def task_ids(self, split: str) -> tuple[str, ...]:
        if split == "train":
            return self.train_task_ids
        if split == "test":
            return self.test_task_ids
        raise ValueError("formal airline split must be 'train' or 'test'")

    def validate_against_tau2(self) -> None:
        for split in ("train", "test"):
            actual = {
                ref.task_id for ref in load_tau_task_refs(self.domain, split=split)
            }
            expected = set(self.task_ids(split))
            if actual != expected:
                missing = sorted(expected - actual, key=int)
                unexpected = sorted(actual - expected, key=int)
                raise RuntimeError(
                    f"tau2 {self.domain}/{split} differs from the locked manifest; "
                    f"missing={missing}, unexpected={unexpected}"
                )


def load_official_airline_split(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    validate_tau2: bool = True,
) -> OfficialAirlineSplit:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = OfficialAirlineSplit(
        domain=str(payload["domain"]),
        tau2_commit=str(payload["tau2_commit"]),
        train_task_ids=tuple(str(value) for value in payload["train_task_ids"]),
        test_task_ids=tuple(str(value) for value in payload["test_task_ids"]),
        evaluation_seeds=tuple(int(value) for value in payload["evaluation_seeds"]),
    )
    if validate_tau2:
        manifest.validate_against_tau2()
    return manifest
