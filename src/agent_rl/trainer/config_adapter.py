"""Load one experiment manifest and its referenced component YAML files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    path: Path
    experiment: str
    model: dict[str, Any]
    environment: dict[str, Any]
    algorithm: dict[str, Any]
    rollout: dict[str, Any]
    reward: dict[str, Any]
    credit: dict[str, Any] | None
    runtime: dict[str, Any]
    raw: dict[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return value


def _resolve(project_root: Path, value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must reference a YAML file")
    return _load_yaml((project_root / value).resolve())


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    raw = _load_yaml(config_path)
    project_root = config_path.parents[2]
    experiment = raw.get("experiment")
    if not isinstance(experiment, str) or not experiment.strip():
        raise ValueError("experiment manifest requires an experiment name")

    credit_ref = raw.get("credit_config")
    credit = (
        None
        if credit_ref is None
        else _resolve(project_root, credit_ref, "credit_config")
    )
    return ExperimentConfig(
        path=config_path,
        experiment=experiment,
        model=_resolve(project_root, raw.get("model_config"), "model_config"),
        environment=_resolve(project_root, raw.get("env_config"), "env_config"),
        algorithm=_resolve(
            project_root, raw.get("algorithm_config"), "algorithm_config"
        ),
        rollout=_resolve(project_root, raw.get("rollout_config"), "rollout_config"),
        reward=_resolve(project_root, raw.get("reward_config"), "reward_config"),
        credit=credit,
        runtime=_resolve(project_root, raw.get("runtime_config"), "runtime_config"),
        raw=raw,
    )
