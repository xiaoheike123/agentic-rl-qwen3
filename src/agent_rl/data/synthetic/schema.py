"""Serializable contract for generated tau2 tasks and their audit metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from tau2.data_model.tasks import Task


SYNTHETIC_SCHEMA_VERSION = 2
SUPPORTED_DOMAINS = frozenset({"airline", "retail", "telecom"})


class SyntheticSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class GenerationMetadata:
    generator: str
    generator_version: str
    seed: int
    template: str
    source_entities: tuple[str, ...] = ()
    created_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        for name, value in (
            ("generator", self.generator),
            ("generator_version", self.generator_version),
            ("template", self.template),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, slots=True)
class VerificationMetadata:
    oracle_verified: bool
    database_changed: bool
    action_count: int
    initial_db_hash: str | None = None
    target_db_hash: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.action_count < 0:
            raise ValueError("action_count must be non-negative")
        if self.oracle_verified and self.action_count == 0:
            raise ValueError("a verified task requires at least one oracle action")
        if self.oracle_verified and self.error is not None:
            raise ValueError("a verified task cannot contain a verification error")
        if self.oracle_verified and not self.database_changed:
            raise ValueError("a verified task must change the environment database")


@dataclass(frozen=True, slots=True)
class SyntheticTaskRecord:
    """A native tau2 Task plus provenance required for safe training."""

    domain: str
    split: SyntheticSplit
    task: dict[str, Any]
    semantic_fingerprint: str
    generation: GenerationMetadata
    verification: VerificationMetadata
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SYNTHETIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SYNTHETIC_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported synthetic schema version {self.schema_version}"
            )
        if self.domain not in SUPPORTED_DOMAINS:
            raise ValueError(f"unsupported synthetic domain: {self.domain!r}")
        if not self.semantic_fingerprint.strip():
            raise ValueError("semantic_fingerprint must not be empty")

        task = Task.model_validate(self.task)
        if not task.id.startswith(f"synthetic-{self.domain}-"):
            raise ValueError("synthetic task IDs must include their domain prefix")
        if not self.verification.oracle_verified:
            raise ValueError("unverified tasks cannot enter the synthetic corpus")

    @property
    def task_id(self) -> str:
        return str(self.task["id"])

    def to_tau_task(self) -> Task:
        return Task.model_validate(self.task)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SyntheticTaskRecord:
        data = dict(value)
        data["split"] = SyntheticSplit(data["split"])
        generation = dict(data["generation"])
        generation["source_entities"] = tuple(
            generation.get("source_entities", ())
        )
        data["generation"] = GenerationMetadata(**generation)
        data["verification"] = VerificationMetadata(**data["verification"])
        return cls(**data)
