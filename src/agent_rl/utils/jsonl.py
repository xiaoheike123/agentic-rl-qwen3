"""Durable JSONL persistence for trajectory records."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

from agent_rl.data.schemas import (
    EpisodeRecord,
    EpisodeStatus,
)


class JsonlFormatError(ValueError):
    """Raised when a JSONL file contains an invalid record."""


class DuplicateEpisodeError(ValueError):
    """Raised when an episode ID already exists in the store."""


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def encode_json_line(value: Any) -> str:
    """Serialize one value as compact, strict JSON."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        default=_json_default,
    )


def iter_json_objects(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield validated JSON objects from a JSONL file."""

    file_path = Path(path)

    if not file_path.exists():
        return

    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise JsonlFormatError(
                    f"Invalid JSON in {file_path} at line {line_number}: {error.msg}"
                ) from error

            if not isinstance(value, dict):
                raise JsonlFormatError(
                    f"Expected an object in {file_path} at line "
                    f"{line_number}, got {type(value).__name__}"
                )

            yield value


class JsonlEpisodeStore:
    """Append and restore finalized episodes from one JSONL file."""

    def __init__(
        self,
        path: str | Path,
        *,
        durable: bool = False,
    ) -> None:
        self.path = Path(path)
        self.durable = durable
        self._write_lock = Lock()
        self._known_episode_ids: set[str] | None = None

    def append(self, episode: EpisodeRecord) -> None:
        """Append one finalized episode and reject duplicate IDs."""

        if episode.status is EpisodeStatus.RUNNING:
            raise ValueError("a running episode cannot be persisted")

        if episode.finished_at is None:
            raise ValueError("a persisted episode must have finished_at")

        line = encode_json_line(episode.to_dict())

        with self._write_lock:
            known_ids = self._get_known_episode_ids_unlocked()

            if episode.episode_id in known_ids:
                raise DuplicateEpisodeError(
                    f"episode {episode.episode_id!r} already exists in {self.path}"
                )

            self.path.parent.mkdir(parents=True, exist_ok=True)

            with self.path.open(
                "a",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                handle.write(line)
                handle.write("\n")
                handle.flush()

                if self.durable:
                    os.fsync(handle.fileno())

            known_ids.add(episode.episode_id)

    def iter_episodes(self) -> Iterator[EpisodeRecord]:
        """Yield validated episode records in file order."""

        for value in iter_json_objects(self.path):
            try:
                yield EpisodeRecord.from_dict(value)
            except (TypeError, ValueError) as error:
                episode_id = value.get("episode_id", "<unknown>")
                raise JsonlFormatError(
                    f"Invalid episode {episode_id!r} in {self.path}: {error}"
                ) from error

    def load(self) -> list[EpisodeRecord]:
        return list(self.iter_episodes())

    def episode_ids(self) -> set[str]:
        with self._write_lock:
            return set(self._get_known_episode_ids_unlocked())

    def completed_episode_ids(self) -> set[str]:
        return {
            episode.episode_id
            for episode in self.iter_episodes()
            if episode.status is EpisodeStatus.COMPLETED
        }

    def failed_episode_ids(self) -> set[str]:
        return {
            episode.episode_id
            for episode in self.iter_episodes()
            if episode.status is EpisodeStatus.FAILED
        }

    def refresh_index(self) -> None:
        """Reload the ID index after an external read-only file transfer."""

        with self._write_lock:
            self._known_episode_ids = None
            self._get_known_episode_ids_unlocked()

    def _get_known_episode_ids_unlocked(self) -> set[str]:
        if self._known_episode_ids is not None:
            return self._known_episode_ids

        known_ids: set[str] = set()

        for value in iter_json_objects(self.path):
            episode_id = value.get("episode_id")

            if not isinstance(episode_id, str) or not episode_id.strip():
                raise JsonlFormatError(f"Record in {self.path} has no valid episode_id")

            if episode_id in known_ids:
                raise JsonlFormatError(
                    f"Duplicate episode ID {episode_id!r} in {self.path}"
                )

            known_ids.add(episode_id)

        self._known_episode_ids = known_ids
        return known_ids

    def __len__(self) -> int:
        return sum(1 for _ in iter_json_objects(self.path))
