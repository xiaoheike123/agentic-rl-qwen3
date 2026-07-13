"""JSONL persistence for validated synthetic tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from agent_rl.data.synthetic.schema import SyntheticTaskRecord
from agent_rl.utils.jsonl import encode_json_line, iter_json_objects


def write_records(
    path: str | Path,
    records: list[SyntheticTaskRecord],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(encode_json_line(record.to_dict()))
            stream.write("\n")
    temporary.replace(output)


def iter_records(path: str | Path) -> Iterator[SyntheticTaskRecord]:
    for value in iter_json_objects(path):
        yield SyntheticTaskRecord.from_dict(value)


def load_records(path: str | Path) -> list[SyntheticTaskRecord]:
    return list(iter_records(path))
