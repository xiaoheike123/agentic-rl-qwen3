"""Consistent standard-library logging setup for project CLIs."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    resolved = getattr(logging, level.upper(), None)
    if not isinstance(resolved, int):
        raise ValueError(f"unknown logging level {level!r}")
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
