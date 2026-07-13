from __future__ import annotations

import hashlib

from tau2.domains.airline.environment import get_environment
from tau2.domains.airline.utils import AIRLINE_DB_PATH


def _file_hash() -> str:
    return hashlib.sha256(AIRLINE_DB_PATH.read_bytes()).hexdigest()


def test_airline_environments_have_isolated_in_memory_databases() -> None:
    file_hash_before = _file_hash()
    first = get_environment()
    second = get_environment()

    initial_hash = first.get_db_hash()
    assert initial_hash == second.get_db_hash()
    assert first.tools.db is not second.tools.db

    reservation = next(
        value
        for value in first.tools.db.reservations.values()
        if value.status != "cancelled"
    )
    first.make_tool_call(
        tool_name="cancel_reservation",
        reservation_id=reservation.reservation_id,
    )

    assert first.get_db_hash() != initial_hash
    assert second.get_db_hash() == initial_hash
    assert get_environment().get_db_hash() == initial_hash
    assert _file_hash() == file_hash_before
