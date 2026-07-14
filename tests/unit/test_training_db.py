from __future__ import annotations

from pathlib import Path

from agent_rl.data.synthetic.training_db import (
    TrainingDatabaseConfig,
    build_training_databases,
    database_identifiers,
    load_training_database,
    training_database_fingerprint,
    validate_training_databases,
)
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.utils import AIRLINE_DB_PATH
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.utils import RETAIL_DB_PATH
from tau2.domains.telecom.data_model import TelecomDB
from tau2.domains.telecom.utils import TELECOM_DB_PATH


def _config(root: Path) -> TrainingDatabaseConfig:
    return TrainingDatabaseConfig(
        output_root=root,
        domains=("airline", "retail", "telecom"),
        seed=43,
        telecom_clone_factor=2,
    )


def test_default_training_database_scope_is_airline_and_retail(tmp_path: Path) -> None:
    config = TrainingDatabaseConfig(output_root=tmp_path / "default")
    manifest = build_training_databases(config)

    assert config.domains == ("airline", "retail")
    assert set(manifest["domains"]) == {"airline", "retail"}


def test_training_database_is_reproducible_and_portable(tmp_path: Path) -> None:
    first = build_training_databases(_config(tmp_path / "first"))
    second = build_training_databases(_config(tmp_path / "second"))

    assert training_database_fingerprint(first) == training_database_fingerprint(
        second
    )
    assert validate_training_databases(_config(tmp_path / "first")) == first


def test_training_database_replaces_all_official_identifiers(tmp_path: Path) -> None:
    root = tmp_path / "training-db"
    manifest = build_training_databases(_config(root))
    sources = {
        "airline": FlightDB.load(AIRLINE_DB_PATH),
        "retail": RetailDB.load(RETAIL_DB_PATH),
        "telecom": TelecomDB.load(TELECOM_DB_PATH),
    }

    for domain, source in sources.items():
        training = load_training_database(root, domain)
        assert not (
            database_identifiers(domain, source)
            & database_identifiers(domain, training)
        )
        assert manifest["domains"][domain]["official_identifier_overlap"] == 0


def test_telecom_database_expands_entities_deterministically(tmp_path: Path) -> None:
    root = tmp_path / "training-db"
    build_training_databases(_config(root))
    source = TelecomDB.load(TELECOM_DB_PATH)
    training = load_training_database(root, "telecom")

    assert len(training.customers) == len(source.customers) * 2
    assert len(training.lines) == len(source.lines) * 2
    assert len(training.bills) == len(source.bills) * 2
    assert len(training.devices) == len(source.devices) * 2


def test_each_episode_receives_an_independent_database_copy(tmp_path: Path) -> None:
    root = tmp_path / "training-db"
    build_training_databases(_config(root))
    first = load_training_database(root, "airline")
    second = load_training_database(root, "airline")
    reservation_id = next(
        key
        for key, reservation in first.reservations.items()
        if reservation.status != "cancelled"
    )
    original_status = second.reservations[reservation_id].status

    first.reservations[reservation_id].status = "cancelled"

    third = load_training_database(root, "airline")
    assert first is not second
    assert second.reservations[reservation_id].status == original_status
    assert third.reservations[reservation_id].status == original_status
