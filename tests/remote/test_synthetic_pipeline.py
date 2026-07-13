import pytest

from agent_rl.data.synthetic.generators import GENERATORS
from agent_rl.data.synthetic.training_db import (
    TrainingDatabaseConfig,
    build_training_databases,
    load_training_database,
)
from agent_rl.data.synthetic.verifier import verify_oracle_task
from agent_rl.envs.tau_env import TauEnv, TauEnvConfig


@pytest.fixture(scope="session")
def training_database_root(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("training-database")
    build_training_databases(
        TrainingDatabaseConfig(
            output_root=root,
            seed=43,
            telecom_clone_factor=2,
        )
    )
    return root


@pytest.mark.parametrize("domain", ["airline", "retail", "telecom"])
def test_first_candidate_is_oracle_valid(
    domain: str,
    training_database_root,
) -> None:
    database = load_training_database(training_database_root, domain)
    candidates = GENERATORS[domain](42, database)
    assert candidates, f"no synthetic candidates for {domain}"

    verification = verify_oracle_task(domain, candidates[0].task, database)

    assert verification.oracle_verified, verification.error
    assert verification.database_changed


def test_tau_env_resolves_embedded_task_without_official_registry_lookup(
    training_database_root,
) -> None:
    database = load_training_database(training_database_root, "airline")
    candidate = GENERATORS["airline"](7, database)[0]
    environment = TauEnv(
        TauEnvConfig(
            domain="airline",
            task_id=candidate.task.id,
            task_data=candidate.task.model_dump(mode="json"),
            database_override=database,
        )
    )

    assert environment._env._get_task().id == candidate.task.id


def test_every_airline_template_has_an_executable_oracle(
    training_database_root,
) -> None:
    database = load_training_database(training_database_root, "airline")
    by_template = {}
    for candidate in GENERATORS["airline"](42, database):
        by_template.setdefault(candidate.generation.template, candidate)

    assert len(by_template) >= 10
    for template, candidate in by_template.items():
        verification = verify_oracle_task("airline", candidate.task, database)
        assert verification.oracle_verified, f"{template}: {verification.error}"


def test_every_retail_template_has_an_executable_oracle(
    training_database_root,
) -> None:
    database = load_training_database(training_database_root, "retail")
    by_template = {}
    for candidate in GENERATORS["retail"](42, database):
        by_template.setdefault(candidate.generation.template, candidate)

    assert len(by_template) >= 16
    for template, candidate in by_template.items():
        verification = verify_oracle_task("retail", candidate.task, database)
        assert verification.oracle_verified, f"{template}: {verification.error}"


def test_every_telecom_template_has_an_executable_oracle(
    training_database_root,
) -> None:
    database = load_training_database(training_database_root, "telecom")
    by_template = {}
    for candidate in GENERATORS["telecom"](2_000_049, database):
        by_template.setdefault(candidate.generation.template, candidate)

    assert len(by_template) >= 25
    for template, candidate in by_template.items():
        verification = verify_oracle_task("telecom", candidate.task, database)
        assert verification.oracle_verified, f"{template}: {verification.error}"
