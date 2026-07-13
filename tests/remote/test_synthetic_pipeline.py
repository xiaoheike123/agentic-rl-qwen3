import pytest

from agent_rl.data.synthetic.generators import GENERATORS
from agent_rl.data.synthetic.overlap import BenchmarkOverlapGuard
from agent_rl.data.synthetic.verifier import verify_oracle_task
from agent_rl.envs.tau_env import TauEnv, TauEnvConfig


@pytest.mark.parametrize("domain", ["airline", "retail", "telecom"])
def test_first_non_overlapping_candidate_is_oracle_valid(domain: str) -> None:
    guard = BenchmarkOverlapGuard(domain)
    candidates = GENERATORS[domain](42)
    accepted = [candidate for candidate in candidates if guard.inspect(candidate.task).passed]
    assert accepted, f"no non-overlapping synthetic candidates for {domain}"

    verification = verify_oracle_task(domain, accepted[0].task)

    assert verification.oracle_verified, verification.error
    assert verification.database_changed


def test_tau_env_resolves_embedded_task_without_official_registry_lookup() -> None:
    candidate = GENERATORS["airline"](7)[0]
    environment = TauEnv(
        TauEnvConfig(
            domain="airline",
            task_id=candidate.task.id,
            task_data=candidate.task.model_dump(mode="json"),
        )
    )

    assert environment._env._get_task().id == candidate.task.id
