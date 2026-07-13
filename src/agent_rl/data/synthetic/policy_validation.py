"""Clean-room policy validation for procedurally generated tasks."""

from __future__ import annotations

from typing import Any

from agent_rl.data.synthetic.generators.common import GeneratedCandidate


def validate_candidate_policy(
    candidate: GeneratedCandidate,
    database: Any | None = None,
) -> None:
    """Reject candidates that violate a domain policy before Oracle execution."""

    if candidate.domain == "airline":
        from agent_rl.data.synthetic.airline_policy import validate_airline_candidate

        validate_airline_candidate(candidate, database)
        return
    if candidate.domain == "retail":
        from agent_rl.data.synthetic.retail_policy import validate_retail_candidate

        validate_retail_candidate(candidate, database)
        return
    if candidate.domain == "telecom":
        from agent_rl.data.synthetic.telecom_policy import validate_telecom_candidate

        validate_telecom_candidate(candidate, database)
        return
    raise ValueError(
        f"clean-room policy validator is not implemented for {candidate.domain}"
    )
