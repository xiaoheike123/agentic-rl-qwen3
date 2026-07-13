"""Domain-specific procedural generators."""

from agent_rl.data.synthetic.generators.airline import generate_airline_candidates
from agent_rl.data.synthetic.generators.common import GeneratedCandidate
from agent_rl.data.synthetic.generators.retail import generate_retail_candidates
from agent_rl.data.synthetic.generators.telecom import generate_telecom_candidates


GENERATORS = {
    "airline": generate_airline_candidates,
    "retail": generate_retail_candidates,
    "telecom": generate_telecom_candidates,
}

__all__ = ["GENERATORS", "GeneratedCandidate"]
