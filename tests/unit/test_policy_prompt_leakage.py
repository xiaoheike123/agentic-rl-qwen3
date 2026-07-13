from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from agent_rl.data.synthetic.generators import GENERATORS
from agent_rl.envs.tau_env import TauEnvConfig, TauReset, TauTransition
from agent_rl.rollout.episode_worker import EpisodeSpec, EpisodeWorker
from agent_rl.rollout.vllm_policy import PolicyOutput, VLLMPolicyConfig


ORACLE_SENTINEL = "ORACLE_SECRET_7F92_DO_NOT_EXPOSE"


class RecordingPolicy:
    config = VLLMPolicyConfig(model="recording-policy")

    def __init__(self) -> None:
        self.requests: list[list[dict[str, Any]]] = []

    def generate(self, messages: Any, tools: Any) -> PolicyOutput:
        del tools
        self.requests.append(deepcopy(messages))
        content = f"public response {len(self.requests)}"
        return PolicyOutput(
            action=content,
            content=content,
            tool_calls=(),
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=3,
            raw_response={
                "choices": [{"message": {"role": "assistant", "content": content}}]
            },
        )


class TwoTurnEnvironment:
    def __init__(self, config: TauEnvConfig) -> None:
        self.config = config
        self.steps = 0

    def reset(self, seed: int | None = None) -> TauReset:
        del seed
        return TauReset(
            observation="The user made a public request.",
            info={"policy": "Public policy.", "tools": []},
            evaluator_info={"simulation_run": "{}"},
        )

    def step(self, action: str) -> TauTransition:
        assert action.startswith("public response")
        self.steps += 1
        done = self.steps == 2
        return TauTransition(
            observation="Another public user message.",
            reward=float(done),
            terminated=done,
            truncated=False,
            info={
                "agent_rl_action_parse_valid": True,
                "agent_rl_action_parse_error": None,
            },
            evaluator_info={
                "simulation_run": {
                    "termination_reason": "user_stop",
                    "messages": [],
                },
                "reward_info": {
                    "reward": float(done),
                    "reward_breakdown": {},
                    "info": {"oracle": ORACLE_SENTINEL},
                },
            },
        )


def test_oracle_answer_never_reaches_any_policy_message(monkeypatch: Any) -> None:
    task = GENERATORS["airline"](7)[0].task.model_dump(mode="json")
    task["evaluation_criteria"]["actions"][0]["arguments"][
        "reservation_id"
    ] = ORACLE_SENTINEL
    assert ORACLE_SENTINEL in json.dumps(task)

    monkeypatch.setattr(
        "agent_rl.rollout.episode_worker.TauEnv", TwoTurnEnvironment
    )
    policy = RecordingPolicy()
    episode = EpisodeWorker(policy=policy).run(  # type: ignore[arg-type]
        EpisodeSpec(
            episode_id="leakage-test",
            group_id="leakage-test",
            env_config=TauEnvConfig(
                domain="airline", task_id=task["id"], task_data=task
            ),
        )
    )

    assert episode.success
    assert len(policy.requests) == 2
    model_input = json.dumps(policy.requests, ensure_ascii=False)
    assert ORACLE_SENTINEL not in model_input
    assert "evaluation_criteria" not in model_input
    assert "initial_state" not in model_input
    assert "synthetic_task" not in model_input
