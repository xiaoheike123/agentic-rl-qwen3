"""Run one policy-controlled tau2 episode and return its trajectory."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from agent_rl.data.schemas import (
    EpisodeRecord,
    EpisodeStatus,
    RewardRecord,
    ToolCallRecord,
    TurnRecord,
)
from agent_rl.envs.action_parser import (
    TAU_STOP_TOOL_NAME,
    is_tau_control_tool,
)
from agent_rl.envs.tau_env import TauEnv, TauEnvConfig, TauTransition
from agent_rl.prompts.action_prompt import build_action_messages
from agent_rl.prompts.context_compression import (
    ContextCompressionResult,
    LightweightContextCompressor,
)
from agent_rl.prompts.tool_render import render_tool_schemas
from agent_rl.rollout.vllm_policy import (
    PolicyOutput,
    PolicyResponseError,
    VLLMPolicy,
)


class EpisodeDataError(RuntimeError):
    """Raised when tau2 returns incomplete or inconsistent trajectory data."""


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """Identity and environment settings for one episode."""

    episode_id: str
    group_id: str
    env_config: TauEnvConfig
    sample_index: int = 0
    trial_id: int = 0
    seed: int | None = None
    policy_version: int | None = None

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("episode_id must not be empty")
        if not self.group_id.strip():
            raise ValueError("group_id must not be empty")
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if self.trial_id < 0:
            raise ValueError("trial_id must be non-negative")


@dataclass(frozen=True, slots=True)
class EpisodeWorkerConfig:
    """Runtime behavior shared by episode workers."""

    success_threshold: float = 1.0
    strict_tool_result_alignment: bool = True
    keep_simulation_run: bool = True
    raise_on_error: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.success_threshold <= 1.0:
            raise ValueError("success_threshold must be between zero and one")


def _decode_json_object(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return deepcopy(value)

    if not isinstance(value, str):
        raise EpisodeDataError(
            f"{field_name} must be a dictionary or JSON string, "
            f"got {type(value).__name__}"
        )

    if not value.strip():
        return {}

    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise EpisodeDataError(
            f"{field_name} contains invalid JSON: {error.msg}"
        ) from error

    if not isinstance(decoded, dict):
        raise EpisodeDataError(
            f"{field_name} must decode to an object, got {type(decoded).__name__}"
        )

    return decoded


def _extract_assistant_message(output: PolicyOutput) -> dict[str, Any]:
    choices = output.raw_response.get("choices")

    if not isinstance(choices, list) or not choices:
        raise EpisodeDataError("vLLM response contains no choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise EpisodeDataError("vLLM first choice is not an object")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise EpisodeDataError("vLLM first choice contains no assistant message")

    return deepcopy(message)


def _build_tool_call_records(output: PolicyOutput) -> list[ToolCallRecord]:
    records: list[ToolCallRecord] = []

    for call in output.tool_calls:
        if not call.tool_call_id.strip():
            raise EpisodeDataError("vLLM tool call contains no call ID")

        records.append(
            ToolCallRecord(
                call_id=call.tool_call_id,
                name=call.name,
                arguments=deepcopy(call.arguments),
                is_control=is_tau_control_tool(call.name),
            )
        )

    return records


def _iter_tool_result_messages(
    message: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Yield leaf ToolMessages from tau2 tool-message containers."""

    nested = message.get("tool_messages")
    if nested is not None:
        if not isinstance(nested, list):
            raise EpisodeDataError("tool_messages must be a list")

        for child in nested:
            if not isinstance(child, dict):
                raise EpisodeDataError(
                    "tool_messages contains a non-object message"
                )
            yield from _iter_tool_result_messages(child)
        return

    if message.get("role") == "tool":
        yield message


def _collect_tool_results(
    simulation_run: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    messages = simulation_run.get("messages") or []

    if not isinstance(messages, list):
        raise EpisodeDataError("simulation_run.messages must be a list")

    results: dict[str, dict[str, Any]] = {}

    for top_level_message in messages:
        if not isinstance(top_level_message, dict):
            raise EpisodeDataError("simulation_run contains a non-object message")

        for message in _iter_tool_result_messages(top_level_message):
            if message.get("requestor", "assistant") != "assistant":
                continue

            call_id = message.get("id")
            if not isinstance(call_id, str) or not call_id.strip():
                # tau2 may emit assistant-addressed tool-like leaves that are
                # not responses to any policy tool call. They cannot hydrate a
                # trajectory record. Ignore them here; strict hydration below
                # still rejects any real policy call whose result is missing.
                continue

            is_error = message.get("error", False)
            if not isinstance(is_error, bool):
                raise EpisodeDataError(
                    f"tool result {call_id!r} has a non-boolean error field"
                )

            content = deepcopy(message.get("content"))
            error_message: str | None = None

            if is_error:
                error_message = (
                    str(content) if content is not None else "tool call failed"
                )

            candidate = {
                "result": content,
                "error": error_message,
            }

            existing = results.get(call_id)
            if existing is not None:
                if existing == candidate:
                    continue

                # tau2 can emit duplicated tool messages for the same call id.
                # Prefer a successful result; otherwise keep the first result
                # so trajectory hydration remains deterministic.
                if (
                    existing.get("error") is not None
                    and candidate.get("error") is None
                ):
                    results[call_id] = candidate

                continue

            results[call_id] = candidate

    return results


def _hydrate_tool_results(
    episode: EpisodeRecord,
    simulation_run: dict[str, Any],
    *,
    strict: bool,
) -> list[str]:
    results = _collect_tool_results(simulation_run)
    termination_reason = simulation_run.get("termination_reason")
    seen_call_ids: set[str] = set()
    missing_call_ids: list[str] = []

    for turn in episode.turns:
        for call in turn.tool_calls:
            if call.call_id in seen_call_ids:
                raise EpisodeDataError(
                    f"trajectory contains duplicate tool call ID {call.call_id!r}"
                )

            seen_call_ids.add(call.call_id)
            tool_result = results.get(call.call_id)

            if call.is_control and call.name != TAU_STOP_TOOL_NAME:
                raise EpisodeDataError(
                    f"unsupported tau2 control tool {call.name!r}"
                )

            if (
                call.is_control
                and tool_result is None
                and termination_reason == "agent_stop"
            ):
                call.result = {"termination_reason": "agent_stop"}
                call.error = None
                call.result_received = True
                continue

            if tool_result is None:
                missing_call_ids.append(call.call_id)
                continue

            call.result = tool_result["result"]
            call.error = tool_result["error"]
            call.result_received = True

    if strict and missing_call_ids:
        missing = ", ".join(repr(item) for item in missing_call_ids)
        raise EpisodeDataError(f"tau2 returned no results for tool calls: {missing}")

    return missing_call_ids


def _build_reward_record(transition: TauTransition) -> RewardRecord:
    reward_info = _decode_json_object(
        transition.evaluator_info.get("reward_info"),
        field_name="reward_info",
    )

    reported_reward = reward_info.get("reward")

    if reported_reward is not None:
        if isinstance(reported_reward, bool) or not isinstance(
            reported_reward, (int, float)
        ):
            raise EpisodeDataError("reward_info.reward must be numeric")

        if abs(float(reported_reward) - transition.reward) > 1e-8:
            raise EpisodeDataError(
                "transition reward does not match reward_info.reward"
            )

    raw_components = reward_info.get("reward_breakdown") or {}

    if not isinstance(raw_components, dict):
        raise EpisodeDataError("reward_info.reward_breakdown must be an object")

    components: dict[str, float] = {}

    for name, value in raw_components.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EpisodeDataError(f"reward component {name!r} must be numeric")

        components[str(name)] = float(value)

    return RewardRecord(
        outcome=transition.reward,
        process=None,
        total=transition.reward,
        components=components,
        evaluator_info=reward_info,
    )


def _get_termination_reason(
    simulation_run: dict[str, Any],
    transition: TauTransition,
) -> str:
    reason = simulation_run.get("termination_reason")

    if isinstance(reason, str) and reason.strip():
        return reason.strip()

    if transition.truncated:
        return "truncated"

    if transition.terminated:
        return "terminated"

    raise EpisodeDataError(
        "cannot determine termination reason from unfinished transition"
    )


def _get_failure_reason(error: Exception) -> str:
    if isinstance(error, PolicyResponseError):
        return "policy_response_error"

    if isinstance(error, EpisodeDataError):
        return "trajectory_data_error"

    return "rollout_error"


def _format_error(error: Exception) -> str:
    message = str(error).strip()

    if message:
        return f"{type(error).__name__}: {message}"

    return type(error).__name__


def _compression_info(
    result: ContextCompressionResult,
) -> dict[str, Any]:
    return {
        "applied": result.applied,
        "original_chars": result.original_chars,
        "compressed_chars": result.compressed_chars,
        "original_events": result.original_events,
        "retained_events": result.retained_events,
        "dropped_events": result.dropped_events,
        "compression_ratio": result.compression_ratio,
    }


class EpisodeWorker:
    """Execute one complete policy-controlled tau2 episode."""

    def __init__(
        self,
        policy: VLLMPolicy,
        config: EpisodeWorkerConfig | None = None,
        context_compressor: LightweightContextCompressor | None = None,
    ) -> None:
        self.policy = policy
        self.config = config or EpisodeWorkerConfig()
        self.context_compressor = context_compressor or LightweightContextCompressor()

    def run(self, spec: EpisodeSpec) -> EpisodeRecord:
        episode = EpisodeRecord(
            episode_id=spec.episode_id,
            group_id=spec.group_id,
            domain=spec.env_config.domain,
            task_id=spec.env_config.task_id,
            model=self.policy.config.model,
            sample_index=spec.sample_index,
            trial_id=spec.trial_id,
            seed=spec.seed,
            policy_version=spec.policy_version,
            user_model=spec.env_config.user_llm,
        )

        try:
            self._run_to_completion(episode, spec)
        except Exception as error:
            if episode.status is EpisodeStatus.RUNNING:
                episode.fail(
                    error=_format_error(error),
                    termination_reason=_get_failure_reason(error),
                )

            if self.config.raise_on_error:
                raise

        return episode

    def _run_to_completion(
        self,
        episode: EpisodeRecord,
        spec: EpisodeSpec,
    ) -> None:
        env = TauEnv(spec.env_config)
        reset = env.reset(seed=spec.seed)

        if not reset.observation.strip():
            raise EpisodeDataError("tau2 returned an empty initial observation")

        domain_policy = reset.info.get("policy")
        if not isinstance(domain_policy, str) or not domain_policy.strip():
            raise EpisodeDataError("tau2 reset info contains no domain policy")

        tools = reset.info.get("tools")
        if not isinstance(tools, (list, tuple)):
            raise EpisodeDataError("tau2 reset info contains no tool sequence")

        tool_schemas = render_tool_schemas(tools)

        episode.metadata["task_source"] = (
            "synthetic" if spec.env_config.task_data is not None else "official"
        )
        episode.metadata["domain_policy"] = domain_policy
        episode.metadata["tool_schemas"] = deepcopy(tool_schemas)
        episode.metadata["all_messages_as_observation"] = (
            spec.env_config.all_messages_as_observation
        )
        episode.metadata["perturbation"] = spec.env_config.perturbation_name or "clean"
        episode.metadata["context_compression_config"] = asdict(
            self.context_compressor.config
        )
        episode.metadata["initial_db_hash"] = reset.initial_db_hash

        observation = reset.observation
        final_transition: TauTransition | None = None

        while final_transition is None:
            compression = self.context_compressor.compress(observation)

            messages = build_action_messages(
                domain_policy=domain_policy,
                observation=compression.text,
            )

            if "prompt_sha256" not in episode.metadata:
                system_prompt = messages[0]["content"]
                episode.metadata["prompt_sha256"] = hashlib.sha256(
                    system_prompt.encode("utf-8")
                ).hexdigest()

            output = self.policy.generate(
                messages=messages,
                tools=tool_schemas,
            )

            transition = env.step(output.action)

            turn = TurnRecord(
                turn_index=len(episode.turns),
                observation=observation,
                prompt_messages=deepcopy(messages),
                action=output.action,
                next_observation=transition.observation,
                assistant_message=_extract_assistant_message(output),
                tool_calls=_build_tool_call_records(output),
                environment_reward=transition.reward,
                terminated=transition.terminated,
                truncated=transition.truncated,
                info={
                    "finish_reason": output.finish_reason,
                    "prompt_tokens": output.prompt_tokens,
                    "completion_tokens": output.completion_tokens,
                    "context_compression": _compression_info(compression),
                    "action_parse_valid": transition.info.get(
                        "agent_rl_action_parse_valid", True
                    ),
                    "action_parse_error": transition.info.get(
                        "agent_rl_action_parse_error"
                    ),
                },
            )

            episode.append_turn(turn)

            if transition.done:
                final_transition = transition
            else:
                observation = transition.observation

        self._finalize_episode(episode, final_transition)

    def _finalize_episode(
        self,
        episode: EpisodeRecord,
        transition: TauTransition,
    ) -> None:
        episode.metadata["final_db_hash"] = transition.db_hash

        simulation_run = _decode_json_object(
            transition.evaluator_info.get("simulation_run"),
            field_name="simulation_run",
        )

        missing_call_ids = _hydrate_tool_results(
            episode,
            simulation_run,
            strict=self.config.strict_tool_result_alignment,
        )

        if missing_call_ids:
            episode.metadata["missing_tool_result_ids"] = missing_call_ids

        if self.config.keep_simulation_run:
            episode.metadata["simulation_run"] = simulation_run
        else:
            episode.metadata["simulation_run_id"] = simulation_run.get("id")

        reward = _build_reward_record(transition)
        termination_reason = _get_termination_reason(
            simulation_run,
            transition,
        )
        outcome = reward.outcome

        if outcome is None:
            raise EpisodeDataError("final outcome reward is missing")

        episode.finish(
            reward=reward,
            success=outcome >= self.config.success_threshold,
            termination_reason=termination_reason,
        )
