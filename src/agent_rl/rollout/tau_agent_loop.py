"""verl-native on-policy agent loop for tau2 environments."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from agent_rl.data.schemas import (
    EpisodeRecord,
    RewardRecord,
    TokenTrace,
    ToolCallRecord,
    TurnRecord,
)
from agent_rl.data.synthetic.training_db import load_training_database
from agent_rl.envs.action_parser import (
    ActionFormatError,
    ModelToolCall,
    is_tau_control_tool,
    to_tau_action,
)
from agent_rl.envs.tau_env import (
    TauEnv,
    TauEnvConfig,
    TauInfrastructureError,
)
from agent_rl.prompts.action_prompt import (
    OBSERVATION_PROMPT,
    build_action_messages,
)
from agent_rl.prompts.context_compression import (
    ContextCompressionConfig,
    LightweightContextCompressor,
)
from agent_rl.prompts.tool_render import render_tool_schemas
from agent_rl.rewards.process_reward import (
    EnvironmentProcessReward,
    ProcessRewardConfig,
)
from agent_rl.rewards.reward_mixer import RewardMixer, RewardMixerConfig
from agent_rl.rollout.episode_worker import (
    _decode_json_object,
    _get_termination_reason,
    _hydrate_tool_results,
)

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)
from verl.experimental.agent_loop.tool_parser import ToolParser


logger = logging.getLogger(__name__)

_worker_episode_semaphore: asyncio.Semaphore | None = None
_worker_episode_limit: int | None = None


def _get_worker_episode_semaphore(limit: int) -> asyncio.Semaphore:
    global _worker_episode_limit, _worker_episode_semaphore

    if limit <= 0:
        raise ValueError("max_concurrent_episodes_per_worker must be positive")
    if _worker_episode_semaphore is None:
        _worker_episode_limit = limit
        _worker_episode_semaphore = asyncio.Semaphore(limit)
    elif _worker_episode_limit != limit:
        raise RuntimeError(
            "all TauAgentLoop instances in one worker must use the same "
            "max_concurrent_episodes_per_worker"
        )
    return _worker_episode_semaphore


@dataclass(frozen=True, slots=True)
class TauAgentLoopSettings:
    max_steps: int = 30
    user_llm: str = "deepseek/deepseek-v4-flash"
    user_llm_args: dict[str, Any] | None = None
    evaluator_llm: str = "deepseek/deepseek-v4-pro"
    evaluator_llm_args: dict[str, Any] | None = None
    all_messages_as_observation: bool = False
    tool_parser: str = "hermes"
    outcome_weight: float = 1.0
    process_weight: float = 0.0
    process_config: dict[str, Any] | None = None
    enable_hindsight_credit: bool = False
    hindsight_config: dict[str, Any] | None = None
    context_max_chars: int = 24_000
    max_action_tokens: int = 2_048
    training_database_root: str | None = None
    max_episode_attempts: int = 3
    retry_backoff_seconds: float = 1.0
    max_concurrent_episodes_per_worker: int = 1

    def __post_init__(self) -> None:
        if self.max_episode_attempts <= 0:
            raise ValueError("max_episode_attempts must be positive")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must not be negative")
        if self.max_concurrent_episodes_per_worker <= 0:
            raise ValueError(
                "max_concurrent_episodes_per_worker must be positive"
            )


class TauAgentLoop(AgentLoopBase):
    """Generate a complete tau2 trajectory using verl's current policy."""

    def __init__(
        self, *args: Any, settings: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.settings = TauAgentLoopSettings(**(settings or {}))
        self.tool_parser = ToolParser.get_tool_parser(
            self.settings.tool_parser,
            self.tokenizer,
        )
        self.compressor = LightweightContextCompressor(
            ContextCompressionConfig(max_chars=self.settings.context_max_chars)
        )
        self.process_reward = EnvironmentProcessReward(
            ProcessRewardConfig(**(self.settings.process_config or {}))
        )
        self.reward_mixer = RewardMixer(
            RewardMixerConfig(
                outcome_weight=self.settings.outcome_weight,
                process_weight=self.settings.process_weight,
            )
        )

    async def run(
        self, sampling_params: dict[str, Any], **kwargs: Any
    ) -> AgentLoopOutput:
        semaphore = _get_worker_episode_semaphore(
            self.settings.max_concurrent_episodes_per_worker
        )
        async with semaphore:
            for attempt in range(1, self.settings.max_episode_attempts + 1):
                try:
                    return await self._run_episode(sampling_params, **kwargs)
                except TauInfrastructureError as error:
                    if attempt >= self.settings.max_episode_attempts:
                        logger.error(
                            "tau2 episode exhausted infrastructure retries: "
                            "domain=%s task_id=%s stage=%s attempts=%d",
                            error.domain,
                            error.task_id,
                            error.stage,
                            attempt,
                        )
                        raise
                    delay = self.settings.retry_backoff_seconds * (
                        2 ** (attempt - 1)
                    )
                    logger.warning(
                        "retrying tau2 episode after infrastructure failure: "
                        "domain=%s task_id=%s stage=%s attempt=%d/%d delay=%.1fs",
                        error.domain,
                        error.task_id,
                        error.stage,
                        attempt,
                        self.settings.max_episode_attempts,
                        delay,
                    )
                    if delay:
                        await asyncio.sleep(delay)

        raise RuntimeError("tau2 retry loop exited without an episode result")

    async def _run_episode(
        self, sampling_params: dict[str, Any], **kwargs: Any
    ) -> AgentLoopOutput:
        domain = _required_string(kwargs, "domain")
        task_id = _required_string(kwargs, "task_id")
        synthetic_task = kwargs.get("synthetic_task")
        if synthetic_task is not None and not isinstance(synthetic_task, dict):
            raise TypeError("dataset field 'synthetic_task' must be an object")
        database_override = None
        database_source = "official_evaluation"
        if synthetic_task is not None:
            if not self.settings.training_database_root:
                raise ValueError(
                    "synthetic rollout requires settings.training_database_root"
                )
            database_override = load_training_database(
                self.settings.training_database_root,
                domain,
            )
            database_source = "pseudonymized_training"
        group_id = str(kwargs.get("uid", task_id))
        sample_index = int(kwargs.get("rollout_n", 0))
        base_seed = kwargs.get("seed")
        episode_seed = (
            int(base_seed) + sample_index if base_seed is not None else None
        )

        env = TauEnv(
            TauEnvConfig(
                domain=domain,
                task_id=task_id,
                max_steps=self.settings.max_steps,
                user_llm=self.settings.user_llm,
                user_llm_args=dict(self.settings.user_llm_args or {}),
                evaluator_llm=self.settings.evaluator_llm,
                evaluator_llm_args=dict(self.settings.evaluator_llm_args or {}),
                all_messages_as_observation=self.settings.all_messages_as_observation,
                task_data=synthetic_task,
                database_override=database_override,
            )
        )
        reset = await self.loop.run_in_executor(
            None,
            lambda: env.reset(seed=episode_seed),
        )
        policy = reset.info.get("policy")
        tools = reset.info.get("tools")
        if not isinstance(policy, str) or not policy.strip():
            raise ValueError("tau2 reset returned no domain policy")
        if not isinstance(tools, (list, tuple)):
            raise ValueError("tau2 reset returned no tool definitions")

        tool_schemas = render_tool_schemas(tools)
        initial_observation = self.compressor.compress(reset.observation).text
        messages = build_action_messages(
            domain_policy=policy,
            observation=initial_observation,
        )
        prompt_ids = await self.apply_chat_template(messages, tools=tool_schemas)
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        turn_token_spans: list[tuple[int, int, int]] = []
        context_rotations = 0
        metrics = AgentLoopMetrics()
        request_id = uuid4().hex

        episode = EpisodeRecord(
            episode_id=f"{task_id}:{sample_index}:{request_id}",
            group_id=group_id,
            domain=domain,
            task_id=task_id,
            model=str(self.config.actor_rollout_ref.model.path),
            sample_index=sample_index,
            seed=episode_seed,
            user_model=self.settings.user_llm,
            metadata={
                "task_source": (
                    "synthetic" if synthetic_task is not None else "official"
                )
            },
        )

        terminated = False
        final_transition = None
        current_observation = reset.observation
        observation_history = [reset.observation]

        while not terminated:
            remaining_tokens = self.rollout_config.response_length - len(response_ids)
            if remaining_tokens <= 0:
                raise RuntimeError("context rotation failed to free response budget")

            started = time.perf_counter()
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids + response_ids,
                sampling_params={
                    **sampling_params,
                    **({"seed": episode_seed} if episode_seed is not None else {}),
                    "stop_token_ids": list(
                        set(
                            (sampling_params.get("stop_token_ids") or [])
                            + self.tool_parser.stop_token_ids
                        )
                    ),
                    "max_tokens": min(
                        int(sampling_params.get("max_tokens", remaining_tokens)),
                        self.settings.max_action_tokens,
                        remaining_tokens,
                    ),
                },
            )
            metrics.generate_sequences += time.perf_counter() - started

            generated_ids = list(output.token_ids)
            if not generated_ids:
                raise RuntimeError("verl rollout server returned no tokens")
            span_start = len(response_ids)
            response_ids.extend(generated_ids)
            response_mask.extend([1] * len(generated_ids))
            response_logprobs.extend(
                list(output.log_probs)
                if output.log_probs
                else [0.0] * len(generated_ids)
            )
            span_end = len(response_ids)
            turn_token_spans.append((len(episode.turns), span_start, span_end))

            content, parsed_calls = await self.tool_parser.extract_tool_calls(
                generated_ids
            )
            if parsed_calls:
                content = None
            else:
                content = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                ).strip()
            try:
                model_calls = tuple(
                    ModelToolCall(
                        name=call.name,
                        arguments=_decode_arguments(call.arguments),
                        tool_call_id=(
                            call.tool_call_id or f"call_{len(episode.turns)}"
                        ),
                    )
                    for call in parsed_calls
                )
                action = to_tau_action(
                    content=content,
                    tool_calls=model_calls,
                )
            except ActionFormatError as error:
                attempted_turns = len(turn_token_spans)

                episode.reward = RewardRecord(
                    outcome=0.0,
                    process=0.0,
                    total=0.0,
                )
                episode.fail(
                    error=f"ActionFormatError: {error}",
                    termination_reason="invalid_action",
                )

                logger.warning(
                    "terminating invalid policy trajectory: "
                    "domain=%s task_id=%s sample_index=%d "
                    "attempted_turns=%d token_ids=%s error=%s",
                    domain,
                    task_id,
                    sample_index,
                    attempted_turns,
                    generated_ids[:16],
                    error,
                )

                return self._build_policy_failure_output(
                    episode=episode,
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_mask=response_mask,
                    response_logprobs=response_logprobs,
                    metrics=metrics,
                    group_id=group_id,
                    sample_index=sample_index,
                    episode_seed=episode_seed,
                    domain=domain,
                    task_id=task_id,
                    database_source=database_source,
                    context_rotations=context_rotations,
                    attempted_turns=attempted_turns,
                    invalid_action_error=str(error),
                    invalid_token_ids=generated_ids,
                    invalid_decoded_output=self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=False,
                    ),
                )

            tool_records = [
                ToolCallRecord(
                    call_id=call.tool_call_id,
                    name=call.name,
                    arguments=dict(call.arguments),
                    is_control=is_tau_control_tool(call.name),
                )
                for call in model_calls
            ]
            started = time.perf_counter()
            transition = await self.loop.run_in_executor(
                None,
                lambda: env.step(action),
            )
            metrics.tool_calls += time.perf_counter() - started

            episode.append_turn(
                TurnRecord(
                    turn_index=len(episode.turns),
                    observation=current_observation,
                    prompt_messages=messages.copy(),
                    action=action,
                    next_observation=transition.observation,
                    assistant_message={
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": [
                            {
                                "id": call.tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(call.arguments),
                                },
                            }
                            for call in model_calls
                        ],
                    },
                    tool_calls=tool_records,
                    token_trace=TokenTrace(
                        response_token_ids=generated_ids,
                        response_logprobs=(
                            list(output.log_probs) if output.log_probs else []
                        ),
                        response_loss_mask=[1] * len(generated_ids),
                    ),
                    environment_reward=transition.reward,
                    terminated=transition.terminated,
                    truncated=transition.truncated,
                    info={
                        "action_parse_valid": transition.info.get(
                            "agent_rl_action_parse_valid", True
                        ),
                        "action_parse_error": transition.info.get(
                            "agent_rl_action_parse_error"
                        ),
                    },
                )
            )
            final_transition = transition
            terminated = transition.done
            if terminated:
                break

            observation_history.append(f"assistant: {action}")
            observation_history.append(transition.observation)
            compressed = self.compressor.compress(transition.observation).text
            next_message: dict[str, Any] = {
                "role": "user",
                "content": OBSERVATION_PROMPT.format(observation=compressed),
            }
            messages.append(next_message)
            environment_ids = await self.apply_chat_template(
                [next_message],
                remove_system_prompt=True,
            )
            environment_ids = self.turn_separator + environment_ids
            if (
                len(response_ids)
                + len(environment_ids)
                + self.settings.max_action_tokens
                > self.rollout_config.response_length
            ):
                compressed_history = self.compressor.compress(
                    "\n".join(observation_history)
                ).text
                messages = build_action_messages(
                    domain_policy=policy,
                    observation=compressed_history,
                )
                prompt_ids = await self.apply_chat_template(
                    messages,
                    tools=tool_schemas,
                )
                response_ids.clear()
                response_mask.clear()
                response_logprobs.clear()
                turn_token_spans.clear()
                context_rotations += 1
                request_id = uuid4().hex
                current_observation = transition.observation
                continue
            response_ids.extend(environment_ids)
            response_mask.extend([0] * len(environment_ids))
            response_logprobs.extend([0.0] * len(environment_ids))
            current_observation = transition.observation

        if final_transition is None:
            raise RuntimeError("tau2 episode produced no transition")

        reward_score, credit_evidence = self._finalize_episode(
            episode,
            final_transition,
            turn_token_spans,
            len(response_ids),
        )
        process_checks = (
            episode.metadata.get("process_reward", {}).get("checks", [])
        )

        def count_checks(name: str, *, passed: bool | None = None) -> int:
            return sum(
                1
                for check in process_checks
                if check.get("name") == name
                and (passed is None or bool(check.get("passed")) is passed)
            )

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.rollout_config.response_length],
            response_mask=response_mask[: self.rollout_config.response_length],
            response_logprobs=response_logprobs[: self.rollout_config.response_length],
            reward_score=reward_score,
            num_turns=len(episode.turns),
            metrics=metrics,
            extra_fields={
                "tau_hindsight_evidence": credit_evidence,
                "reward_extra_info": {
                    "tau_episode_id": episode.episode_id,
                    "tau_group_id": group_id,
                    "tau_sample_index": sample_index,
                    "tau_seed": episode_seed,
                    "tau_domain": domain,
                    "tau_task_id": task_id,
                    "tau_success": bool(episode.success),
                    "tau_outcome_reward": float(episode.reward.outcome or 0.0),
                    "tau_process_reward": float(episode.reward.process or 0.0),
                    "tau_total_turns": len(episode.turns),
                    "tau_prompt_tokens": len(prompt_ids),
                    "tau_response_tokens": sum(response_mask),
                    "tau_tool_call_count": sum(
                        len(turn.tool_calls) for turn in episode.turns
                    ),
                    "tau_hit_max_turns": len(episode.turns) >= self.settings.max_steps,
                    "tau_retained_credit_turns": len(turn_token_spans),
                    "tau_context_rotations": context_rotations,
                    "tau_database_source": database_source,
                    "tau_termination_reason": episode.termination_reason,
                    "tau_invalid_action_count": count_checks(
                        "action_parse", passed=False
                    ),
                    "tau_invalid_action_error": "",
                    "tau_invalid_token_ids": "",
                    "tau_invalid_decoded_output": "",
                    "tau_tool_error_count": count_checks(
                        "tool_execution", passed=False
                    ),
                    "tau_missing_result_count": count_checks(
                        "tool_result_received", passed=False
                    ),
                    "tau_recovery_count": count_checks(
                        "error_recovery", passed=True
                    ),
                    "tau_unresolved_error_count": count_checks(
                        "error_recovery", passed=False
                    ),
                    "tau_abnormal_truncation_count": count_checks(
                        "abnormal_truncation", passed=False
                    ),
                },
            },
        )

    def _build_policy_failure_output(
        self,
        *,
        episode: EpisodeRecord,
        prompt_ids: list[int],
        response_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float],
        metrics: AgentLoopMetrics,
        group_id: str,
        sample_index: int,
        episode_seed: int | None,
        domain: str,
        task_id: str,
        database_source: str,
        context_rotations: int,
        attempted_turns: int,
        invalid_action_error: str,
        invalid_token_ids: list[int],
        invalid_decoded_output: str,
    ) -> AgentLoopOutput:
        response_limit = self.rollout_config.response_length

        kept_response_ids = response_ids[:response_limit]
        kept_response_mask = response_mask[:response_limit]
        kept_response_logprobs = response_logprobs[:response_limit]

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=kept_response_ids,
            response_mask=kept_response_mask,
            response_logprobs=kept_response_logprobs,
            reward_score=0.0,
            num_turns=attempted_turns,
            metrics=metrics,
            extra_fields={
                "tau_hindsight_evidence": [0.0] * len(kept_response_ids),
                "reward_extra_info": {
                    "tau_episode_id": episode.episode_id,
                    "tau_group_id": group_id,
                    "tau_sample_index": sample_index,
                    "tau_seed": episode_seed,
                    "tau_domain": domain,
                    "tau_task_id": task_id,
                    "tau_success": False,
                    "tau_outcome_reward": 0.0,
                    "tau_process_reward": 0.0,
                    "tau_total_turns": attempted_turns,
                    "tau_prompt_tokens": len(prompt_ids),
                    "tau_response_tokens": sum(kept_response_mask),
                    "tau_tool_call_count": sum(
                        len(turn.tool_calls) for turn in episode.turns
                    ),
                    "tau_hit_max_turns": (
                        attempted_turns >= self.settings.max_steps
                    ),
                    "tau_retained_credit_turns": 0,
                    "tau_context_rotations": context_rotations,
                    "tau_database_source": database_source,
                    "tau_termination_reason": episode.termination_reason,
                    "tau_invalid_action_count": 1,
                    "tau_invalid_action_error": invalid_action_error,
                    "tau_invalid_token_ids": json.dumps(invalid_token_ids),
                    "tau_invalid_decoded_output": invalid_decoded_output,
                    "tau_tool_error_count": 0,
                    "tau_missing_result_count": 0,
                    "tau_recovery_count": 0,
                    "tau_unresolved_error_count": 0,
                    "tau_abnormal_truncation_count": 0,
                },
            },
        )
    def _finalize_episode(
        self,
        episode: EpisodeRecord,
        transition: Any,
        turn_token_spans: list[tuple[int, int, int]],
        response_length: int,
    ) -> tuple[float, list[float]]:
        if transition.done:
            simulation_run = _decode_json_object(
                transition.evaluator_info.get("simulation_run"),
                field_name="simulation_run",
            )
            _hydrate_tool_results(episode, simulation_run, strict=True)
            outcome = float(transition.reward)
            termination_reason = _get_termination_reason(
                simulation_run,
                transition,
            )
        else:
            outcome = 0.0
            termination_reason = "response_length"
            if episode.turns:
                episode.turns[-1].truncated = True

        episode.finish(
            reward=RewardRecord(outcome=outcome, total=outcome),
            success=outcome >= 1.0,
            termination_reason=termination_reason,
        )
        process = self.process_reward.evaluate(episode)
        mixed = self.reward_mixer.mix(episode, process)

        credit_evidence = [0.0] * response_length
        if self.settings.enable_hindsight_credit:
            from agent_rl.credit.hindsight_credit import (
                HindsightCreditAssigner,
                HindsightCreditConfig,
            )

            credit = HindsightCreditAssigner(
                HindsightCreditConfig(**(self.settings.hindsight_config or {}))
            ).assign(episode)
            for turn_index, start, end in turn_token_spans:
                turn_evidence = credit.turn_evidence[turn_index]
                for index in range(start, min(end, response_length)):
                    credit_evidence[index] = turn_evidence

        return mixed.total, credit_evidence


def _required_string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"dataset field {key!r} must be a non-empty string")
    return value


def _decode_arguments(arguments: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return dict(arguments)

    if not isinstance(arguments, str):
        raise ActionFormatError(
            "tool arguments must be a JSON string or object"
        )

    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError as error:
        raise ActionFormatError(
            f"model emitted invalid tool JSON: {arguments}"
        ) from error

    if not isinstance(decoded, dict):
        raise ActionFormatError(
            "tool arguments must decode to an object"
        )

    return decoded
