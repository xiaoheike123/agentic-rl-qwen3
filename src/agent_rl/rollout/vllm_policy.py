from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from openai import OpenAI

from ..envs.action_parser import ModelToolCall, to_tau_action


class PolicyResponseError(RuntimeError):
    """Raised when a vLLM response cannot become a valid policy action."""


@dataclass(frozen=True, slots=True)
class VLLMPolicyConfig:
    model: str = "Qwen3-8B"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: float = 120.0
    max_retries: int = 2
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")

        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP or HTTPS URL")

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    action: str
    content: str | None
    tool_calls: tuple[ModelToolCall, ...]
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    raw_response: dict[str, Any]


class VLLMPolicy:
    def __init__(self, config: VLLMPolicyConfig) -> None:
        self.config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    def generate(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> PolicyOutput:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": self.config.enable_thinking,
                }
            },
        }

        if tools:
            request["tools"] = list(tools)
            request["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**request)
        choice = response.choices[0]
        response_message = choice.message

        parsed_tool_calls = tuple(
            self._parse_tool_call(tool_call)
            for tool_call in (response_message.tool_calls or ())
        )

        try:
            action = to_tau_action(
                content=response_message.content,
                tool_calls=parsed_tool_calls,
            )
        except ValueError as error:
            raise PolicyResponseError(
                f"Invalid policy response: {error}"
            ) from error

        usage = response.usage

        return PolicyOutput(
            action=action,
            content=response_message.content,
            tool_calls=parsed_tool_calls,
            finish_reason=choice.finish_reason,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            raw_response=response.model_dump(mode="json"),
        )

    @staticmethod
    def _parse_tool_call(tool_call: Any) -> ModelToolCall:
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as error:
            raise PolicyResponseError(
                f"Tool arguments are not valid JSON: "
                f"{tool_call.function.arguments}"
            ) from error

        if not isinstance(arguments, dict):
            raise PolicyResponseError(
                "Tool arguments must decode to a JSON object"
            )

        return ModelToolCall(
            name=tool_call.function.name,
            arguments=arguments,
            tool_call_id=tool_call.id,
        )