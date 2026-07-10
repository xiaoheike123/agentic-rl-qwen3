"""Thin wrapper around tau2.gym.AgentGymEnv."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from tau2.gym.gym_agent import AgentGymEnv


@dataclass(frozen=True, slots=True)#保存创建环境需要的参数
class TauEnvConfig:
    domain: str
    task_id: str
    max_steps: int = 50
    solo_mode: bool = False
    user_llm: str = "deepseek/deepseek-v4-flash"
    user_llm_args: dict[str, Any] | None = None
    all_messages_as_observation: bool = True

    def __post_init__(self) -> None:
        if not self.domain.strip():
            raise ValueError("domain must not be empty")

        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")

        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")


@dataclass(frozen=True, slots=True)#表示reset()的返回值
class TauReset:
    observation: str
    info: dict[str, Any]


@dataclass(frozen=True, slots=True)#表示一次step()的返回值
class TauTransition:
    observation: str
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class TauEnv:
    def __init__(self, config: TauEnvConfig) -> None:
        self.config = config

        self._env = AgentGymEnv(
            domain=config.domain,
            task_id=config.task_id,
            max_steps=config.max_steps,
            solo_mode=config.solo_mode,
            user_llm=config.user_llm,
            user_llm_args=deepcopy(config.user_llm_args),
            all_messages_as_observation=config.all_messages_as_observation,
        )

        self._has_reset = False
        self._done = False
        self._action_count = 0
        self._last_info: dict[str, Any] = {}

    @property
    def action_count(self) -> int:
        return self._action_count

    @property
    def done(self) -> bool:
        return self._done

    @property
    def last_info(self) -> dict[str, Any]:
        return self._last_info

    def reset(self, seed: int | None = None) -> TauReset:
        observation, info = self._env.reset(seed=seed)

        self._has_reset = True
        self._done = False
        self._action_count = 0
        self._last_info = info

        return TauReset(
            observation=observation,
            info=info,
        )

    def step(self, action: str) -> TauTransition:
        if not self._has_reset:
            raise RuntimeError("reset() must be called before step()")

        if self._done:
            raise RuntimeError("The episode has already finished")

        if not isinstance(action, str):
            raise TypeError("action must be a string")

        if not action.strip():
            raise ValueError("action must not be empty")

        observation, reward, terminated, truncated, info = self._env.step(
            action
        )

        self._action_count += 1
        self._done = terminated or truncated
        self._last_info = info

        return TauTransition(
            observation=observation,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
