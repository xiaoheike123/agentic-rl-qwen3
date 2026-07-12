"""Typed single-episode wrapper around tau2 AgentGymEnv."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from tau2.gym.gym_agent import AgentGymEnv
from tau2.utils.tools import parse_action_string


TaskTransform = Callable[[Any], Any]
EnvironmentTransform = Callable[[Any], Any]


class _TransformableAgentGymEnv(AgentGymEnv):
    """AgentGymEnv extension point used only by robustness evaluation."""

    def __init__(
        self,
        *args: Any,
        task_transform: TaskTransform | None = None,
        environment_transform: EnvironmentTransform | None = None,
        **kwargs: Any,
    ) -> None:
        self._task_transform = task_transform
        self._environment_transform = environment_transform
        self._transformed_task: Any = None
        super().__init__(*args, **kwargs)

    def _get_task(self) -> Any:
        if self._transformed_task is None:
            task = super()._get_task()
            self._transformed_task = (
                self._task_transform(deepcopy(task))
                if self._task_transform is not None
                else task
            )
        return deepcopy(self._transformed_task)

    def _get_environment(self) -> Any:
        environment = super()._get_environment()
        if self._environment_transform is not None:
            environment = self._environment_transform(environment)
        return environment


@dataclass(frozen=True, slots=True)
class TauEnvConfig:
    """Configuration required to construct one tau2 environment."""

    domain: str
    task_id: str
    max_steps: int = 50
    solo_mode: bool = False
    user_llm: str = "deepseek/deepseek-v4-flash"
    user_llm_args: dict[str, Any] | None = None
    all_messages_as_observation: bool = True
    task_transform: TaskTransform | None = None
    environment_transform: EnvironmentTransform | None = None
    perturbation_name: str | None = None

    def __post_init__(self) -> None:
        if not self.domain.strip():
            raise ValueError("domain must not be empty")

        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")

        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")

        if not self.user_llm.strip():
            raise ValueError("user_llm must not be empty")

        if self.user_llm_args is not None and not isinstance(self.user_llm_args, dict):
            raise TypeError("user_llm_args must be a dictionary or None")

        if self.task_transform is not None and not callable(self.task_transform):
            raise TypeError("task_transform must be callable or None")

        if self.environment_transform is not None and not callable(
            self.environment_transform
        ):
            raise TypeError("environment_transform must be callable or None")

        if self.perturbation_name is not None and not self.perturbation_name.strip():
            raise ValueError("perturbation_name must not be blank")


@dataclass(frozen=True, slots=True)
class TauReset:
    """Initial observation and public tau2 episode context."""

    observation: str
    info: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TauTransition:
    """One action result returned by tau2."""

    observation: str
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class TauEnv:
    """Manage the lifecycle and state checks for one tau2 episode."""

    def __init__(self, config: TauEnvConfig) -> None:
        self.config = config
        self._env = _TransformableAgentGymEnv(
            domain=config.domain,
            task_id=config.task_id,
            max_steps=config.max_steps,
            solo_mode=config.solo_mode,
            user_llm=config.user_llm,
            user_llm_args=deepcopy(config.user_llm_args),
            all_messages_as_observation=config.all_messages_as_observation,
            task_transform=config.task_transform,
            environment_transform=config.environment_transform,
        )
        self._has_reset = False
        self._done = False
        self._action_count = 0
        self._last_info: dict[str, Any] = {}

    @property
    def has_reset(self) -> bool:
        return self._has_reset

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

        self._validate_observation(observation)

        if not isinstance(info, dict):
            raise TypeError(
                f"tau2 reset info must be a dictionary, got {type(info).__name__}"
            )

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
            raise RuntimeError("the episode has already finished")

        if not isinstance(action, str):
            raise TypeError("action must be a string")

        if not action.strip():
            raise ValueError("action must not be empty")

        action_parse_valid = True
        action_parse_error: str | None = None
        try:
            parse_action_string(action)
        except Exception as error:
            action_parse_valid = False
            action_parse_error = f"{type(error).__name__}: {error}"

        observation, reward, terminated, truncated, info = self._env.step(action)

        self._validate_observation(observation)

        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise TypeError(f"tau2 reward must be numeric, got {type(reward).__name__}")

        if not isinstance(terminated, bool):
            raise TypeError("tau2 terminated flag must be boolean")

        if not isinstance(truncated, bool):
            raise TypeError("tau2 truncated flag must be boolean")

        if not isinstance(info, dict):
            raise TypeError(
                f"tau2 step info must be a dictionary, got {type(info).__name__}"
            )

        info = dict(info)
        info["agent_rl_action_parse_valid"] = action_parse_valid
        info["agent_rl_action_parse_error"] = action_parse_error

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

    @staticmethod
    def _validate_observation(observation: Any) -> None:
        if not isinstance(observation, str):
            raise TypeError(
                f"tau2 observation must be a string, got {type(observation).__name__}"
            )
