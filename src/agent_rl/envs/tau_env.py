"""Typed single-episode wrapper around tau2 AgentGymEnv."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from tau2.data_model.tasks import Task
from tau2.evaluator import evaluator_nl_assertions
from tau2.gym.gym_agent import AgentGymEnv
from tau2.registry import registry
from tau2.utils.tools import parse_action_string


TaskTransform = Callable[[Any], Any]
EnvironmentTransform = Callable[[Any], Any]

# Keep this as an allowlist: new upstream tau2 info fields stay private until
# they are reviewed explicitly. In particular, ``task`` contains evaluator
# criteria and oracle actions and must never cross the policy-facing boundary.
PUBLIC_TAU_INFO_KEYS = frozenset({"policy", "tools"})
EVALUATOR_TAU_INFO_KEYS = frozenset({"simulation_run", "reward_info"})


def configure_tau_nl_evaluator(
    model: str,
    model_args: dict[str, Any] | None,
) -> None:
    """Configure tau2's process-global NL assertion judge for this worker."""

    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = model
    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = deepcopy(
        model_args or {}
    )


class TauInfrastructureError(RuntimeError):
    """Raised when tau2 cannot produce a valid episode state."""

    def __init__(
        self,
        stage: str,
        domain: str,
        task_id: str,
    ) -> None:
        self.stage = stage
        self.domain = domain
        self.task_id = task_id
        super().__init__(
            "tau2 infrastructure failure during "
            f"{stage} (domain={domain!r}, task_id={task_id!r})"
        )

    def __reduce__(self) -> tuple[type[TauInfrastructureError], tuple[str, str, str]]:
        """Reconstruct the exception across multiprocessing and Ray boundaries."""

        return type(self), (self.stage, self.domain, self.task_id)


def _has_simulation_run(evaluator_info: dict[str, Any]) -> bool:
    value = evaluator_info.get("simulation_run")

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return False

    if hasattr(value, "model_dump"):
        value = value.model_dump()

    return isinstance(value, dict) and bool(value)


def public_tau_info(
    info: dict[str, Any],
    *,
    domain: str,
    task_id: str,
) -> dict[str, Any]:
    """Return only information that policy-side rollout code may consume."""

    public = {
        key: value
        for key, value in info.items()
        if key in PUBLIC_TAU_INFO_KEYS
    }
    public["domain"] = domain
    public["task_id"] = task_id
    return public


def evaluator_tau_info(info: dict[str, Any]) -> dict[str, Any]:
    """Return runtime evidence reserved for reward and trajectory finalization."""

    return {
        key: value
        for key, value in info.items()
        if key in EVALUATOR_TAU_INFO_KEYS
    }


class _TransformableAgentGymEnv(AgentGymEnv):
    """AgentGymEnv extension point used only by robustness evaluation."""

    def __init__(
        self,
        *args: Any,
        task_transform: TaskTransform | None = None,
        environment_transform: EnvironmentTransform | None = None,
        task_override: Task | None = None,
        database_override: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._task_transform = task_transform
        self._environment_transform = environment_transform
        self._task_override = deepcopy(task_override)
        self._database_override = deepcopy(database_override)
        self._transformed_task: Any = None
        super().__init__(*args, **kwargs)

    def _get_task(self) -> Any:
        if self._transformed_task is None:
            task = (
                deepcopy(self._task_override)
                if self._task_override is not None
                else super()._get_task()
            )
            self._transformed_task = (
                self._task_transform(deepcopy(task))
                if self._task_transform is not None
                else task
            )
        return deepcopy(self._transformed_task)

    def _get_environment(self) -> Any:
        if self._database_override is None:
            environment = super()._get_environment()
        else:
            environment = registry.get_env_constructor(self.domain)(
                db=deepcopy(self._database_override),
                solo_mode=self.solo_mode,
            )
        if self._environment_transform is not None:
            environment = self._environment_transform(environment)
        return environment

    def get_db_hash(self) -> str | None:
        """Return the hash of the live episode DB without exposing DB contents."""

        if self._orchestrator is None:
            raise RuntimeError("environment must be reset before reading its DB hash")
        return self._orchestrator.environment.get_db_hash()


@dataclass(frozen=True, slots=True)
class TauEnvConfig:
    """Configuration required to construct one tau2 environment."""

    domain: str
    task_id: str
    max_steps: int = 50
    solo_mode: bool = False
    user_llm: str = "deepseek/deepseek-v4-flash"
    user_llm_args: dict[str, Any] | None = None
    evaluator_llm: str = "deepseek/deepseek-v4-pro"
    evaluator_llm_args: dict[str, Any] | None = None
    all_messages_as_observation: bool = True
    task_transform: TaskTransform | None = None
    environment_transform: EnvironmentTransform | None = None
    perturbation_name: str | None = None
    task_data: dict[str, Any] | None = None
    database_override: Any | None = None

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

        if not self.evaluator_llm.strip():
            raise ValueError("evaluator_llm must not be empty")

        if self.evaluator_llm_args is not None and not isinstance(
            self.evaluator_llm_args, dict
        ):
            raise TypeError("evaluator_llm_args must be a dictionary or None")

        if self.task_transform is not None and not callable(self.task_transform):
            raise TypeError("task_transform must be callable or None")

        if self.environment_transform is not None and not callable(
            self.environment_transform
        ):
            raise TypeError("environment_transform must be callable or None")

        if self.perturbation_name is not None and not self.perturbation_name.strip():
            raise ValueError("perturbation_name must not be blank")

        if self.task_data is not None:
            task = Task.model_validate(self.task_data)
            if task.id != self.task_id:
                raise ValueError(
                    "task_data.id must match the configured task_id: "
                    f"{task.id!r} != {self.task_id!r}"
                )


@dataclass(frozen=True, slots=True)
class TauReset:
    """Initial observation and public tau2 episode context."""

    observation: str
    info: dict[str, Any]
    evaluator_info: dict[str, Any]
    initial_db_hash: str | None = None


@dataclass(frozen=True, slots=True)
class TauTransition:
    """One action result returned by tau2."""

    observation: str
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    evaluator_info: dict[str, Any]
    db_hash: str | None = None

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class TauEnv:
    """Manage the lifecycle and state checks for one tau2 episode."""

    def __init__(self, config: TauEnvConfig) -> None:
        self.config = config
        configure_tau_nl_evaluator(
            config.evaluator_llm,
            config.evaluator_llm_args,
        )
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
            task_override=(
                Task.model_validate(config.task_data)
                if config.task_data is not None
                else None
            ),
            database_override=config.database_override,
        )
        self._has_reset = False
        self._done = False
        self._action_count = 0
        self._last_info: dict[str, Any] = {}
        self._last_evaluator_info: dict[str, Any] = {}

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

    @property
    def last_evaluator_info(self) -> dict[str, Any]:
        return self._last_evaluator_info

    def reset(self, seed: int | None = None) -> TauReset:
        observation, info = self._env.reset(seed=seed)

        self._validate_observation(observation)

        if not isinstance(info, dict):
            raise TypeError(
                f"tau2 reset info must be a dictionary, got {type(info).__name__}"
            )

        evaluator_info = evaluator_tau_info(info)
        if not observation.strip():
            raise TauInfrastructureError(
                stage="reset",
                domain=self.config.domain,
                task_id=self.config.task_id,
            )
        info = public_tau_info(
            info,
            domain=self.config.domain,
            task_id=self.config.task_id,
        )

        self._has_reset = True
        self._done = False
        self._action_count = 0
        self._last_info = info
        self._last_evaluator_info = evaluator_info

        return TauReset(
            observation=observation,
            info=info,
            evaluator_info=evaluator_info,
            initial_db_hash=self._env.get_db_hash(),
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

        try:
            observation, reward, terminated, truncated, info = self._env.step(action)
        except Exception as error:
            raise TauInfrastructureError(
                stage="step",
                domain=self.config.domain,
                task_id=self.config.task_id,
            ) from error

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

        evaluator_info = evaluator_tau_info(info)
        done = terminated or truncated
        if not done and not observation.strip():
            raise TauInfrastructureError(
                stage="step_observation",
                domain=self.config.domain,
                task_id=self.config.task_id,
            )
        if done and not _has_simulation_run(evaluator_info):
            raise TauInfrastructureError(
                stage="step_termination",
                domain=self.config.domain,
                task_id=self.config.task_id,
            )
        info = public_tau_info(
            info,
            domain=self.config.domain,
            task_id=self.config.task_id,
        )
        info["agent_rl_action_parse_valid"] = action_parse_valid
        info["agent_rl_action_parse_error"] = action_parse_error

        self._action_count += 1
        self._done = terminated or truncated
        self._last_info = info
        self._last_evaluator_info = evaluator_info

        return TauTransition(
            observation=observation,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
            evaluator_info=evaluator_info,
            db_hash=self._env.get_db_hash(),
        )

    @staticmethod
    def _validate_observation(observation: Any) -> None:
        if not isinstance(observation, str):
            raise TypeError(
                f"tau2 observation must be a string, got {type(observation).__name__}"
            )
