"""Fail-fast checks for the locked official-airline experiment protocol."""

from __future__ import annotations

from agent_rl.trainer.config_adapter import ExperimentConfig


LORA_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}

TRAIN_MAX_STEPS = 64
EVALUATION_MAX_STEPS = 200


def validate_experiment_config(config: ExperimentConfig) -> None:
    evaluation_only = bool(config.raw.get("evaluation_only", False))
    if config.environment.get("domain") != "airline":
        raise ValueError("formal experiments must use airline only")
    if config.environment.get("train_split") != "train":
        raise ValueError("official airline train split is required")
    if config.environment.get("evaluation_split") != "test":
        raise ValueError("official airline test split is required")
    expected_max_steps = (
        EVALUATION_MAX_STEPS if evaluation_only else TRAIN_MAX_STEPS
    )
    actual_max_steps = int(config.environment.get("max_steps", 0))
    if actual_max_steps != expected_max_steps:
        run_kind = "evaluation" if evaluation_only else "training"
        raise ValueError(
            f"formal {run_kind} max agent turns must be {expected_max_steps}, "
            f"got {actual_max_steps}"
        )
    if int(config.rollout.get("max_action_tokens", 0)) != 256:
        raise ValueError("per-turn generation must be capped at 256 tokens")
    if int(config.model["max_prompt_length"]) + int(
        config.model["max_response_length"]
    ) != 16_384:
        raise ValueError("the model token budget must total 16K")

    if not evaluation_only:
        if config.model.get("training_method") != "lora":
            raise ValueError("E1-E5 training must use LoRA")
        if int(config.model.get("lora_rank", 0)) != 64:
            raise ValueError("formal LoRA rank must be 64")
        if int(config.model.get("lora_alpha", 0)) != 64:
            raise ValueError("formal LoRA alpha must be 64")
        if set(config.model.get("lora_target_modules") or ()) != LORA_TARGETS:
            raise ValueError("formal LoRA target modules do not match the protocol")
        if int(config.runtime["rollout_group_size"]) != 4:
            raise ValueError("formal GRPO group size must be 4")
        if int(config.runtime["ppo_epochs"]) != 1:
            raise ValueError("PPO epochs must be 1")
        if int(config.runtime["total_epochs"]) != 75:
            raise ValueError("formal training runtime must target 75 global steps")
        if bool(config.runtime["val_before_train"]):
            raise ValueError("official test evaluation is forbidden during training")
        if int(config.runtime["test_freq"]) != -1:
            raise ValueError("official test evaluation is forbidden during training")

    expected = {
        "E1": ("sequence", "outcome", False),
        "E2": ("balanced", "outcome", False),
        "E3": ("sequence", "environment_process", False),
        "E4": ("sequence", "outcome", True),
        "E5": ("balanced", "environment_process", True),
    }
    if config.experiment in expected:
        if bool(config.algorithm.get("bypass_mode", False)):
            raise ValueError(
                f"{config.experiment} must keep bypass disabled for the formal matrix"
            )
        aggregation, reward_mode, has_credit = expected[config.experiment]
        actual = (
            config.algorithm.get("aggregation"),
            config.reward.get("mode"),
            config.credit is not None,
        )
        if actual != (aggregation, reward_mode, has_credit):
            raise ValueError(
                f"{config.experiment} does not match the locked matrix: {actual}"
            )
