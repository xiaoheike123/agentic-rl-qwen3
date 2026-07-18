"""Build locked official-airline datasets and launch one verl experiment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from agent_rl.trainer.config_adapter import ExperimentConfig, load_experiment_config
from agent_rl.trainer.preflight import validate_experiment_config


_VERL_ALGORITHMS_MODULE = "agent_rl.trainer.verl_algorithms"


def _verl_subprocess_environment() -> dict[str, str]:
    """Load custom verl registries in the driver and every Ray child process."""

    environment = os.environ.copy()
    external_modules = [
        module.strip()
        for module in environment.get("VERL_USE_EXTERNAL_MODULES", "").split(",")
        if module.strip()
    ]
    if _VERL_ALGORITHMS_MODULE not in external_modules:
        external_modules.append(_VERL_ALGORITHMS_MODULE)
    environment["VERL_USE_EXTERNAL_MODULES"] = ",".join(external_modules)
    return environment


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, dict)):
        return yaml.safe_dump(value, default_flow_style=True).strip()
    return str(value)


def _override(name: str, value: Any) -> str:
    return f"{name}={_stringify(value)}"


def _rollout_parallelism(config: ExperimentConfig) -> tuple[int, int]:
    max_concurrent = int(config.rollout.get("max_concurrent_episodes", 8))
    requested_workers = int(config.runtime.get("agent_loop_workers", 8))
    if max_concurrent <= 0 or requested_workers <= 0:
        raise ValueError("rollout concurrency and worker counts must be positive")
    workers = min(requested_workers, max_concurrent)
    return workers, max(1, max_concurrent // workers)


def _assert_formal_airline_config(config: ExperimentConfig) -> None:
    domain = config.environment.get("domain")
    domains = config.environment.get("domains")
    if domain != "airline" or domains not in (None, [], ()):
        raise ValueError(
            "formal experiments are locked to environment.domain='airline'"
        )
    if config.environment.get("train_split") != "train":
        raise ValueError("formal training must use the official airline train split")
    if config.environment.get("evaluation_split") != "test":
        raise ValueError("formal evaluation must use the official airline test split")


def _output_root(config: ExperimentConfig) -> Path:
    return Path(
        os.environ.get("AGENT_RL_OUTPUT_ROOT", config.runtime["output_root"])
    )


def _prepare_files(config: ExperimentConfig) -> tuple[Path, Path, Path]:
    from agent_rl.data.build_dataset import (
        build_official_test_dataset,
        build_official_train_dataset,
    )

    _assert_formal_airline_config(config)
    runtime = config.runtime
    dataset_root = Path(
        os.environ.get("AGENT_RL_DATASET_ROOT", runtime["dataset_root"])
    )
    output_root = _output_root(config)
    official_root = dataset_root / "official_airline"
    evaluation_only = bool(config.raw.get("evaluation_only", False))
    seed = int(runtime.get("seed", 42))

    if evaluation_only:
        validation_path = official_root / "test_4seed.jsonl"
        count = build_official_test_dataset(validation_path)
        if count != 80:
            raise RuntimeError(f"locked final evaluation expected 80 rows, got {count}")
        train_path = validation_path
    else:
        task_limit = config.raw.get("train_task_limit")
        task_limit = int(task_limit) if task_limit is not None else None
        expected_count = task_limit or 30
        filename = "train.jsonl" if task_limit is None else f"train_first_{task_limit}.jsonl"
        train_path = official_root / filename
        count = build_official_train_dataset(
            train_path,
            seed=seed,
            task_limit=task_limit,
        )
        if count != expected_count:
            raise RuntimeError(
                f"locked training expected {expected_count} rows, got {count}"
            )
        # verl requires a val file even when validation is disabled. Reusing the
        # train reference here does not trigger evaluation because both
        # val_before_train and test_freq are disabled for training runs.
        validation_path = train_path

    run_dir = output_root / config.experiment.lower()
    run_dir.mkdir(parents=True, exist_ok=True)
    loop_path = run_dir / "tau_agent_loop.yaml"
    _, episodes_per_worker = _rollout_parallelism(config)
    loop_config = [
        {
            "name": "tau_agent",
            "_target_": "agent_rl.rollout.tau_agent_loop.TauAgentLoop",
            "settings": {
                "max_steps": int(config.environment["max_steps"]),
                "user_llm": config.environment["user_llm"],
                "user_llm_args": dict(
                    config.environment.get("user_llm_args") or {}
                ),
                "evaluator_llm": config.environment["evaluator_llm"],
                "evaluator_llm_args": dict(
                    config.environment.get("evaluator_llm_args") or {}
                ),
                "all_messages_as_observation": bool(
                    config.environment.get("all_messages_as_observation", False)
                ),
                "tool_parser": "hermes",
                "outcome_weight": float(config.reward.get("outcome_weight", 1.0)),
                "process_weight": float(config.reward.get("process_weight", 0.0)),
                "process_config": dict(config.reward.get("process") or {}),
                "enable_hindsight_credit": config.credit is not None,
                "hindsight_config": (
                    {
                        key: value
                        for key, value in config.credit.items()
                        if key
                        in {
                            "process_alignment_scale",
                            "minimum_weight",
                            "maximum_weight",
                        }
                    }
                    if config.credit is not None
                    else None
                ),
                "context_max_chars": int(
                    config.rollout.get("context_compression", {}).get(
                        "max_chars", 24_000
                    )
                ),
                "max_action_tokens": int(
                    config.rollout.get("max_action_tokens", 256)
                ),
                "max_episode_attempts": int(
                    config.rollout.get("max_episode_attempts", 3)
                ),
                "retry_backoff_seconds": float(
                    config.rollout.get("retry_backoff_seconds", 1.0)
                ),
                "max_concurrent_episodes_per_worker": episodes_per_worker,
            },
        }
    ]
    with loop_path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(loop_config, stream, sort_keys=False)
    return train_path, validation_path, loop_path


def _advantage_estimator(config: ExperimentConfig) -> str:
    balanced = config.algorithm["aggregation"] == "balanced"
    if config.credit is not None:
        return "tau_hindsight_balanced_grpo" if balanced else "tau_hindsight_grpo"
    return "tau_balanced_grpo" if balanced else "grpo"


def build_verl_command(config: ExperimentConfig) -> list[str]:
    validate_experiment_config(config)
    train_path, validation_path, loop_path = _prepare_files(config)
    runtime = config.runtime
    model = config.model
    algorithm = config.algorithm
    model_path = os.environ.get("AGENT_RL_MODEL_PATH", runtime["model_path"])
    evaluation_only = bool(config.raw.get("evaluation_only", False))
    prompt_length = int(model["max_prompt_length"])
    response_length = int(model["max_response_length"])
    rollout_n = 1 if evaluation_only else int(runtime["rollout_group_size"])
    workers, _ = _rollout_parallelism(config)
    run_root = _output_root(config) / config.experiment.lower()

    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        _override("algorithm.adv_estimator", _advantage_estimator(config)),
        _override("algorithm.use_kl_in_reward", False),
        _override(
            "algorithm.rollout_correction.bypass_mode",
            bool(algorithm.get("bypass_mode", False)),
        ),
        _override(
            "algorithm.rollout_correction.loss_type",
            algorithm.get("bypass_loss_type", "ppo_clip"),
        ),
        _override("data.train_files", [str(train_path)]),
        _override("data.val_files", [str(validation_path)]),
        _override("data.train_batch_size", runtime["train_batch_size"]),
        _override("data.max_prompt_length", prompt_length),
        _override("data.max_response_length", response_length),
        _override("data.filter_overlong_prompts", False),
        _override("data.truncation", "error"),
        _override("+data.apply_chat_template_kwargs.enable_thinking", False),
        _override("actor_rollout_ref.model.path", model_path),
        _override(
            "actor_rollout_ref.model.external_lib",
            "agent_rl.trainer.verl_algorithms",
        ),
        _override("actor_rollout_ref.model.lora_rank", model.get("lora_rank", 0)),
        _override("actor_rollout_ref.model.lora_alpha", model.get("lora_alpha", 16)),
        _override(
            "actor_rollout_ref.model.target_modules",
            model.get("lora_target_modules", "all-linear"),
        ),
        _override(
            "actor_rollout_ref.model.lora.merge",
            model.get("lora_merge", False),
        ),
        _override(
            "actor_rollout_ref.model.enable_gradient_checkpointing",
            model.get("gradient_checkpointing", False),
        ),
        _override(
            "actor_rollout_ref.model.enable_activation_offload",
            model.get("activation_offload", False),
        ),
        _override(
            "actor_rollout_ref.model.use_remove_padding",
            model.get("use_remove_padding", True),
        ),
        _override(
            "actor_rollout_ref.model.use_fused_kernels",
            model.get("use_fused_kernels", True),
        ),
        _override(
            "actor_rollout_ref.actor.policy_loss.loss_mode",
            algorithm["verl_loss_mode"],
        ),
        _override(
            "actor_rollout_ref.actor.loss_agg_mode",
            algorithm["verl_loss_agg_mode"],
        ),
        _override("actor_rollout_ref.actor.clip_ratio_low", algorithm["clip_ratio_low"]),
        _override("actor_rollout_ref.actor.clip_ratio_high", algorithm["clip_ratio_high"]),
        _override("actor_rollout_ref.actor.use_kl_loss", not evaluation_only),
        _override("actor_rollout_ref.actor.kl_loss_coef", runtime["kl_loss_coef"]),
        _override("actor_rollout_ref.actor.ppo_epochs", runtime["ppo_epochs"]),
        _override("actor_rollout_ref.actor.optim.lr", runtime["optimizer_lr"]),
        _override("actor_rollout_ref.actor.grad_clip", runtime["grad_clip"]),
        _override("actor_rollout_ref.actor.fsdp_config.use_orig_params", True),
        _override(
            "actor_rollout_ref.actor.ppo_mini_batch_size",
            runtime["ppo_mini_batch_size"],
        ),
        _override(
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
            runtime["ppo_micro_batch_size_per_gpu"],
        ),
        _override("actor_rollout_ref.actor.use_dynamic_bsz", True),
        _override(
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
            runtime.get(
                "ppo_max_token_len_per_gpu",
                prompt_length + response_length,
            ),
        ),
        _override("actor_rollout_ref.rollout.name", "vllm"),
        _override("actor_rollout_ref.rollout.mode", "async"),
        _override("actor_rollout_ref.rollout.tensor_model_parallel_size", 1),
        _override("actor_rollout_ref.rollout.n", rollout_n),
        _override("actor_rollout_ref.rollout.val_kwargs.n", 1),
        _override("actor_rollout_ref.rollout.temperature", config.rollout["temperature"]),
        _override("actor_rollout_ref.rollout.top_p", config.rollout["top_p"]),
        _override("actor_rollout_ref.rollout.seed", runtime["seed"]),
        _override("actor_rollout_ref.rollout.full_determinism", evaluation_only),
        _override(
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            runtime["gpu_memory_utilization"],
        ),
        _override(
            "actor_rollout_ref.rollout.max_num_batched_tokens",
            runtime["max_num_batched_tokens"],
        ),
        _override("actor_rollout_ref.rollout.max_num_seqs", runtime["max_num_seqs"]),
        _override("actor_rollout_ref.rollout.free_cache_engine", True),
        _override("actor_rollout_ref.rollout.enable_chunked_prefill", True),
        _override("actor_rollout_ref.rollout.enable_prefix_caching", True),
        _override("actor_rollout_ref.rollout.calculate_log_probs", True),
        _override("actor_rollout_ref.rollout.agent.default_agent_loop", "tau_agent"),
        _override(
            "actor_rollout_ref.rollout.agent.agent_loop_config_path",
            str(loop_path),
        ),
        _override(
            "+actor_rollout_ref.rollout.agent.agent_loop_manager_class",
            "agent_rl.trainer.tau_agent_loop_manager.TauAgentLoopManager",
        ),
        _override("actor_rollout_ref.rollout.agent.num_workers", workers),
        _override("trainer.project_name", runtime["project_name"]),
        _override("trainer.experiment_name", config.experiment.lower()),
        _override("trainer.logger", runtime.get("logger", ["console"])),
        _override("trainer.n_gpus_per_node", runtime["n_gpus_per_node"]),
        _override("trainer.nnodes", runtime["nnodes"]),
        _override("trainer.total_epochs", runtime["total_epochs"]),
        _override("trainer.save_freq", runtime["save_freq"]),
        _override("trainer.test_freq", runtime["test_freq"]),
        _override("trainer.val_before_train", runtime["val_before_train"]),
        _override("trainer.resume_mode", runtime["resume_mode"]),
        _override(
            "trainer.max_actor_ckpt_to_keep",
            runtime["max_actor_ckpt_to_keep"],
        ),
        _override(
            "trainer.default_local_dir",
            str(run_root / "checkpoints"),
        ),
        _override(
            "trainer.rollout_data_dir",
            str(run_root / "rollouts"),
        ),
        _override(
            "trainer.validation_data_dir",
            str(run_root / "validation"),
        ),
    ]

    adapter_path = os.environ.get(
        "AGENT_RL_LORA_ADAPTER_PATH",
        model.get("lora_adapter_path") or "",
    )
    if adapter_path:
        command.append(
            _override("actor_rollout_ref.model.lora_adapter_path", adapter_path)
        )
    if config.credit is not None:
        command.extend(
            [
                _override(
                    "+algorithm.hindsight_process_alignment_scale",
                    config.credit.get("process_alignment_scale", 1.0),
                ),
                _override(
                    "+algorithm.hindsight_minimum_weight",
                    config.credit.get("minimum_weight", 0.05),
                ),
                _override(
                    "+algorithm.hindsight_maximum_weight",
                    config.credit.get("maximum_weight", 3.0),
                ),
            ]
        )
    if evaluation_only:
        command.append(_override("trainer.val_only", True))
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    command = build_verl_command(config) + args.overrides
    print("Launching:")
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(
            command,
            check=True,
            env=_verl_subprocess_environment(),
        )


if __name__ == "__main__":
    main()
