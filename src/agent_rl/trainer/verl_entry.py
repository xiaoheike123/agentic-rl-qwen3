"""Prepare tau2 datasets and launch one configured verl experiment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from agent_rl.trainer.config_adapter import (
    ExperimentConfig,
    load_experiment_config,
)


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return yaml.safe_dump(value, default_flow_style=True).strip()
    return str(value)


def _override(name: str, value: Any) -> str:
    return f"{name}={_stringify(value)}"


def _rollout_parallelism(config: ExperimentConfig) -> tuple[int, int]:
    max_concurrent = int(config.rollout.get("max_concurrent_episodes", 8))
    requested_workers = int(config.runtime.get("agent_loop_workers", 8))
    if max_concurrent <= 0:
        raise ValueError("rollout.max_concurrent_episodes must be positive")
    if requested_workers <= 0:
        raise ValueError("runtime.agent_loop_workers must be positive")

    workers = min(requested_workers, max_concurrent)
    episodes_per_worker = max(1, max_concurrent // workers)
    return workers, episodes_per_worker


def _prepare_files(config: ExperimentConfig) -> tuple[Path, Path, Path]:
    from agent_rl.data.build_dataset import (
        build_official_eval_dataset,
    )
    from agent_rl.data.synthetic.builder import (
        SyntheticBuildConfig,
        build_synthetic_corpus,
        validate_corpus_manifest,
    )
    from agent_rl.data.synthetic.audit import (
        AuditStatus,
        SyntheticAuditConfig,
        audit_synthetic_corpus,
        write_audit_report,
    )
    from agent_rl.data.synthetic.sampler import build_balanced_verl_dataset
    from agent_rl.data.synthetic.schema import SyntheticSplit

    runtime = config.runtime
    dataset_root = Path(
        os.environ.get("AGENT_RL_DATASET_ROOT", runtime["dataset_root"])
    )
    output_root = Path(os.environ.get("AGENT_RL_OUTPUT_ROOT", runtime["output_root"]))
    domains = tuple(config.environment.get("domains") or ())
    if not domains:
        domain = config.environment.get("domain")
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("environment config requires domain or domains")
        domains = (domain,)

    seed = int(runtime.get("seed", 42))
    if bool(config.raw.get("evaluation_only", False)):
        eval_root = dataset_root / "official_eval"
        validation_path = eval_root / "base.jsonl"
        build_official_eval_dataset(
            validation_path,
            domains=domains,
            split="base",
            seed=seed,
        )
        train_path = validation_path
    else:
        if set(domains) != {"airline", "retail", "telecom"}:
            raise ValueError(
                "formal synthetic training requires airline, retail, and telecom"
            )
        corpus_root = Path(
            os.environ.get(
                "AGENT_RL_SYNTHETIC_ROOT",
                runtime.get("synthetic_corpus_root", dataset_root / "synthetic"),
            )
        )
        synthetic_seed = int(runtime.get("synthetic_seed", 43))
        build_config = SyntheticBuildConfig(
            output_root=corpus_root,
            seed=synthetic_seed,
            max_train_per_domain=int(
                runtime.get("synthetic_max_train_per_domain", 128)
            ),
            max_validation_per_domain=int(
                runtime.get("synthetic_max_validation_per_domain", 22)
            ),
            training_database_root=Path(
                runtime.get("training_database_root", dataset_root / "training_db")
            ),
            telecom_clone_factor=int(runtime.get("telecom_clone_factor", 16)),
        )
        if not (corpus_root / "manifest.json").is_file():
            build_synthetic_corpus(build_config)
        else:
            validate_corpus_manifest(build_config)
        audit_report = audit_synthetic_corpus(
            SyntheticAuditConfig(corpus_root=corpus_root)
        )
        write_audit_report(
            audit_report,
            json_path=corpus_root / "quality_audit.json",
            markdown_path=corpus_root / "quality_audit.md",
        )
        if audit_report.status is AuditStatus.FAIL:
            failed = [
                f"{item.domain or 'global'}:{item.code}"
                for item in audit_report.findings
                if item.status is AuditStatus.FAIL
            ]
            raise RuntimeError(
                "synthetic corpus failed quality audit: " + ", ".join(failed)
            )
        balanced_root = dataset_root / "balanced"
        train_path = balanced_root / "train.jsonl"
        validation_path = balanced_root / "validation.jsonl"
        build_balanced_verl_dataset(
            train_path,
            corpus_root=corpus_root,
            split=SyntheticSplit.TRAIN,
            seed=synthetic_seed,
        )
        build_balanced_verl_dataset(
            validation_path,
            corpus_root=corpus_root,
            split=SyntheticSplit.VALIDATION,
            seed=synthetic_seed + 100_000,
        )

    run_dir = output_root / config.experiment.lower()
    run_dir.mkdir(parents=True, exist_ok=True)
    loop_path = run_dir / "tau_agent_loop.yaml"
    _, episodes_per_worker = _rollout_parallelism(config)
    loop_config = [
        {
            "name": "tau_agent",
            "_target_": "agent_rl.rollout.tau_agent_loop.TauAgentLoop",
            "settings": {
                "max_steps": int(config.environment.get("max_steps", 30)),
                "user_llm": config.environment["user_llm"],
                "user_llm_args": dict(config.environment.get("user_llm_args") or {}),
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
                "max_action_tokens": 2_048,
                "max_episode_attempts": int(
                    config.rollout.get("max_episode_attempts", 3)
                ),
                "retry_backoff_seconds": float(
                    config.rollout.get("retry_backoff_seconds", 1.0)
                ),
                "max_concurrent_episodes_per_worker": episodes_per_worker,
                "training_database_root": str(
                    runtime.get(
                        "training_database_root",
                        dataset_root / "training_db",
                    )
                ),
            },
        }
    ]
    with loop_path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(loop_config, stream, sort_keys=False)
    return train_path, validation_path, loop_path


def build_verl_command(config: ExperimentConfig) -> list[str]:
    train_path, validation_path, loop_path = _prepare_files(config)
    runtime = config.runtime
    model = config.model
    algorithm = config.algorithm
    model_path = os.environ.get("AGENT_RL_MODEL_PATH", runtime["model_path"])

    if config.credit is not None:
        adv_estimator = "tau_hindsight_balanced_grpo"
    elif algorithm["aggregation"] == "balanced":
        adv_estimator = "tau_balanced_grpo"
    else:
        adv_estimator = "grpo"

    response_length = int(model["max_response_length"])
    prompt_length = int(model["max_prompt_length"])
    logger = runtime.get("logger", ["console"])
    agent_loop_workers, _ = _rollout_parallelism(config)
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        _override("algorithm.adv_estimator", adv_estimator),
        _override("algorithm.use_kl_in_reward", False),
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
            "actor_rollout_ref.model.external_lib", "agent_rl.trainer.verl_algorithms"
        ),
        _override("actor_rollout_ref.model.lora_rank", model.get("lora_rank", 0)),
        _override("actor_rollout_ref.model.lora_alpha", model.get("lora_alpha", 16)),
        _override(
            "actor_rollout_ref.model.enable_gradient_checkpointing",
            model.get("gradient_checkpointing", True),
        ),
        _override(
            "actor_rollout_ref.actor.policy_loss.loss_mode", algorithm["verl_loss_mode"]
        ),
        _override(
            "actor_rollout_ref.actor.loss_agg_mode", algorithm["verl_loss_agg_mode"]
        ),
        _override(
            "actor_rollout_ref.actor.clip_ratio_low", algorithm["clip_ratio_low"]
        ),
        _override(
            "actor_rollout_ref.actor.clip_ratio_high", algorithm["clip_ratio_high"]
        ),
        _override(
            "actor_rollout_ref.actor.use_kl_loss", algorithm.get("use_kl_loss", False)
        ),
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
            prompt_length + response_length,
        ),
        _override("actor_rollout_ref.rollout.name", "vllm"),
        _override("actor_rollout_ref.rollout.mode", "async"),
        _override("actor_rollout_ref.rollout.tensor_model_parallel_size", 1),
        _override(
            "actor_rollout_ref.rollout.n",
            (
                runtime.get("evaluation_trials", 4)
                if bool(config.raw.get("evaluation_only", False))
                else runtime["rollout_group_size"]
            ),
        ),
        _override(
            "actor_rollout_ref.rollout.temperature",
            config.rollout.get("temperature", 1.0),
        ),
        _override("actor_rollout_ref.rollout.top_p", config.rollout.get("top_p", 0.95)),
        _override(
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            runtime["gpu_memory_utilization"],
        ),
        _override(
            "actor_rollout_ref.rollout.max_num_batched_tokens",
            runtime["max_num_batched_tokens"],
        ),
        _override("actor_rollout_ref.rollout.max_num_seqs", runtime["max_num_seqs"]),
        _override("actor_rollout_ref.rollout.agent.default_agent_loop", "tau_agent"),
        _override(
            "actor_rollout_ref.rollout.agent.agent_loop_config_path", str(loop_path)
        ),
        _override(
            "+actor_rollout_ref.rollout.agent.agent_loop_manager_class",
            "agent_rl.trainer.tau_agent_loop_manager.TauAgentLoopManager",
        ),
        _override(
            "actor_rollout_ref.rollout.agent.num_workers",
            agent_loop_workers,
        ),
        _override(
            "trainer.project_name", runtime.get("project_name", "agent-rl-qwen3")
        ),
        _override("trainer.experiment_name", config.experiment.lower()),
        _override("trainer.logger", logger),
        _override("trainer.n_gpus_per_node", runtime["n_gpus_per_node"]),
        _override("trainer.nnodes", runtime["nnodes"]),
        _override("trainer.total_epochs", runtime["total_epochs"]),
        _override("trainer.save_freq", runtime["save_freq"]),
        _override("trainer.test_freq", runtime["test_freq"]),
        _override(
            "trainer.max_actor_ckpt_to_keep",
            runtime.get("max_actor_ckpt_to_keep", 1),
        ),
        _override(
            "trainer.default_local_dir",
            str(
                Path(runtime["output_root"]) / config.experiment.lower() / "checkpoints"
            ),
        ),
        _override(
            "trainer.rollout_data_dir",
            str(Path(runtime["output_root"]) / config.experiment.lower() / "rollouts"),
        ),
        _override(
            "trainer.validation_data_dir",
            str(
                Path(runtime["output_root"]) / config.experiment.lower() / "validation"
            ),
        ),
    ]
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
    if bool(config.raw.get("evaluation_only", False)):
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
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
