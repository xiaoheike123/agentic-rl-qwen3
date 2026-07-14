from pathlib import Path

from agent_rl.trainer import verl_entry
from agent_rl.trainer.config_adapter import load_experiment_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _command(monkeypatch, config_name: str) -> list[str]:
    config = load_experiment_config(PROJECT_ROOT / "configs" / "train" / config_name)
    monkeypatch.setattr(
        verl_entry,
        "_prepare_files",
        lambda _config: (Path("train.jsonl"), Path("validation.jsonl"), Path("loop.yaml")),
    )
    return verl_entry.build_verl_command(config)


def test_official_eval_uses_preexpanded_four_seed_rows(monkeypatch) -> None:
    command = _command(monkeypatch, "e0_base_eval.yaml")
    assert "actor_rollout_ref.rollout.n=1" in command
    assert "actor_rollout_ref.rollout.val_kwargs.n=1" in command
    assert "trainer.val_only=true" in command
    assert "actor_rollout_ref.rollout.full_determinism=true" in command


def test_training_uses_group_four_without_test_validation(monkeypatch) -> None:
    command = _command(monkeypatch, "e1_grpo_sequence.yaml")
    assert "actor_rollout_ref.rollout.n=4" in command
    assert "trainer.val_before_train=false" in command
    assert "trainer.test_freq=-1" in command
    assert "trainer.val_only=true" not in command


def test_e4_uses_sequence_hindsight_estimator(monkeypatch) -> None:
    command = _command(monkeypatch, "e4_sequence_hindsight.yaml")
    assert "algorithm.adv_estimator=tau_hindsight_grpo" in command
    assert "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean" in command


def test_lora_protocol_is_explicit_in_command(monkeypatch) -> None:
    command = _command(monkeypatch, "e1_grpo_sequence.yaml")
    assert "actor_rollout_ref.model.lora_rank=64" in command
    assert "actor_rollout_ref.actor.optim.lr=1e-05" in command
    assert "actor_rollout_ref.actor.fsdp_config.use_orig_params=true" in command
    assert "algorithm.rollout_correction.bypass_mode=false" in command
    assert "actor_rollout_ref.rollout.free_cache_engine=true" in command
    assert "actor_rollout_ref.rollout.calculate_log_probs=true" in command


def test_token_budget_probe_changes_scheduling_only(monkeypatch) -> None:
    command = _command(monkeypatch, "g4_perf_tokens32k.yaml")
    assert "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" in command
    assert "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768" in command
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=32768" in command
    assert "algorithm.rollout_correction.bypass_mode=false" in command


def test_micro_batch_probe_changes_micro_batch_only(monkeypatch) -> None:
    command = _command(monkeypatch, "g4_perf_micro2.yaml")
    assert "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2" in command
    assert "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768" in command
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=32768" in command
    assert "actor_rollout_ref.rollout.agent.num_workers=8" in command


def test_concurrency_probe_uses_sixteen_isolated_workers(monkeypatch) -> None:
    command = _command(monkeypatch, "g4_perf_concurrency16.yaml")
    assert "actor_rollout_ref.rollout.agent.num_workers=16" in command
    assert "actor_rollout_ref.rollout.max_num_seqs=16" in command


def test_bypass_probe_is_explicit_and_keeps_rollout_logprobs(monkeypatch) -> None:
    command = _command(monkeypatch, "g4_bypass_on.yaml")
    assert "algorithm.rollout_correction.bypass_mode=true" in command
    assert "algorithm.rollout_correction.loss_type=ppo_clip" in command
    assert "actor_rollout_ref.rollout.calculate_log_probs=true" in command
    assert "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2" in command
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=32768" in command
    assert "actor_rollout_ref.rollout.agent.num_workers=16" in command


def test_output_root_override_applies_to_every_verl_artifact(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_RL_OUTPUT_ROOT", "/tmp/locked-output")
    command = _command(monkeypatch, "e1_grpo_sequence.yaml")
    run_root = Path("/tmp/locked-output") / "e1"

    assert f"trainer.default_local_dir={run_root / 'checkpoints'}" in command
    assert f"trainer.rollout_data_dir={run_root / 'rollouts'}" in command
    assert f"trainer.validation_data_dir={run_root / 'validation'}" in command
