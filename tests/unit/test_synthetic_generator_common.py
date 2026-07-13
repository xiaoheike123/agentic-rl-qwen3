import json
from pathlib import Path

import pytest

from agent_rl.data.synthetic.builder import (
    SyntheticBuildConfig,
    validate_corpus_manifest,
)
from agent_rl.data.synthetic.generators.common import (
    GENERATOR_VERSION,
    OracleActionSpec,
    make_candidate,
)


def test_make_candidate_preserves_ordered_oracle_actions() -> None:
    candidate = make_candidate(
        domain="telecom",
        template="roaming_and_data",
        seed=42,
        entities=("customer-1", "line-1"),
        reason_for_call="Enable roaming and add data.",
        known_info="Customer and line identifiers are available.",
        task_instructions="Request both changes and confirm the charge.",
        actions=(
            OracleActionSpec(
                name="enable_roaming",
                arguments={"customer_id": "customer-1", "line_id": "line-1"},
            ),
            OracleActionSpec(
                name="refuel_data",
                arguments={
                    "customer_id": "customer-1",
                    "line_id": "line-1",
                    "gb_amount": 2.0,
                },
            ),
        ),
        communicate_info=["line-1", "roaming", "2 GB"],
        purpose="Exercise a two-action target.",
    )

    actions = candidate.task.evaluation_criteria.actions
    assert actions is not None
    assert [action.name for action in actions] == ["enable_roaming", "refuel_data"]
    assert [action.action_id for action in actions] == [
        "oracle_enable_roaming_0",
        "oracle_refuel_data_1",
    ]
    assert candidate.generation.generator_version == "2.0.0"


def test_make_candidate_rejects_an_empty_action_sequence() -> None:
    with pytest.raises(ValueError, match="at least one Oracle action"):
        make_candidate(
            domain="telecom",
            template="empty",
            seed=42,
            entities=("customer-1",),
            reason_for_call="Do nothing.",
            known_info="No information.",
            task_instructions="Do nothing.",
            actions=(),
            communicate_info=[],
            purpose="Invalid empty target.",
        )


def test_manifest_rejects_a_stale_generator_version(tmp_path: Path) -> None:
    config = SyntheticBuildConfig(output_root=tmp_path, domains=("airline",))
    domain_root = tmp_path / "airline"
    domain_root.mkdir()
    (domain_root / "train.jsonl").write_text("", encoding="utf-8")
    (domain_root / "validation.jsonl").write_text("", encoding="utf-8")
    manifest = {
        "config": {
            "generator_version": GENERATOR_VERSION,
            "domains": ["airline"],
            "seed": 42,
            "validation_fraction": 0.15,
            "similarity_threshold": 0.82,
            "max_per_split_per_domain": None,
        }
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    validate_corpus_manifest(config)

    manifest["config"]["generator_version"] = "1.0.0"
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="generator_version"):
        validate_corpus_manifest(config)
