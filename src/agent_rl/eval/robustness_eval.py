"""Run paired clean and interaction-perturbed tau2 evaluations."""

from __future__ import annotations

import json
import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import yaml
from dotenv import load_dotenv

from agent_rl.data.schemas import EpisodeRecord
from agent_rl.envs.tau_env import TauEnvConfig
from agent_rl.robustness.information_order import (
    InformationOrderVariant,
    make_information_order_transform,
)
from agent_rl.robustness.metrics import compute_robustness_metrics
from agent_rl.robustness.paraphrase import (
    ParaphraseVariant,
    make_paraphrase_transform,
)
from agent_rl.robustness.tool_failure import (
    RecoverableToolFailureInjector,
    ToolFailurePlan,
)
from agent_rl.rollout.episode_worker import EpisodeSpec, EpisodeWorker
from agent_rl.rollout.vllm_policy import VLLMPolicy, VLLMPolicyConfig
from agent_rl.utils.jsonl import JsonlEpisodeStore


@dataclass(frozen=True, slots=True)
class RobustnessCase:
    task_id: str
    paraphrase: ParaphraseVariant
    information_order: InformationOrderVariant
    tool_failure: ToolFailurePlan


@dataclass(frozen=True, slots=True)
class RobustnessEvaluation:
    clean: tuple[EpisodeRecord, ...]
    paraphrase: tuple[EpisodeRecord, ...]
    information_order: tuple[EpisodeRecord, ...]
    tool_failure: tuple[EpisodeRecord, ...]
    metrics: dict[str, dict[str, Any]]


def load_robustness_cases(path: str | Path) -> tuple[RobustnessCase, ...]:
    cases = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                task_id = row["task_id"]
                cases.append(
                    RobustnessCase(
                        task_id=task_id,
                        paraphrase=ParaphraseVariant(
                            task_id=task_id,
                            **row["paraphrase"],
                        ),
                        information_order=InformationOrderVariant(
                            task_id=task_id,
                            **row["information_order"],
                        ),
                        tool_failure=ToolFailurePlan(**row.get("tool_failure", {})),
                    )
                )
            except Exception as error:
                raise ValueError(
                    f"invalid robustness case at line {line_number}: {error}"
                ) from error
    if not cases:
        raise ValueError("robustness manifest contains no cases")
    return tuple(cases)


class RobustnessEvaluator:
    def __init__(self, worker: EpisodeWorker, base_env: TauEnvConfig) -> None:
        self.worker = worker
        self.base_env = base_env

    def evaluate(
        self,
        cases: Sequence[RobustnessCase],
        *,
        trials: int = 1,
        seed: int = 42,
    ) -> RobustnessEvaluation:
        if trials <= 0:
            raise ValueError("trials must be positive")

        results: dict[str, list[EpisodeRecord]] = {
            "clean": [],
            "paraphrase": [],
            "information_order": [],
            "tool_failure": [],
        }
        for case_index, case in enumerate(cases):
            for trial in range(trials):
                trial_seed = seed + case_index * 10_000 + trial
                environments = {
                    "clean": replace(
                        self.base_env,
                        task_id=case.task_id,
                        perturbation_name="clean",
                    ),
                    "paraphrase": replace(
                        self.base_env,
                        task_id=case.task_id,
                        task_transform=make_paraphrase_transform(case.paraphrase),
                        perturbation_name="user_paraphrase",
                    ),
                    "information_order": replace(
                        self.base_env,
                        task_id=case.task_id,
                        task_transform=make_information_order_transform(
                            case.information_order
                        ),
                        perturbation_name="information_order_shift",
                    ),
                }
                injector = RecoverableToolFailureInjector(case.tool_failure)
                environments["tool_failure"] = replace(
                    self.base_env,
                    task_id=case.task_id,
                    environment_transform=injector.transform,
                    perturbation_name="recoverable_tool_failure",
                )

                for name, env_config in environments.items():
                    episode = self.worker.run(
                        EpisodeSpec(
                            episode_id=f"robust::{name}::{case.task_id}::{trial}",
                            group_id=f"robust::{case.task_id}",
                            env_config=env_config,
                            sample_index=0,
                            trial_id=trial,
                            seed=trial_seed,
                        )
                    )
                    if name == "tool_failure":
                        episode.metadata["injected_tool_failures"] = (
                            injector.injected_count
                        )
                    results[name].append(episode)

        metrics = {
            name: asdict(
                compute_robustness_metrics(
                    results["clean"],
                    results[name],
                    tool_failure=name == "tool_failure",
                )
            )
            for name in ("paraphrase", "information_order", "tool_failure")
        }
        return RobustnessEvaluation(
            clean=tuple(results["clean"]),
            paraphrase=tuple(results["paraphrase"]),
            information_order=tuple(results["information_order"]),
            tool_failure=tuple(results["tool_failure"]),
            metrics=metrics,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    project_root = config_path.parents[2]
    load_dotenv(project_root / ".env")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    worker = EpisodeWorker(
        VLLMPolicy(
            VLLMPolicyConfig(
                model=config["model"],
                base_url=config["base_url"],
                api_key=config.get("api_key", "EMPTY"),
                temperature=float(config.get("temperature", 0.0)),
                max_tokens=int(config.get("max_tokens", 2048)),
                enable_thinking=bool(config.get("enable_thinking", False)),
            )
        )
    )
    evaluator = RobustnessEvaluator(
        worker,
        TauEnvConfig(
            domain=config["domain"],
            task_id="replaced-by-case",
            max_steps=int(config.get("max_steps", 50)),
            user_llm=config["user_llm"],
            user_llm_args=dict(config.get("user_llm_args") or {}),
            evaluator_llm=config.get(
                "evaluator_llm", "deepseek/deepseek-v4-pro"
            ),
            evaluator_llm_args=dict(config.get("evaluator_llm_args") or {}),
            all_messages_as_observation=True,
        ),
    )
    manifest_path = (project_root / config["manifest"]).resolve()
    evaluation = evaluator.evaluate(
        load_robustness_cases(manifest_path),
        trials=int(config.get("trials", 1)),
        seed=int(config.get("seed", 42)),
    )
    output_dir = (project_root / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("clean", "paraphrase", "information_order", "tool_failure"):
        store = JsonlEpisodeStore(output_dir / f"{name}.jsonl")
        for episode in getattr(evaluation, name):
            store.append(episode)
    with (output_dir / "metrics.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(evaluation.metrics, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(evaluation.metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
