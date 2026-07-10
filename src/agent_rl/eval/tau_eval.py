from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from tau2 import TextRunConfig
from tau2.runner import run_domain


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    if not isinstance(config_data, dict):
        raise ValueError(
            f"Config must contain a YAML mapping: {config_path}"
        )

    return config_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tau2 text evaluation from a YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML file matching tau2.TextRunConfig.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    config_path = args.config.resolve()
    config_data = load_yaml_config(config_path)
    run_config = TextRunConfig.model_validate(config_data)

    print(f"Evaluation config: {config_path}")
    print(f"Domain: {run_config.domain}")
    print(f"Agent model: {run_config.llm_agent}")
    print(f"User model: {run_config.llm_user}")

    run_domain(run_config)


if __name__ == "__main__":
    main()

