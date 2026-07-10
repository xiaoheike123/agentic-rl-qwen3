#!/usr/bin/env bash
set -euo pipefail
python -m agent_rl.data.build_dataset "$@"

