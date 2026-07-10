#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../tau2-bench"
uv sync --extra gym

