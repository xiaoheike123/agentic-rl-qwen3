#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/root/autodl-tmp/conda-envs/agent-rl-train}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
UV_BIN="${UV_BIN:-uv}"
CONDA_BIN="${CONDA_BIN:-conda}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    "$CONDA_BIN" create -y -p "$ENV_PREFIX" python=3.12
fi

cd "$PROJECT_ROOT"

"$UV_BIN" pip install --python "$PYTHON_BIN" 'vllm==0.24.0' --torch-backend=auto
"$UV_BIN" pip install --python "$PYTHON_BIN" \
    accelerate codetiming datasets dill hydra-core 'numpy>=2.0.0' pandas peft \
    'pyarrow>=19.0.0' pybind11 pylatexenc 'ray[default]>=2.41.0' torchdata \
    'tensordict>=0.8.0,<=0.10.0,!=0.9.0' 'transformers!=5.6.0' \
    wandb packaging tensorboard pytest
"$UV_BIN" pip install --python "$PYTHON_BIN" -e './tau2-bench[gym]'
"$UV_BIN" pip install --python "$PYTHON_BIN" -e ./verl --no-deps
"$UV_BIN" pip install --python "$PYTHON_BIN" -e '.[dev]'

"$PYTHON_BIN" -c "import agent_rl, tau2, torch, verl, vllm; print('torch', torch.__version__); print('vllm', vllm.__version__); assert torch.cuda.is_available(); print('project imports: OK')"
"$PYTHON_BIN" -m compileall -q src
echo "Unified training runtime is ready: $PYTHON_BIN"
