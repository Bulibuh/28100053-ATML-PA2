#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export UV_CACHE_DIR="$(pwd)/.cache/uv"
if [ ! -x .venv/bin/python ]; then
    uv venv .venv
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    uv pip install --python .venv/bin/python 'torch==2.6.0+cpu' --index-url https://download.pytorch.org/whl/cpu
fi
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-dev.txt
source scripts/dev_env.sh
python -m scripts.check_environment
python -m scripts.verify_starter
