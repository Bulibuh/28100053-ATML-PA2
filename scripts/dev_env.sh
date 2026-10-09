#!/usr/bin/env bash
# Source this file from any directory. All caches remain local to the checkout.
ATML_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="$ATML_REPO_ROOT/.cache/huggingface"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_HUB_DISABLE_XET=1
export MPLCONFIGDIR="$ATML_REPO_ROOT/.cache/matplotlib"
export IPYTHONDIR="$ATML_REPO_ROOT/.cache/ipython"
export XDG_CACHE_HOME="$ATML_REPO_ROOT/.cache/xdg"
export JUPYTER_RUNTIME_DIR="$ATML_REPO_ROOT/.cache/jupyter/runtime"
export JUPYTER_DATA_DIR="$ATML_REPO_ROOT/.cache/jupyter/data"
export JUPYTER_CONFIG_DIR="$ATML_REPO_ROOT/.cache/jupyter/config"
mkdir -p "$HF_HOME" "$HF_XET_CACHE" "$MPLCONFIGDIR" "$IPYTHONDIR" "$XDG_CACHE_HOME" "$JUPYTER_RUNTIME_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_CONFIG_DIR"
if [ -f "$ATML_REPO_ROOT/.venv/bin/activate" ]; then
    source "$ATML_REPO_ROOT/.venv/bin/activate"
fi
