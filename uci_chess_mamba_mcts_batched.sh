#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_DIR/.venv-rocm"
TORCH_LIB="$("$VENV/bin/python" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
exec "$REPO_DIR/target/release/examples/uci_chess_mamba_mcts_batched" "$@"
