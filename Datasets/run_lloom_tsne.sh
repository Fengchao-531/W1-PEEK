#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/../.venv-lloom/bin/python}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$PYTHON_BIN" "$SCRIPT_DIR/visualize_lloom_tsne.py" --threads "$OMP_NUM_THREADS" "$@"
