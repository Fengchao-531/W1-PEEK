#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/../.venv-lloom/bin/python}"
export MPLBACKEND=Agg
exec "$PYTHON_BIN" "$SCRIPT_DIR/visualize_lloom_analysis.py" "$@"
