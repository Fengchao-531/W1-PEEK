#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$DATASETS_DIR/../.venv-lloom/bin/python}"
export MPLBACKEND=Agg
"$PYTHON_BIN" "$SCRIPT_DIR/visualize_domain_two_topics_only.py" "$@"
exec "$PYTHON_BIN" "$SCRIPT_DIR/visualize_exact_label_combinations.py" "$@"
