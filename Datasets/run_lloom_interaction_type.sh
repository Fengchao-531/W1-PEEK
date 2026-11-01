#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/../.venv-lloom/bin/python}"
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/Annotation/annotate.py" --dimension interaction_type "$@"
