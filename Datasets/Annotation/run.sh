#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec ../../.venv-lloom/bin/python -u annotate.py "$@"
