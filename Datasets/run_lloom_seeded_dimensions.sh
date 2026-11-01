#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  read -rsp 'OpenAI API key: ' OPENAI_API_KEY
  printf '\n'
  export OPENAI_API_KEY
fi
../.venv-lloom/bin/python -u run_lloom_seeded_dimensions.py "$@"
unset OPENAI_API_KEY
printf 'Forward port 8765, then open http://localhost:8765/domain/ or /interaction_type/ for the task you ran.\n'
../.venv-lloom/bin/python -m http.server 8765 --bind 127.0.0.1 --directory lloom_defined_dimensions_10
