#!/usr/bin/env zsh
# GEPA tool-description evolution launcher. ~/.hermes/.env is NOT shell-sourceable
# (contains CLI fragments), so the key is extracted, never sourced.
set -euo pipefail
cd /home/pat/.hermes/hermes-agent-self-evolution

# Hermes session env leaks its own 3.14 venv via these — strip or the
# evolution .venv's imports resolve against the wrong site-packages.
unset VIRTUAL_ENV PYTHONPATH PYTHONHOME 2>/dev/null || true

export OPENROUTER_API_KEY=$(sed -n 's/^OPENROUTER_API_KEY=//p' /home/pat/.hermes/.env | head -n1 | sed 's/["\x27]//g')
if [[ -z "$OPENROUTER_API_KEY" ]]; then
  echo "OPENROUTER_API_KEY not found in ~/.hermes/.env" >&2
  exit 1
fi

exec .venv/bin/python -m evolution.tools.evolve_tool_descriptions \
  --tool search_files \
  --hermes-repo /home/pat/.hermes/hermes-agent \
  --days 90 \
  --iterations 5 \
  --optimizer-model openrouter/deepseek/deepseek-v4.1-flash-1017 \
  --eval-model openrouter/deepseek/deepseek-v4.1-flash-1017
