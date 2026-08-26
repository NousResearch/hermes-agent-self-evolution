#!/usr/bin/env bash
# One-shot runner for the Hermes skill self-evolution tool.
# Usage: ./evolve.sh <skill-name> [iterations] [extra args...]
# Examples:
#   ./evolve.sh github-code-review 10
#   ./evolve.sh excel-vba-automation 5 --eval-source sessiondb
set -e
cd "$(dirname "$0")"

SKILL="$1"; shift || true
ITER="${1:-10}"; shift 2>/dev/null || true

if [ -z "$SKILL" ]; then
  echo "Usage: ./evolve.sh <skill-name> [iterations] [--eval-source synthetic|sessiondb|golden] ..."
  exit 1
fi

# Locate the hermes-agent repo: explicit env var wins, then common locations.
if [ -z "$HERMES_AGENT_REPO" ]; then
  for candidate in \
    "$HOME/.hermes/hermes-agent" \
    "$LOCALAPPDATA/hermes/hermes-agent" \
    "$HOME/AppData/Local/hermes/hermes-agent" \
    "../hermes-agent"; do
    if [ -d "$candidate" ]; then
      HERMES_AGENT_REPO="$candidate"
      break
    fi
  done
fi

if [ -z "$HERMES_AGENT_REPO" ]; then
  echo "hermes-agent repo not found. Set HERMES_AGENT_REPO=/path/to/hermes-agent" >&2
  exit 1
fi

# Pick the venv interpreter for the current platform.
if [ -x .venv/Scripts/python.exe ]; then
  PYTHON=".venv/Scripts/python.exe"
elif [ -x .venv/bin/python ]; then
  PYTHON=".venv/bin/python"
else
  echo "No .venv found. Run: python -m venv .venv && .venv/bin/pip install -e ." >&2
  exit 1
fi

# Clear PYTHONPATH: a globally-set value can collide with hermes' own venv.
env -u PYTHONPATH \
  "$PYTHON" -m evolution.skills.evolve_skill \
  --skill "$SKILL" \
  --iterations "$ITER" \
  --hermes-repo "$HERMES_AGENT_REPO" \
  "$@"
