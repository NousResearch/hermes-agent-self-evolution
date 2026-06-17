#!/bin/bash
# Wrapper script for hermes-agent-self-evolution
# Reads Pioneer API key directly from ~/.hermes/.env (one-line grep)

set -e

# Read key directly - simpler than sourcing
PIONEER_KEY=$(grep '^PIONEER_API_KEY=' ~/.hermes/.env | head -1 | cut -d= -f2-)
if [ -z "$PIONEER_KEY" ]; then
    echo "ERROR: PIONEER_API_KEY not found in ~/.hermes/.env" >&2
    exit 1
fi

export OPENAI_API_KEY="$PIONEER_KEY"
export OPENAI_BASE_URL="https://api.pioneer.ai/v1"
export HERMES_AGENT_REPO="/Volumes/1TB/AI_Workspace/Hermes/.hermes/profiles/clean"

PYTHON="/Volumes/1TB/AI_Workspace/Hermes/.hermes/hermes-agent/venv/bin/python3"
REPO_DIR="/Volumes/1TB/AI_Workspace/hermes-agent-self-evolution"

cd "$REPO_DIR" || exit 1

# Run evolution
EVOLVE_EXIT=0
"$PYTHON" -m evolution.skills.evolve_skill "$@" || EVOLVE_EXIT=$?

# Auto-write results to Hindsight
RESULT_DIR=$(ls -td results/*/ 2>/dev/null | head -1)
if [ -n "$RESULT_DIR" ] && [ -f "$RESULT_DIR/metrics.json" ]; then
    HINDSIGHT_URL="http://192.168.50.225:8788/v1/default/banks/hermes-clean" \
        "$PYTHON" scripts/hindsight_writeback.py "$RESULT_DIR" 2>/dev/null || true
fi

exit $EVOLVE_EXIT