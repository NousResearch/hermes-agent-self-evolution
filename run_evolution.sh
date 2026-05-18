#!/bin/bash
# Safe self-evolution runner (no hardcoded secrets).
# Usage:
#   export HERMES_AGENT_REPO=/path/to/hermes-agent
#   export OPENAI_API_KEY=...
#   export OPENAI_BASE_URL=https://api.groq.com/openai/v1   # optional
#   ./run_evolution.sh

set -euo pipefail
cd "$(dirname "$0")"

: "${HERMES_AGENT_REPO:?HERMES_AGENT_REPO must be set}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.groq.com/openai/v1}"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="output/evolution_runs"
mkdir -p "$LOG_DIR"

echo "=== Starting Self-Evolution at $(date) ==="
echo "Repo: ${HERMES_AGENT_REPO}"
echo "Skill: code-review"
echo "Optimizer: GEPA (5 iterations)"
echo "Base URL: ${OPENAI_BASE_URL}"
echo ""

HERMES_AGENT_REPO="$HERMES_AGENT_REPO" \
OPENAI_API_KEY="$OPENAI_API_KEY" \
OPENAI_BASE_URL="$OPENAI_BASE_URL" \
python -m evolution.skills.evolve_skill \
  --skill code-review \
  --iterations 5 \
  --eval-source synthetic \
  --optimizer-model "groq/llama-3.3-70b-versatile" \
  --eval-model "openai/llama-3.3-70b-versatile" \
  --dry-run \
  2>&1 | tee "$LOG_DIR/code-review_${STAMP}.log"

echo ""
echo "=== Dry-run complete at $(date) ==="
