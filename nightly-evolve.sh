#!/bin/bash
# Nightly skill evolution — picks one skill, runs GEPA, reports results.
# Designed to be called by Hermes cron.

set -euo pipefail

cd ~/dev/hermes-agent-self-evolution
set -a && source ~/.hermes/.env && set +a
export HERMES_AGENT_REPO=~/.hermes/hermes-agent
export ANTHROPIC_API_KEY="${ANTHROPIC_TOKEN}"

# LiteLLM retry config — default backoff (0.5s init, 8s max) is too short
# for Anthropic's per-minute rate limits. Stretch it out.
export INITIAL_RETRY_DELAY=5
export MAX_RETRY_DELAY=60

# Priority skills to evolve (most-used, highest impact)
SKILLS=(
  systematic-debugging
  github-code-review
  test-driven-development
  code-review
  github-pr-workflow
  github-issues
  google-workspace
  himalaya
  dogfood
  plan
  writing-plans
  subagent-driven-development
  native-mcp
  searxng-brave-fallback
  obsidian
)

# State file to track which skill is next
STATE_FILE=~/.hermes/skill-evolution-state
RESULTS_DIR=~/dev/hermes-agent-self-evolution/output

# Read last index, advance to next
LAST_INDEX=0
if [[ -f "$STATE_FILE" ]]; then
  LAST_INDEX=$(cat "$STATE_FILE")
fi
NEXT_INDEX=$(( (LAST_INDEX + 1) % ${#SKILLS[@]} ))
echo "$NEXT_INDEX" > "$STATE_FILE"

SKILL="${SKILLS[$NEXT_INDEX]}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Nightly Skill Evolution ==="
echo "Skill: $SKILL"
echo "Time: $(date)"
echo ""

# Run evolution using ChatGPT OAuth (gpt-5.2) (LM calls have num_retries=8 + extended backoff)
PYTHONUNBUFFERED=1 /usr/bin/python -u -m evolution.skills.evolve_skill \
    --skill "$SKILL" \
    --iterations 5 \
    --eval-source synthetic \
    --eval-model chatgpt/gpt-5.2 \
    --optimizer-model chatgpt/gpt-5.2 \
    2>&1

# Find the latest output dir for this skill
LATEST=$(ls -td "$RESULTS_DIR/$SKILL"/20* 2>/dev/null | head -1)
if [[ -n "$LATEST" && -f "$LATEST/metrics.json" ]]; then
  echo ""
  echo "=== Metrics ==="
  cat "$LATEST/metrics.json"
  
  echo ""
  echo "=== Diff ==="
  if [[ -f "$LATEST/evolved_skill.md" ]]; then
    diff "$LATEST/baseline_skill.md" "$LATEST/evolved_skill.md" || true
  elif [[ -f "$LATEST/evolved_FAILED.md" ]]; then
    echo "Evolution failed constraints. Check $LATEST/evolved_FAILED.md"
  fi
fi
