#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source .venv311/bin/activate
if [[ -f /Users/kirniy/.hermes/.env ]]; then
  set -a
  source /Users/kirniy/.hermes/.env
  set +a
fi
export HERMES_AGENT_REPO="${HERMES_AGENT_REPO:-/Users/kirniy/.hermes/hermes-agent}"
export PYTHONUNBUFFERED=1
# Force Gemini through local SOCKS5 geo-bypass proxy (Nano Banana pattern)
export GEMINI_PROXY="${GEMINI_PROXY:-socks5://127.0.0.1:10818}"
export ALL_PROXY="${ALL_PROXY:-socks5h://127.0.0.1:10818}"
export HTTPS_PROXY="${HTTPS_PROXY:-socks5h://127.0.0.1:10818}"
export HTTP_PROXY="${HTTP_PROXY:-socks5h://127.0.0.1:10818}"
GOOGLE_KEY_ENV=""
GOOGLE_KEY_VALUE=""
POOL_SIZE="$(python ./scripts/google_key_pool.py count 2>/dev/null || echo 0)"
select_google_key() {
  local selection
  selection="$(python ./scripts/google_key_pool.py select 2>/dev/null)" || return 1
  GOOGLE_KEY_ENV="$(printf '%s\n' "$selection" | sed -n '1p')"
  GOOGLE_KEY_VALUE="$(printf '%s\n' "$selection" | sed -n '2p')"
  export GOOGLE_API_KEY="$GOOGLE_KEY_VALUE"
  export GEMINI_API_KEY="$GOOGLE_KEY_VALUE"
}

SKILL_NAME="${SKILL_NAME:-github-code-review}"
ITERATIONS="${ITERATIONS:-1}"
EVAL_SOURCE="${EVAL_SOURCE:-synthetic}"
OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-gemini/gemini-3.1-pro-preview}"
EVAL_MODEL="${EVAL_MODEL:-gemini/gemini-3.1-pro-preview}"
RUN_TESTS="${RUN_TESTS:-0}"

cmd=(python -m evolution.skills.evolve_skill
  --skill "$SKILL_NAME"
  --iterations "$ITERATIONS"
  --eval-source "$EVAL_SOURCE"
  --optimizer-model "$OPTIMIZER_MODEL"
  --eval-model "$EVAL_MODEL"
)

if [[ "$RUN_TESTS" == "1" ]]; then
  cmd+=(--run-tests)
fi

printf 'Running nightly skill evolution: %s\n' "$SKILL_NAME"
printf 'Repo: %s\n' "$REPO_ROOT"
printf 'Hermes repo: %s\n' "$HERMES_AGENT_REPO"
printf 'Pool size: %s\n' "$POOL_SIZE"
printf 'Command: %q ' "${cmd[@]}"
printf '\n\n'

attempt=0
max_attempts=${POOL_SIZE:-0}
if [[ "$max_attempts" -le 0 ]]; then
  max_attempts=1
fi

while [[ $attempt -lt $max_attempts ]]; do
  attempt=$((attempt + 1))
  if select_google_key; then
    printf 'Attempt %s/%s with key %s\n' "$attempt" "$max_attempts" "$GOOGLE_KEY_ENV"
  fi
  set +e
  "${cmd[@]}"
  status=$?
  set -e
  if [[ $status -eq 0 ]]; then
    exit 0
  fi
  if [[ -n "$GOOGLE_KEY_ENV" ]]; then
    python ./scripts/google_key_pool.py mark-bad "$GOOGLE_KEY_ENV" "run_failed_exit_${status}" >/dev/null 2>&1 || true
  fi
  printf 'Attempt %s failed with exit %s\n\n' "$attempt" "$status"
done

exit ${status:-1}
