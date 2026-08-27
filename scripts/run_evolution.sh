#!/usr/bin/env bash
# Scheduled evolution sweep — the cron entry point.
#
# This is deliberately thin. The previous deployed driver was ~120 lines of
# bash living outside the repository, so the rotation policy, the failure
# handling and the notification were neither tested nor reviewed with the code
# they drove. All of that now lives in evolution/monitor/, and this wrapper
# only does what a shell wrapper should: resolve the environment, take a lock,
# and hand off.
#
# Install:
#   0 20 * * 5 /path/to/scripts/run_evolution.sh >> /var/log/hermes-evolution.log 2>&1
#
# Environment:
#   HERMES_DATA_DIR   Hermes data directory (state.db, profiles, cron).
#                     Inside the container this is NOT $HOME.
#   HERMES_AGENT_REPO hermes-agent source checkout.
#   EVOLUTION_VENV    Virtualenv to run from. Defaults to <repo>/.venv.
#   SKILLS_PER_RUN    Skills to evolve per sweep (default 4).
#   ITERATIONS        GEPA full evaluations per skill (default 10).
#   MODEL             Model for both optimizer and eval roles.
#   TIME_BUDGET_MIN   Stop starting new skills after this long (default 240).
#   EVOLUTION_WEBHOOK_URL / EVOLUTION_WEBHOOK_SECRET[_FILE]
#                     Optional remote notification. A local status file is
#                     always written regardless.
#   DRY_RUN=1         Validate setup for each skill without optimizing.
#   EXTRA_ARGS        Anything else to pass through (e.g. --create-pr).

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${EVOLUTION_VENV:-$REPO/.venv}"
PYTHON="$VENV/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"

SKILLS_PER_RUN="${SKILLS_PER_RUN:-4}"
ITERATIONS="${ITERATIONS:-10}"
EVAL_SOURCE="${EVAL_SOURCE:-sessiondb}"
MODEL="${MODEL:-openai-codex/gpt-5.6-luna}"
TIME_BUDGET_MIN="${TIME_BUDGET_MIN:-240}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/output}"
LOCKFILE="${LOCKFILE:-$OUTPUT_ROOT/.sweep.lock}"

mkdir -p "$OUTPUT_ROOT"

# Single instance: a long sweep must never overlap the next trigger.
exec 9>"$LOCKFILE"
if command -v flock >/dev/null 2>&1; then
  if ! flock -n 9; then
    echo "$(date -u +%FT%TZ) another sweep is already running; exiting"
    exit 0
  fi
fi

ARGS=(
  --skills-per-run "$SKILLS_PER_RUN"
  --iterations "$ITERATIONS"
  --eval-source "$EVAL_SOURCE"
  --optimizer-model "$MODEL"
  --eval-model "$MODEL"
  --time-budget-min "$TIME_BUDGET_MIN"
  --output-root "$OUTPUT_ROOT"
  --notify
)
[ "${DRY_RUN:-0}" = "1" ] && ARGS+=(--dry-run)
[ -n "${HERMES_DATA_DIR:-}" ] && ARGS+=(--hermes-data-dir "$HERMES_DATA_DIR")
[ -n "${HERMES_AGENT_REPO:-}" ] && ARGS+=(--hermes-repo "$HERMES_AGENT_REPO")
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=($EXTRA_ARGS)

echo "$(date -u +%FT%TZ) starting sweep: ${ARGS[*]}"
cd "$REPO" || exit 1
"$PYTHON" -m evolution.monitor.run_rotation "${ARGS[@]}"
STATUS=$?

# The exit code reflects the sweep, not the notification. A sweep that
# evaluated nothing must not look healthy to whatever is watching this job.
echo "$(date -u +%FT%TZ) sweep finished with status $STATUS"
exit "$STATUS"
