#!/usr/bin/env bash
#
# Propose-only sentinel scan — $0, pure git, no LLM, never opens a PR.
#
# Runs `python -m evolution.monitor` against a target repo to refresh the
# code-evolution triage queue. Safe to schedule (launchd/cron): the scan only
# reads the target repo's git stream and writes a triage_queue.json + report.
#
# It deliberately does NOT pass --attempt-top: that flag runs the repair loop and
# is the only step that spends money, so it stays a manual, cost-capped, human-
# triggered action (see docs/operating_the_sentinel.md). Do not add it here.
#
# Usage:
#   scripts/monitor_scan.sh <target-repo>
#   SENTINEL_SINCE_DAYS=30 scripts/monitor_scan.sh <target-repo>
# Or set SENTINEL_TARGET_REPO instead of the positional argument.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_REPO="${1:-${SENTINEL_TARGET_REPO:-}}"
SINCE_DAYS="${SENTINEL_SINCE_DAYS:-90}"
MAX_PER_TOOL="${SENTINEL_MAX_PER_TOOL:-5}"

if [[ -z "$TARGET_REPO" ]]; then
  echo "error: target repo not given (pass as \$1 or set SENTINEL_TARGET_REPO)" >&2
  exit 2
fi
if [[ ! -d "$TARGET_REPO/.git" ]]; then
  echo "error: '$TARGET_REPO' is not a git repo" >&2
  exit 2
fi

PY="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "error: repo venv python not found at $PY (run 'uv sync' or create .venv)" >&2
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT="$REPO_ROOT/output/monitor/$TS"

# Timestamped output keeps the audit trail of what was flagged when; the 'latest'
# symlink gives humans a stable path to the current queue without destroying history.
"$PY" -m evolution.monitor \
  --repo "$TARGET_REPO" \
  --since-days "$SINCE_DAYS" \
  --max-per-tool "$MAX_PER_TOOL" \
  --output-dir "$OUT"

ln -sfn "$OUT" "$REPO_ROOT/output/monitor/latest"
echo "queue refreshed: $OUT  (latest -> $OUT)"
