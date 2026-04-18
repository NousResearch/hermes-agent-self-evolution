#!/bin/bash
# Hermes self-evolution nightly pipeline
#
# Phases:
#   1. Preflight smoke (t1 baseline + t5 propose-mode structural; zero-token, ~10s)
#   2. Evolve skill in propose-mode (safe: no auto-merge to bundled skills)
#   3. Build markdown digest of last 24h of activity
#
# Defaults are safe. Override via env vars:
#   SKILL=github-code-review
#   ITERATIONS=10
#   MODE=propose           (propose | auto)
#   SKIP_SMOKE=0           (1 to bypass preflight)
#   SKIP_EVOLVE=0          (1 to only build digest)
#   WINDOW_HOURS=24
#   DELIVER=0              (1 to send digest via send_message; requires HERMES_CLI)

set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"

# ─── environment ──────────────────────────────────────────────────────────────
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:20128/v1}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:20128/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy-local}"

# shellcheck disable=SC1091
source venv/bin/activate

SKILL="${SKILL:-github-code-review}"
ITERATIONS="${ITERATIONS:-10}"
MODEL="${MODEL:-openai/cx/gpt-5.4}"
MODE="${MODE:-propose}"
WINDOW_HOURS="${WINDOW_HOURS:-24}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_EVOLVE="${SKIP_EVOLVE:-0}"
DELIVER="${DELIVER:-0}"

STAMP="$(date +%Y%m%d-%H%M%S)"
DATE_DIR="$(date +%Y-%m-%d)"

LOG_DIR="${ROOT}/logs"
DIGEST_DIR="${ROOT}/logs/digests"
SMOKE_DIR="${ROOT}/logs/smoke"
mkdir -p "$LOG_DIR" "$DIGEST_DIR" "$SMOKE_DIR"

NIGHTLY_LOG="${LOG_DIR}/nightly-${STAMP}.log"
SMOKE_LOG="${SMOKE_DIR}/smoke-${STAMP}.log"
EVOLVE_LOG="${LOG_DIR}/evolve-${SKILL}-${STAMP}.log"
DIGEST_FILE="${DIGEST_DIR}/${DATE_DIR}.md"

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$NIGHTLY_LOG"
}

fail() {
    log "FAIL: $*"
    log "Nightly aborted. See $NIGHTLY_LOG"
    exit 1
}

log "=== Hermes self-evolution nightly ==="
log "skill=$SKILL iters=$ITERATIONS mode=$MODE model=$MODEL window=${WINDOW_HOURS}h"
log "nightly log: $NIGHTLY_LOG"

# ─── phase 1: smoke preflight ─────────────────────────────────────────────────
# Runs T1 (dry-run on all top-5 skills) + T5 (propose-mode structural dry-run).
# Both are zero-token — total ~10s. Use `bash smoke_test.sh full` weekly for the
# expensive T2/T3/T4 tiers that hit the LLM.
if [[ "$SKIP_SMOKE" == "1" ]]; then
    log "Phase 1 (smoke) skipped via SKIP_SMOKE=1"
else
    log "Phase 1: smoke preflight (t1 + t5, zero-token)"
    if ! bash smoke_test.sh t1 >>"$SMOKE_LOG" 2>&1; then
        fail "smoke t1 failed — see $SMOKE_LOG"
    fi
    log "  t1 OK"
    if ! bash smoke_test.sh t5 >>"$SMOKE_LOG" 2>&1; then
        fail "smoke t5 failed — see $SMOKE_LOG"
    fi
    log "  t5 OK"
fi

# ─── phase 2: evolve ──────────────────────────────────────────────────────────
EVOLVE_EXIT=0
if [[ "$SKIP_EVOLVE" == "1" ]]; then
    log "Phase 2 (evolve) skipped via SKIP_EVOLVE=1"
else
    log "Phase 2: evolve skill=$SKILL mode=$MODE"
    set +e
    python -m evolution.skills.evolve_skill \
        --skill "$SKILL" \
        --iterations "$ITERATIONS" \
        --optimizer-model "$MODEL" \
        --eval-model "$MODEL" \
        --mode "$MODE" \
        >>"$EVOLVE_LOG" 2>&1
    EVOLVE_EXIT=$?
    set -e
    if [[ "$EVOLVE_EXIT" -ne 0 ]]; then
        log "  evolve exited non-zero ($EVOLVE_EXIT) — continuing to digest"
    else
        log "  evolve OK — log: $EVOLVE_LOG"
    fi
fi

# ─── phase 3: digest ──────────────────────────────────────────────────────────
log "Phase 3: build digest (window=${WINDOW_HOURS}h → $DIGEST_FILE)"
set +e
python -m evolution.review.digest \
    --hours "$WINDOW_HOURS" \
    --output "$DIGEST_FILE" \
    --format markdown \
    >>"$NIGHTLY_LOG" 2>&1
DIGEST_EXIT=$?
set -e
if [[ "$DIGEST_EXIT" -ne 0 ]]; then
    log "  digest failed (exit $DIGEST_EXIT) — see $NIGHTLY_LOG"
else
    log "  digest OK — $DIGEST_FILE"
fi

# ─── phase 4: delivery (optional) ─────────────────────────────────────────────
# Delivery is handled externally — cron can chain `send-message` via hermes-agent
# or a separate skill. We just produce a file; delivery is not our concern.
# Set DELIVER_CMD to a shell command that reads $DIGEST_FILE if you want inline delivery.
if [[ "$DELIVER" == "1" ]] && [[ -f "$DIGEST_FILE" ]] && [[ -n "${DELIVER_CMD:-}" ]]; then
    log "Phase 4: running DELIVER_CMD"
    set +e
    DIGEST_FILE="$DIGEST_FILE" bash -c "$DELIVER_CMD" >>"$NIGHTLY_LOG" 2>&1 || \
        log "  delivery failed (non-fatal)"
    set -e
fi

log "=== nightly complete ==="
log "evolve exit: $EVOLVE_EXIT | digest exit: $DIGEST_EXIT"
log "digest:  $DIGEST_FILE"
log "evolve:  $EVOLVE_LOG"
log "smoke:   $SMOKE_LOG"

# Overall exit: only fail if digest failed (evolve non-zero is not a nightly failure —
# it means the run rejected changes, which is expected in propose-mode).
exit "$DIGEST_EXIT"
