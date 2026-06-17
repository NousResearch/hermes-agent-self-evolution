#!/bin/bash
# Smoke test for hermes self-evolution pipeline.
#
# Tiers:
#   T1 — dry-run on each top-5 skill: validates skill discovery, frontmatter, constraints. No LLM.
#   T2 — local LLM endpoint reachability: verifies cx/gpt-5.4 responds via OpenAI-compat API.
#   T3 — real 1-iteration run on smallest top-5 skill: full loop end-to-end. EXPENSIVE.
#   T4 — propose-mode E2E with real optimization. EXPENSIVE (~4min, real tokens).
#   T5 — propose-mode STRUCTURAL (dry-run). Zero tokens, ~2s. Nightly cron uses T1 + T5.
#
# Usage:
#   ./smoke_test.sh            # run T1 + T5 (fast, zero-token — nightly default)
#   ./smoke_test.sh t1         # only tier 1
#   ./smoke_test.sh t2         # only tier 2
#   ./smoke_test.sh t3         # only tier 3 (expensive)
#   ./smoke_test.sh t4         # only tier 4 (expensive)
#   ./smoke_test.sh t5         # only tier 5 (cheap propose-mode check)
#   ./smoke_test.sh full       # run everything including T2/T3/T4 (weekly manual)
#   VERBOSE=1 ./smoke_test.sh  # stream subprocess output

set -uo pipefail

cd "$(dirname "$0")"

# ── env ──────────────────────────────────────────────────────────────
export OPENAI_API_BASE="http://localhost:20128/v1"
export OPENAI_BASE_URL="http://localhost:20128/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy-local-key}"

MODEL="openai/cx/gpt-5.4"
RAW_MODEL="cx/gpt-5.4"
TIER="${1:-all}"
VERBOSE="${VERBOSE:-0}"

# Top 5 skills targeted for weekly evolution.
# Format: "skill_name:hermes_repo_root"
#   - bundled skills: hermes_repo = ~/.hermes/hermes-agent
#   - user skills:    hermes_repo = ~/.hermes
TOP5=(
  "systematic-debugging:$HOME/.hermes/hermes-agent"
  "writing-plans:$HOME/.hermes/hermes-agent"
  "test-driven-development:$HOME/.hermes/hermes-agent"
  "auditing-third-party-guides:$HOME/.hermes"
  "hermes-cron-scripts:$HOME/.hermes"
)

# ── style ────────────────────────────────────────────────────────────
C_G="\033[32m"; C_R="\033[31m"; C_Y="\033[33m"; C_B="\033[36m"; C_0="\033[0m"
pass()  { printf "  ${C_G}✓${C_0} %s\n" "$*"; }
fail()  { printf "  ${C_R}✗${C_0} %s\n" "$*"; FAILS=$((FAILS+1)); }
warn()  { printf "  ${C_Y}⚠${C_0} %s\n" "$*"; }
hdr()   { printf "\n${C_B}━━ %s ━━${C_0}\n" "$*"; }

FAILS=0
RUN_LOG="$(mktemp -t smoke-evolution.XXXXXX)"
trap 'rm -f "$RUN_LOG"' EXIT

# ── venv ─────────────────────────────────────────────────────────────
if [ ! -f venv/bin/activate ]; then
  echo "ERROR: venv/bin/activate not found in $(pwd)" >&2
  exit 2
fi
# shellcheck disable=SC1091
source venv/bin/activate

# ═════════════════════════════════════════════════════════════════════
# T1 — Dry-run skill discovery + constraint validation
# ═════════════════════════════════════════════════════════════════════
run_t1() {
  hdr "T1 — dry-run on top 5 skills (no LLM cost)"
  for entry in "${TOP5[@]}"; do
    skill="${entry%%:*}"
    repo="${entry##*:}"
    if [ "$VERBOSE" = "1" ]; then
      python -m evolution.skills.evolve_skill \
        --skill "$skill" \
        --hermes-repo "$repo" \
        --dry-run \
        --optimizer-model "$MODEL" \
        --eval-model "$MODEL"
      rc=$?
    else
      python -m evolution.skills.evolve_skill \
        --skill "$skill" \
        --hermes-repo "$repo" \
        --dry-run \
        --optimizer-model "$MODEL" \
        --eval-model "$MODEL" >"$RUN_LOG" 2>&1
      rc=$?
    fi
    if [ $rc -eq 0 ]; then
      pass "$skill (repo=$(basename "$repo"))"
    else
      fail "$skill (repo=$(basename "$repo")) — exit $rc"
      [ "$VERBOSE" != "1" ] && tail -20 "$RUN_LOG" | sed 's/^/      /'
    fi
  done
}

# ═════════════════════════════════════════════════════════════════════
# T2 — Local endpoint reachability (cx/gpt-5.4)
# ═════════════════════════════════════════════════════════════════════
run_t2() {
  hdr "T2 — local LLM endpoint ($OPENAI_API_BASE)"

  # 2a. raw HTTP ping — just check something responds on /v1/models or /v1/chat/completions
  http_code=$(curl -s -o "$RUN_LOG" -w "%{http_code}" -m 5 \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    "$OPENAI_API_BASE/models" 2>/dev/null || echo "000")
  if [ "$http_code" = "200" ] || [ "$http_code" = "404" ]; then
    # 404 is fine — endpoint is up but /models not implemented; chat endpoint is what matters
    pass "endpoint reachable (HTTP $http_code on /models)"
  else
    fail "endpoint unreachable (HTTP $http_code) — is the gateway running?"
    return
  fi

  # 2b. real chat completion via litellm (what DSPy uses under the hood)
  python - <<PY 2>&1 | tee "$RUN_LOG" >/dev/null
import os, sys
try:
    import litellm
    resp = litellm.completion(
        model=os.environ.get("SMOKE_MODEL", "openai/cx/gpt-5.4"),
        messages=[{"role": "user", "content": "reply with the single word: pong"}],
        max_tokens=10,
        timeout=15,
    )
    out = resp["choices"][0]["message"]["content"].strip().lower()
    print(f"RESPONSE={out}")
    sys.exit(0 if "pong" in out or len(out) > 0 else 1)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(2)
PY
  rc=${PIPESTATUS[0]}
  if [ $rc -eq 0 ]; then
    resp=$(grep -m1 "^RESPONSE=" "$RUN_LOG" | cut -d= -f2-)
    pass "chat completion via litellm (response: \"$resp\")"
  else
    fail "litellm chat completion failed (exit $rc)"
    [ -s "$RUN_LOG" ] && tail -10 "$RUN_LOG" | sed 's/^/      /'
  fi
}

# ═════════════════════════════════════════════════════════════════════
# T3 — Real 1-iteration evolution run on smallest skill
# ═════════════════════════════════════════════════════════════════════
run_t3() {
  hdr "T3 — real 1-iteration run (writing-plans, smallest of top 5)"

  # Use writing-plans: smallest (7.3k chars) of the bundled top-5 → fastest synthetic dataset gen.
  skill="writing-plans"
  repo="$HOME/.hermes/hermes-agent"

  # isolate output dir so we can assert on it
  smoke_out="$(pwd)/output/_smoke"
  rm -rf "$smoke_out"
  mkdir -p "$smoke_out"

  start=$(date +%s)
  if [ "$VERBOSE" = "1" ]; then
    python -m evolution.skills.evolve_skill \
      --skill "$skill" \
      --hermes-repo "$repo" \
      --iterations 1 \
      --optimizer-model "$MODEL" \
      --eval-model "$MODEL" 2>&1 | tee "$RUN_LOG"
    rc=${PIPESTATUS[0]}
  else
    python -m evolution.skills.evolve_skill \
      --skill "$skill" \
      --hermes-repo "$repo" \
      --iterations 1 \
      --optimizer-model "$MODEL" \
      --eval-model "$MODEL" >"$RUN_LOG" 2>&1
    rc=$?
  fi
  elapsed=$(($(date +%s) - start))

  if [ $rc -ne 0 ]; then
    fail "evolution run failed (exit $rc, ${elapsed}s)"
    tail -30 "$RUN_LOG" | sed 's/^/      /'
    return
  fi
  pass "evolution run completed (exit 0, ${elapsed}s)"

  # Assert expected artifacts landed on disk.
  run_dir=$(find output/"$skill" -maxdepth 1 -type d -name "20*" 2>/dev/null | sort | tail -1)
  if [ -z "$run_dir" ]; then
    fail "no output dir under output/$skill/ — artifacts not written"
    return
  fi
  pass "output dir created: $run_dir"

  for f in metrics.json baseline_skill.md evolved_skill.md; do
    if [ -s "$run_dir/$f" ]; then
      pass "$f present ($(wc -c < "$run_dir/$f") bytes)"
    else
      fail "$f missing or empty"
    fi
  done

  # Sanity-check metrics.json keys.
  if [ -s "$run_dir/metrics.json" ]; then
    py_out=$(python - <<PY 2>&1
import json, sys
m = json.load(open("$run_dir/metrics.json"))
need = ["baseline_score","evolved_score","improvement","elapsed_seconds","iterations"]
missing = [k for k in need if k not in m]
if missing:
    print("MISSING:" + ",".join(missing)); sys.exit(1)
print(f"OK baseline={m['baseline_score']:.3f} evolved={m['evolved_score']:.3f} delta={m['improvement']:+.3f}")
PY
)
    if [ $? -eq 0 ]; then
      pass "metrics.json schema — $py_out"
    else
      fail "metrics.json schema — $py_out"
    fi
  fi
}

# ═════════════════════════════════════════════════════════════════════
# T5 — Propose-mode structural check (dry-run, zero tokens)
# ═════════════════════════════════════════════════════════════════════
# Validates the propose-mode code path without any LLM calls:
#   - CLI accepts --mode propose + --proposals-dir
#   - Skill discovery + frontmatter parsing works for the evolution target
#   - Proposals dir is writable
#   - All imports resolve (catches regressions from refactors)
#
# Use T4 (full E2E with real optimization) only for manual weekly verification —
# it's expensive (~4min, real LLM spend). Nightly cron should use T1 + T5.
run_t5() {
  hdr "T5 — propose-mode structural (dry-run, zero tokens)"

  skill="github-code-review"
  repo="$HOME/.hermes/hermes-agent"
  prop_dir="$(pwd)/proposals/_smoke_t5"
  rm -rf "$prop_dir"
  mkdir -p "$prop_dir"

  # Confirm proposals dir is writable BEFORE invoking Python.
  if ! touch "$prop_dir/.writable" 2>/dev/null; then
    fail "proposals dir not writable: $prop_dir"
    return
  fi
  rm -f "$prop_dir/.writable"
  pass "proposals dir writable"

  start=$(date +%s)
  if [ "$VERBOSE" = "1" ]; then
    python -m evolution.skills.evolve_skill \
      --skill "$skill" \
      --hermes-repo "$repo" \
      --dry-run \
      --mode propose \
      --proposals-dir "$prop_dir" \
      --optimizer-model "$MODEL" \
      --eval-model "$MODEL" 2>&1 | tee "$RUN_LOG"
    rc=${PIPESTATUS[0]}
  else
    python -m evolution.skills.evolve_skill \
      --skill "$skill" \
      --hermes-repo "$repo" \
      --dry-run \
      --mode propose \
      --proposals-dir "$prop_dir" \
      --optimizer-model "$MODEL" \
      --eval-model "$MODEL" >"$RUN_LOG" 2>&1
    rc=$?
  fi
  elapsed=$(($(date +%s) - start))

  if [ $rc -ne 0 ]; then
    fail "propose-mode dry-run failed (exit $rc, ${elapsed}s)"
    tail -20 "$RUN_LOG" | sed 's/^/      /'
    return
  fi
  pass "propose-mode dry-run OK (${elapsed}s, zero tokens)"

  # Confirm the dry-run emitted the expected validation markers.
  if grep -q "DRY RUN" "$RUN_LOG"; then
    pass "dry-run confirmation logged"
  else
    fail "dry-run marker missing from output — code path may be broken"
  fi
}

# ═════════════════════════════════════════════════════════════════════
# T4 — Propose-mode E2E: gate-rejected run writes proposal, leaves live skill untouched
# ═════════════════════════════════════════════════════════════════════
# WARNING: T4 is expensive (~4min, real LLM spend). Not for nightly cron.
# Use for manual weekly verification. Nightly cron uses T1 + T5 (both zero-token).
run_t4() {
  hdr "T4 — propose-mode E2E (github-code-review, 1 iter, gate-rejected)"

  skill="github-code-review"
  repo="$HOME/.hermes/hermes-agent"
  live_path="$repo/skills/github/$skill/SKILL.md"

  if [ ! -f "$live_path" ]; then
    fail "live skill not found at $live_path — cannot run T4"
    return
  fi

  # Snapshot live mtime + checksum BEFORE the run.
  mtime_before=$(stat -f '%m' "$live_path" 2>/dev/null || stat -c '%Y' "$live_path")
  sha_before=$(shasum -a 256 "$live_path" | awk '{print $1}')

  prop_dir="$(pwd)/proposals/_smoke_propose"
  rm -rf "$prop_dir"
  mkdir -p "$prop_dir"

  start=$(date +%s)
  if [ "$VERBOSE" = "1" ]; then
    python -m evolution.skills.evolve_skill \
      --skill "$skill" \
      --hermes-repo "$repo" \
      --iterations 1 \
      --optimizer-model "$MODEL" \
      --eval-model "$MODEL" \
      --mode propose \
      --proposals-dir "$prop_dir" 2>&1 | tee "$RUN_LOG"
    rc=${PIPESTATUS[0]}
  else
    python -m evolution.skills.evolve_skill \
      --skill "$skill" \
      --hermes-repo "$repo" \
      --iterations 1 \
      --optimizer-model "$MODEL" \
      --eval-model "$MODEL" \
      --mode propose \
      --proposals-dir "$prop_dir" >"$RUN_LOG" 2>&1
    rc=$?
  fi
  elapsed=$(($(date +%s) - start))

  if [ $rc -ne 0 ]; then
    fail "propose-mode run failed (exit $rc, ${elapsed}s)"
    tail -30 "$RUN_LOG" | sed 's/^/      /'
    return
  fi
  pass "propose-mode run completed (exit 0, ${elapsed}s)"

  # 1. Live skill must NOT have been touched.
  mtime_after=$(stat -f '%m' "$live_path" 2>/dev/null || stat -c '%Y' "$live_path")
  sha_after=$(shasum -a 256 "$live_path" | awk '{print $1}')
  if [ "$mtime_before" = "$mtime_after" ] && [ "$sha_before" = "$sha_after" ]; then
    pass "live skill untouched (mtime + sha256 match)"
  else
    fail "live skill modified by propose mode (mtime $mtime_before→$mtime_after, sha $sha_before→$sha_after)"
  fi

  # 2. Proposal dir must contain expected artifacts.
  prop_run=$(find "$prop_dir/$skill" -maxdepth 1 -type d -name "20*" 2>/dev/null | sort | tail -1)
  if [ -z "$prop_run" ]; then
    fail "no proposal dir under $prop_dir/$skill/ — artifacts not written"
    return
  fi
  pass "proposal dir created: $(basename "$prop_run")"

  for f in decision.json baseline_skill.md evolved_skill.md constraints.json review.md STATUS; do
    if [ -e "$prop_run/$f" ]; then
      pass "$f present"
    else
      fail "$f missing from proposal"
    fi
  done

  # 3. decision.json schema + correctness.
  if [ -s "$prop_run/decision.json" ]; then
    py_out=$(python - <<PY 2>&1
import json, sys
d = json.load(open("$prop_run/decision.json"))
need = ["skill_name","timestamp","mode","baseline_score","evolved_score","improvement","auto_merge","gate_reason"]
missing = [k for k in need if k not in d]
if missing:
    print("MISSING:" + ",".join(missing)); sys.exit(1)
if d["mode"] != "propose":
    print(f"BAD MODE: {d['mode']}"); sys.exit(1)
if d["auto_merge"] is not False:
    print(f"BAD AUTO_MERGE: {d['auto_merge']} (propose mode must be false)"); sys.exit(1)
print(f"OK mode={d['mode']} auto_merge={d['auto_merge']} Δ={d['improvement']:+.3f}")
PY
)
    if [ $? -eq 0 ]; then
      pass "decision.json schema — $py_out"
    else
      fail "decision.json schema — $py_out"
    fi
  fi

  # 4. No .bak files created under live skill dir (propose must never touch backups).
  bak_count=$(find "$(dirname "$live_path")" -maxdepth 1 -name "*.bak.*" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$bak_count" = "0" ]; then
    pass "no .bak files created in live skill dir"
  else
    fail ".bak files found — propose mode should never create backups ($bak_count found)"
  fi
}

# ── dispatch ─────────────────────────────────────────────────────────
# Default ("") runs T1 + T5 — the fast, zero-token nightly profile.
# "full" runs everything including the expensive T2/T3/T4 tiers.
case "$TIER" in
  t1|T1) run_t1 ;;
  t2|T2) run_t2 ;;
  t3|T3) run_t3 ;;
  t4|T4) run_t4 ;;
  t5|T5) run_t5 ;;
  all|"") run_t1; run_t5 ;;
  full)  run_t1; run_t2; run_t3; run_t4; run_t5 ;;
  *) echo "unknown tier: $TIER (use t1|t2|t3|t4|t5|all|full)"; exit 2 ;;
esac

# ── summary ──────────────────────────────────────────────────────────
hdr "summary"
if [ $FAILS -eq 0 ]; then
  printf "${C_G}ALL CHECKS PASSED${C_0} — pipeline ready for nightly crons.\n"
  exit 0
else
  printf "${C_R}$FAILS FAILURE(S)${C_0} — fix before arming crons.\n"
  exit 1
fi
