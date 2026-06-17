# Operations Guide

Day-to-day operation of `hermes-agent-self-evolution`: running evolutions,
reviewing proposals, reading nightly digests, and troubleshooting.

If you're looking for **what the system does** and **how the code is
organized**, see [`ARCHITECTURE.md`](ARCHITECTURE.md). This doc is for the
operator.

---

## 1. Daily Workflow

The system runs autonomously on a nightly cron. Your daily touch-points are:

| Action | When | Tool |
|--------|------|------|
| Read digest | Morning | Check Telegram delivery from cron, or `cat logs/digests/YYYY-MM-DD.md` |
| Review proposals | If digest shows PENDING items | `python -m evolution.review.proposal_reviewer ...` |
| Approve or reject | Per-proposal decision | `reviewer approve <skill> <ts>` / `reject ...` |
| Inspect failures | If cron reports ❌ | `tail -100 logs/nightly-<ts>.log` |

Everything else — smoke preflight, evolution, digest generation — happens
without you.

---

## 2. Nightly Cron

### Schedule

- Job name: `hermes-self-evolution-nightly`
- Job ID: `cdfe2c858ba5`
- Schedule: `0 7 * * *` (Asia/Tehran local time)
- Delivery: `origin` — posts digest to the M K Telegram DM

### What it runs

The cron prompt invokes the pipeline directly:

```bash
cd ~/.hermes/self-evolution && bash nightly.sh 2>&1 | tail -40
```

Then reads `logs/digests/YYYY-MM-DD.md` and delivers it via Telegram with a
status header.

### Pipeline phases (inside `nightly.sh`)

1. **Phase 1 — smoke preflight.** `smoke_test.sh t1` (dry-run 5 skills)
   + `smoke_test.sh t5` (propose-mode structural check). Both zero-token,
   total ~10s. Non-zero exit aborts the run.
2. **Phase 2 — evolve.** `evolve_skill --skill github-code-review --mode
   propose`. Uses `cx/gpt-5.4` for both optimizer and eval model. Non-zero
   exit is **logged but does not abort** — in propose-mode, the proposal
   lands regardless and is surfaced in the digest.
3. **Phase 3 — digest.** `evolution.review.digest --hours 24 --output
   logs/digests/YYYY-MM-DD.md`. Deterministic, offline, zero-token.
4. **Phase 4 — delivery (optional).** Only runs if `DELIVER=1` and
   `DELIVER_CMD` are set. Currently unused — the cron prompt handles
   delivery instead.

### Managing the cron

```bash
# List all Hermes crons
cronjob list

# Run nightly immediately (don't wait for 07:00)
cronjob run cdfe2c858ba5

# Pause nightly (e.g. before a disruptive refactor)
cronjob pause cdfe2c858ba5
cronjob resume cdfe2c858ba5
```

---

## 3. Running Evolutions Manually

### Dry-run a skill (zero token cost)

```bash
cd ~/.hermes/self-evolution
source venv/bin/activate
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --dry-run
```

Validates setup (skill discovery, config load) without calling any LLM.

### Full propose-mode run

```bash
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --optimizer-model openai/cx/gpt-5.4 \
    --eval-model openai/cx/gpt-5.4 \
    --mode propose
```

- Lands a proposal under `proposals/github-code-review/YYYYMMDD_HHMMSS/`.
- Never overwrites the live skill.
- Regression is NOT a failure — the proposal queues regardless.

### Auto-merge run (gated)

```bash
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --mode auto \
    --min-improvement 0.05 \
    --regression-tolerance 0.01 \
    --run-tests
```

- Auto-merges only if: `constraints_passed=True`, `delta ≥ min_improvement`,
  and `--run-tests` pytest passes.
- Otherwise falls back to a proposal.
- Writes `.bak.YYYYMMDD_HHMMSS` alongside the live skill before overwrite.

### Full CLI reference

```
--skill TEXT                     Name of the skill to evolve  [required]
--iterations INTEGER             Number of GEPA iterations (default: 10)
--eval-source [synthetic|golden|sessiondb]
                                 Source for evaluation dataset
--dataset-path TEXT              Path to existing eval dataset (JSONL)
--optimizer-model TEXT           Model for GEPA reflections (default: openai/gpt-4.1)
--eval-model TEXT                Model for evaluations (default: openai/gpt-4.1-mini)
--hermes-repo TEXT               Path to hermes-agent repo
--run-tests                      Run full pytest suite as constraint gate
--dry-run                        Validate setup without running optimization
--mode [propose|auto]            propose: write to review queue
                                 auto: overwrite live skill if gate passes
--min-improvement FLOAT          Minimum holdout Δ for auto-merge (default: 0.02)
--regression-tolerance FLOAT     Negative Δ tolerance before flagging regression (default: 0.01)
--proposals-dir TEXT             Where propose-only writes land (default: proposals)
```

---

## 4. Proposal Review

### List pending proposals

```bash
python -m evolution.review.proposal_reviewer list
```

Shows: skill name, timestamp, STATUS, decision summary.

### Inspect a proposal

```bash
# Full human-readable review
python -m evolution.review.proposal_reviewer show github-code-review 20260418_071200

# Just the diff
python -m evolution.review.proposal_reviewer diff github-code-review 20260418_071200
```

### Approve (write-back to live skill)

```bash
python -m evolution.review.proposal_reviewer approve github-code-review 20260418_071200
```

- Invokes `write_back_skill(mode='auto', auto_merge=True)`.
- Creates `.bak.<ts>` alongside the live skill.
- Overwrites the live file with the evolved version.
- Marks the proposal `STATUS=APPROVED`.

### Reject

```bash
python -m evolution.review.proposal_reviewer reject github-code-review 20260418_071200 \
    --reason "hallucinated pytest flag that doesn't exist"
```

- Marks the proposal `STATUS=REJECTED`.
- Records reason in `decision.json`.
- Leaves live skill untouched.

### Arguments recap

| Command | Positional args | Flags |
|---------|-----------------|-------|
| `list` | — | `--proposals-dir` |
| `show` | `<skill> <timestamp>` | `--proposals-dir` |
| `diff` | `<skill> <timestamp>` | `--proposals-dir` |
| `approve` | `<skill> <timestamp>` | `--proposals-dir`, `--hermes-repo` |
| `reject` | `<skill> <timestamp>` | `--proposals-dir`, `--reason` |

`<timestamp>` is the directory name under `proposals/<skill>/`, e.g.
`20260418_071200`.

---

## 5. Digest Format

Generated at `logs/digests/YYYY-MM-DD.md`. Covers the last 24 hours by
default.

### Top-level sections

1. **Summary counts** — total / PENDING / APPROVED / REJECTED.
2. **Per-skill breakdown** — grouped list with score deltas.
3. **Per-proposal details** — timestamp, gate decision, improvement Δ,
   regression flag, constraint results, path to review bundle.

### Regenerate a digest manually

```bash
python -m evolution.review.digest \
    --hours 24 \
    --output /tmp/digest.md \
    --format markdown

# Or get JSON for tooling
python -m evolution.review.digest --hours 168 --format json
```

`--hours 168` gives you a weekly view.

---

## 6. Smoke Test Harness

### Tiers

| Tier | What | Cost | Runtime |
|------|------|------|---------|
| t1 | Dry-run 5 skills — CLI + skill discovery | 0 tokens | ~8s |
| t2 | Dataset builder end-to-end | 0 tokens | ~2s |
| t3 | Constraint validator | 0 tokens | ~1s |
| t4 | Full GEPA + regression eval | ~$0.30 | ~4min |
| t5 | Propose-mode structural dry-run | 0 tokens | ~2s |

### Running

```bash
# Default (nightly pair) — t1 + t5, ~10s, zero tokens
bash smoke_test.sh

# Single tier
bash smoke_test.sh t4

# All tiers (including expensive t4) — weekly sanity check
bash smoke_test.sh full
```

### When to use `full`

Run `bash smoke_test.sh full` weekly or before a substantive code change.
It exercises the real GEPA loop end-to-end and catches integration-level
regressions the fast tiers miss.

---

## 7. Troubleshooting

### Nightly cron failed

Check the delivered Telegram status header. If ❌ or ⚠️:

```bash
# Latest nightly log
ls -t ~/.hermes/self-evolution/logs/nightly-*.log | head -1 | xargs tail -80
```

Common causes:

- **Smoke preflight failure.** Check `logs/smoke/smoke-<ts>.log`. Usually
  means a skill was renamed/deleted in `hermes-agent` or the CLI broke.
  Re-run: `bash smoke_test.sh`.
- **Evolve exit non-zero.** Non-fatal in propose-mode; still produces a
  digest. In auto-mode it means the gate blocked a merge — expected, not
  a bug.
- **Digest failed.** Check `NIGHTLY_LOG`. Usually a stat failure on
  `proposals/` — fix with `mkdir -p proposals/ logs/digests/`.

### LLM call errors

The pipeline uses the local `cx/gpt-5.4` endpoint at
`http://localhost:20128/v1`. If evolution logs show connection refused:

```bash
# Verify the gateway is up
curl -s http://localhost:20128/v1/models | head

# If down, restart your LLM gateway (outside the scope of this repo)
```

### Proposals not appearing

```bash
# Check the tree structure
find proposals -type f -name 'STATUS' | head

# Expected layout:
# proposals/<skill>/<YYYYMMDD_HHMMSS>/STATUS
```

If `proposals/` is empty but evolve runs complete, check
`logs/evolve-<skill>-<ts>.log` for a "wrote proposal" line near the end.

### Reviewer CLI "not found"

The timestamp must be the **exact** directory name. Use `list` to copy
the right one:

```bash
python -m evolution.review.proposal_reviewer list
```

### Test suite failing

```bash
cd ~/.hermes/self-evolution
source venv/bin/activate
pytest tests/ -x -q

# If a specific test fails, run it in isolation with verbose output
pytest tests/core/test_write_back.py -v
```

Baseline expectation: **218/218 green**.

### Rolling back an accidental auto-merge

`write_back_skill` always creates a `.bak.<ts>` file alongside the live
skill. To roll back:

```bash
LIVE=~/.hermes/hermes-agent/skills/github/github-code-review/SKILL.md
BACKUP=$(ls -t ${LIVE}.bak.* | head -1)
cp "$BACKUP" "$LIVE"
```

Backups are timestamp-named and never overwritten, so all historical
versions remain.

### Stale `run_evolution.sh`

`run_evolution.sh` is a legacy shim that invokes `evolve_skill.py`
directly (no smoke, no digest). The nightly cron uses `nightly.sh`, not
this. Use `nightly.sh` for anything production-facing. `run_evolution.sh`
is retained for quick manual iteration only.

---

## 8. Storage & Retention

| Path | Purpose | Retention |
|------|---------|-----------|
| `logs/nightly-*.log` | Full nightly run output | Manual — recommend keep ~30 days |
| `logs/evolve-*-*.log` | Per-evolution run output | Manual |
| `logs/smoke/smoke-*.log` | Smoke preflight output | Manual |
| `logs/digests/YYYY-MM-DD.md` | Daily digest | Keep indefinitely (small) |
| `proposals/<skill>/<ts>/` | Review bundles | Archive REJECTED after review |
| `{live_skill}.bak.<ts>` | Backup before auto-merge | Keep indefinitely |

None of these are in git (`.gitignore` covers `logs/`, `proposals/`,
`output/`, `snapshots/`).

Suggested cleanup cron (not installed by default):

```bash
# Trim nightly logs older than 30 days
find ~/.hermes/self-evolution/logs -name 'nightly-*.log' -mtime +30 -delete
find ~/.hermes/self-evolution/logs -name 'evolve-*.log' -mtime +30 -delete
```

---

## 9. Environment Setup

### Minimal

```bash
cd ~/.hermes/self-evolution
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Required env vars (set in `nightly.sh` and `run_evolution.sh`)

```bash
export OPENAI_API_BASE="http://localhost:20128/v1"
export OPENAI_BASE_URL="http://localhost:20128/v1"   # LiteLLM respects this
export OPENAI_API_KEY="***"                           # dummy OK for local gateway
```

### Override knobs (nightly only)

```bash
SKILL=some-other-skill \
ITERATIONS=5 \
MODE=auto \
WINDOW_HOURS=48 \
bash nightly.sh
```

Other supported env vars: `SKIP_SMOKE=1`, `SKIP_EVOLVE=1` (digest only),
`DELIVER=1` + `DELIVER_CMD=...` (custom delivery).

---

## 10. Quick Reference Card

```bash
# Run nightly now
cd ~/.hermes/self-evolution && bash nightly.sh

# Review
python -m evolution.review.proposal_reviewer list
python -m evolution.review.proposal_reviewer show <skill> <ts>
python -m evolution.review.proposal_reviewer diff <skill> <ts>
python -m evolution.review.proposal_reviewer approve <skill> <ts>
python -m evolution.review.proposal_reviewer reject <skill> <ts> --reason "..."

# Smoke
bash smoke_test.sh          # cheap (default, t1+t5)
bash smoke_test.sh full     # expensive (all tiers)

# Tests
pytest tests/ -q

# Manually regenerate a digest
python -m evolution.review.digest --hours 24 --output /tmp/d.md

# Cron management
cronjob list
cronjob run cdfe2c858ba5
```
