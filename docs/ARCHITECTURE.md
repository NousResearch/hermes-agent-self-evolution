# Architecture

High-level design of `hermes-agent-self-evolution`. This doc is the map —
where things live, how they connect, and where to extend.

---

## 1. Goals & Non-Goals

**Goals**

- Evolve Hermes Agent artifacts (skills, tools, prompts, code) via API-only
  optimization — no GPU training, no fine-tuning.
- Safe-by-default: every evolution produces a **proposal** for human review.
  Auto-merge is opt-in and gated.
- Deterministic operational layer: the nightly cron is zero-token on the
  happy path except for the evolution itself.
- Standalone repo — we depend on `hermes-agent` as a read-target but never
  mutate it in-process. All writes go through reviewed proposals.

**Non-goals**

- Runtime hot-swap of evolved artifacts into active sessions.
- Fine-tuning model weights (`BootstrapFinetune` is deliberately unused).
- Mutating `hermes-agent` without review.

---

## 2. Module Map

```
evolution/
├── core/                         # Shared primitives — used by every phase
│   ├── config.py                 # EvolutionConfig dataclass + repo discovery
│   ├── constraints.py            # ConstraintValidator (size, frontmatter, pytest)
│   ├── dataset_builder.py        # SyntheticDatasetBuilder, GoldenDatasetLoader
│   ├── external_importers.py    # Import Claude Code / Copilot / Hermes sessions
│   ├── fitness.py                # LLMJudge, FitnessScore, skill_fitness_metric
│   ├── proposals.py              # ProposalWriter + on-disk layout
│   ├── regression_guard.py       # AutoMergeGate (baseline vs. evolved delta)
│   └── write_back.py             # Atomic skill overwrite + timestamped .bak
│
├── skills/                       # Phase 1 — active
│   ├── skill_module.py           # SKILL.md ⇄ DSPy Module adapter
│   └── evolve_skill.py           # CLI entrypoint — orchestrates the full loop
│
├── review/
│   ├── proposal_reviewer.py     # CLI: list / show / diff / approve / reject
│   └── digest.py                 # Nightly markdown digest (offline, deterministic)
│
├── monitor/                      # Phase 5 (planned) — continuous improvement
├── prompts/                      # Phase 3 (planned) — system prompt evolution
├── tools/                        # Phase 2 (planned) — tool description evolution
└── code/                         # Phase 4 (planned) — Darwinian code evolver
```

### Tests

```
tests/
├── core/
│   ├── test_constraints.py
│   ├── test_external_importers.py
│   ├── test_proposals.py
│   ├── test_regression_guard.py
│   └── test_write_back.py
├── review/
│   ├── test_digest.py
│   └── test_proposal_reviewer.py
└── skills/
    └── test_skill_module.py
```

218 tests, all green. Every core module has dedicated unit coverage; review
layer has integration-style tests over a temp `proposals/` fixture.

---

## 3. Data Flow — Phase 1 Skill Evolution

The whole pipeline lives in `evolution.skills.evolve_skill.evolve()`:

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. find_skill(skill_name, hermes_agent_path)                        │
│    └─> Path to ~/.hermes/hermes-agent/skills/**/SKILL.md            │
│                                                                      │
│ 2. load_skill(skill_path)                                           │
│    └─> { raw, frontmatter, body, name, description }                │
│                                                                      │
│ 3. Build eval dataset  ──── eval_source ∈ {synthetic, golden, sessiondb}
│    ├─ synthetic:   SyntheticDatasetBuilder (LLM judge reads skill)  │
│    ├─ golden:      GoldenDatasetLoader (JSONL)                      │
│    └─ sessiondb:   build_dataset_from_external(...)                 │
│                     ↳ external_importers: claude-code + copilot +   │
│                       hermes session logs → EvalExamples            │
│                                                                      │
│ 4. Wrap as DSPy module                                               │
│    └─> SkillModule(skill_text) — skill body lives as the predictor  │
│        Signature.instructions — this is what GEPA mutates           │
│                                                                      │
│ 5. Baseline evaluate on train split                                  │
│    └─> skill_fitness_metric(example, pred) -> FitnessScore          │
│        composite = 0.5*correctness + 0.3*procedure + 0.2*concise    │
│                                                                      │
│ 6. GEPA optimize                                                     │
│    ├─> dspy.GEPA(metric=skill_fitness_metric, auto='light')         │
│    └─> candidate.signature.instructions = evolved body              │
│                                                                      │
│ 7. ConstraintValidator.validate_all(evolved_skill)                   │
│    ├─ size ≤ max_skill_size (15KB default)                          │
│    ├─ YAML frontmatter parses + name/description preserved          │
│    ├─ body non-empty, not broken                                    │
│    └─ (optional) pytest suite passes  ──── --run-tests              │
│                                                                      │
│ 8. Holdout evaluate baseline vs. evolved                             │
│    └─> delta = evolved_score - baseline_score                       │
│                                                                      │
│ 9. AutoMergeGate.evaluate(baseline, evolved, constraints_passed)     │
│    └─> GateDecision(auto_merge, reason, improvement, regression)    │
│                                                                      │
│10. Branch on mode:                                                   │
│    ├─ propose:                                                       │
│    │   └─> ProposalWriter.write(record)  →  proposals/{skill}/{ts}/ │
│    │       (always — regression is NOT a failure here)              │
│    │                                                                 │
│    └─ auto:                                                          │
│        ├─ if decision.auto_merge:                                    │
│        │   └─> write_back_skill(...) — atomic overwrite + .bak      │
│        └─ else:                                                      │
│            └─> ProposalWriter.write(record) (graceful fallback)      │
└─────────────────────────────────────────────────────────────────────┘
```

### Proposal on-disk layout

```
proposals/{skill_name}/{YYYYMMDD_HHMMSS}/
├── baseline_skill.md      # verbatim original
├── evolved_skill.md       # frontmatter + evolved body
├── diff.patch             # unified diff
├── decision.json          # GateDecision + scores + delta + metadata
├── constraints.json       # [{name, passed, message, details}, …]
├── review.md              # human-readable summary + approval checklist
└── STATUS                 # PENDING | APPROVED | REJECTED
```

Layout is intentionally `pathlib`-only: no DB, no index. The reviewer CLI
walks `proposals/` directly.

---

## 4. Key Abstractions

### 4.1 `SkillModule` — the DSPy adapter

The bridge that lets a plain-markdown `SKILL.md` be optimized by DSPy:

- The skill body becomes the `instructions` field of a DSPy `Signature`.
- GEPA treats it as a mutable prompt parameter.
- After optimization, `module.predictor.signature.instructions` is the
  evolved body.
- Frontmatter is preserved verbatim and re-attached at write time.

Why: DSPy has no native "optimize a markdown file" concept; `SkillModule`
is the minimal wrapper that makes a skill file a first-class optimization
target without forking DSPy.

### 4.2 `ConstraintValidator` — the safety gate

Every evolved artifact must pass these before it's eligible for merge OR
proposal:

| Constraint | Check | Rejects if |
|------------|-------|-----------|
| `size` | `len(evolved) ≤ max_skill_size` | Skill exceeds 15KB |
| `frontmatter_valid` | YAML parses | Broken/missing frontmatter |
| `name_preserved` | slug unchanged | Name/description mutated |
| `body_nonempty` | body has content | GEPA returned empty string |
| `pytest` *(opt-in)* | `pytest tests/ -q` exits 0 | Any test fails |

Returns `List[ConstraintResult]`. A single `passed=False` rejects the
candidate.

### 4.3 `AutoMergeGate` — the merge decision

Decides auto-merge based on baseline vs. evolved holdout scores:

- `constraints_passed=False` → reject
- `delta < -regression_tolerance` → reject, regression=True
- `delta < min_improvement` → reject, no regression
- else → auto_merge=True

Always writes a `GateDecision`; `proposals.py` serializes this into
`decision.json`.

### 4.4 `ProposalWriter` — the review queue

Writes the on-disk layout above. Uses `difflib.unified_diff` for
`diff.patch`, plain JSON for `decision.json` + `constraints.json`. No
network, no DB.

### 4.5 `write_back_skill` — the auto-merge safety path

`auto` mode and `ProposalReviewer approve` both go through this function:

1. Compute `.bak` path: `{live_path}.bak.{YYYYMMDD_HHMMSS}`.
2. `shutil.copy2` live → backup.
3. Write evolved body to `live_path`.
4. Return `WriteBackResult(merged=True, live_path, backup_path, reason)`.

This is the single choke point for any mutation of the `hermes-agent`
repo's skill files.

### 4.6 `Digest` — the nightly summarizer

Input: `proposals/` tree. Output: markdown report for the last N hours
(default 24). Scans by `decision.json` mtime, groups by skill, tallies
PENDING/APPROVED/REJECTED, per-proposal sections with score delta, gate
reason, and path for review.

**Deliberately self-contained**: zero LLM calls, zero network, pure
filesystem read. This keeps nightly cron deterministic and cheap.

---

## 5. Operational Layer

### 5.1 `smoke_test.sh` — the preflight

Five tiers, each guarded by a dispatch case:

| Tier | Purpose | Cost | Runtime |
|------|---------|------|---------|
| t1 | Dry-run 5 skills — CLI, skill discovery | 0 tokens | ~8s |
| t2 | Dataset builder end-to-end | 0 tokens | ~2s |
| t3 | Constraint validator | 0 tokens | ~1s |
| t4 | Full GEPA + regression eval | ~$0.30 | ~4min |
| t5 | Propose-mode structural dry-run | 0 tokens | ~2s |

Default (no arg) = `t1 + t5`. `full` = all five. Nightly runs the default.

### 5.2 `nightly.sh` — the orchestrator

Three phases:

1. **Smoke preflight** — `bash smoke_test.sh` (t1+t5). Exits non-zero
   aborts the whole run.
2. **Evolve** — `python -m evolution.skills.evolve_skill ...`. Skipped
   with `SKIP_EVOLVE=1` (used for digest-only dev runs).
3. **Digest** — `python -m evolution.review.digest --hours 24 --output
   logs/digests/YYYY-MM-DD.md`.

All phases log to `logs/nightly-<ts>.log`; digest path is the canonical
artifact the cron reads.

### 5.3 Cron topology

Single Hermes cron (`hermes-self-evolution-nightly`, Asia/Tehran 07:00)
runs `run_evolution.sh`, which activates the venv and invokes
`nightly.sh`. Cron prompt reads the latest digest and delivers a summary
via the Hermes messaging layer — it does NOT re-run any evolution.

---

## 6. Extension Points

### Adding a new evolution phase (tool / prompt / code)

1. Create `evolution/<phase>/` with a module analogous to
   `skills/skill_module.py` (the DSPy adapter) and a CLI analogous to
   `skills/evolve_skill.py`.
2. Reuse `core/` primitives — `ConstraintValidator`, `AutoMergeGate`,
   `ProposalWriter`, `write_back_*`, `FitnessScore`.
3. Add constraints specific to the target (e.g. tool desc ≤500 chars
   lives in `ConstraintValidator` already — reuse).
4. Add `tests/<phase>/` mirroring `tests/skills/`.
5. Wire a smoke tier for it in `smoke_test.sh`.
6. Register a new optional phase in `nightly.sh` behind an env flag
   (`EVOLVE_TOOLS=1`, etc.).

### Adding a constraint

Add a method to `ConstraintValidator` returning `ConstraintResult`, call
it from `validate_all()`, write a unit test in
`tests/core/test_constraints.py`.

### Adding a new eval source

Extend `eval_source` choice in `evolve_skill.py`'s click decorator, add
a builder function in `core/dataset_builder.py` or
`core/external_importers.py`, ensure it returns an `EvalDataset` with
`train_examples` and `holdout_examples`.

### Adding a reviewer CLI command

`evolution/review/proposal_reviewer.py` uses `argparse` subcommands.
Each command is a top-level function `cmd_<name>(args)`; register in
the parser block at bottom of file and add a unit test in
`tests/review/test_proposal_reviewer.py`.

---

## 7. Design Decisions

### Why propose-mode is the default

`auto` mode is powerful but hazardous: a plausible-looking but
subtly-wrong mutation could silently degrade a skill used in every
session. Propose-mode makes the human the last gate. The reviewer CLI
makes approval a 30-second operation, so the cost of the safety is
minimal.

### Why the digest is offline

If digest called an LLM, the nightly cron would have two billing
surfaces. Keeping it deterministic means the only variable cost is the
evolution itself, and the digest can regenerate cheaply from historical
proposal data for audits.

### Why `propose-mode regression != failure`

In propose mode, the whole point is to queue the candidate for review.
Exit-code failure on regression would make the nightly cron look broken
when it's actually working correctly — writing a proposal IS the success
path. In `auto` mode, regression must still fail the run because the
gate is blocking a merge.

### Why `write_back_skill` always creates a `.bak`

Recovery is `cp {live}.bak.{ts} {live}`. No git operations, no network,
works even if `hermes-agent` is in a dirty state. The `.bak` naming
scheme is tombstone-style (timestamp, never overwritten) so you can
diff across historical merges.

### Why we don't hot-swap into active sessions

Skills are loaded at session start and baked into prompt caches. Mid-
session swap breaks caching and gives inconsistent behavior between
turns. Evolution is a background process; the next fresh session picks
up the merged version.

---

## 8. Dependencies

| Dep | Role |
|-----|------|
| `dspy` | Predictor abstraction + GEPA optimizer |
| `gepa` | Reflective prompt evolution |
| `click` | CLI for `evolve_skill.py` |
| `rich` | Console output formatting |
| `pyyaml` | Frontmatter parsing |
| `pytest` | Optional constraint (`--run-tests`) |

No runtime dependency on `hermes-agent` as a Python package — we only
read its repo filesystem via `HERMES_AGENT_REPO`.

---

## 9. File References

- Config: [`evolution/core/config.py`](../evolution/core/config.py)
- Constraints: [`evolution/core/constraints.py`](../evolution/core/constraints.py)
- Proposals: [`evolution/core/proposals.py`](../evolution/core/proposals.py)
- Regression gate: [`evolution/core/regression_guard.py`](../evolution/core/regression_guard.py)
- Write-back: [`evolution/core/write_back.py`](../evolution/core/write_back.py)
- Skill module: [`evolution/skills/skill_module.py`](../evolution/skills/skill_module.py)
- Skill evolver: [`evolution/skills/evolve_skill.py`](../evolution/skills/evolve_skill.py)
- Reviewer CLI: [`evolution/review/proposal_reviewer.py`](../evolution/review/proposal_reviewer.py)
- Digest: [`evolution/review/digest.py`](../evolution/review/digest.py)
- Smoke harness: [`smoke_test.sh`](../smoke_test.sh)
- Nightly: [`nightly.sh`](../nightly.sh)
