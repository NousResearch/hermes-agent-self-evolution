# 🧬 Hermes Agent Self-Evolution

**Evolutionary self-improvement for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Hermes Agent Self-Evolution uses [DSPy](https://github.com/stanfordnlp/dspy) + [GEPA](https://github.com/gepa-ai/gepa) (Genetic-Pareto Prompt Evolution) to automatically evolve and optimize Hermes Agent's skills, tool descriptions, system prompts, and code — producing measurably better versions through reflective evolutionary search.

**No GPU training required.** Everything operates via API calls — mutating text, evaluating results, and selecting the best variants. ~$2–10 per optimization run.

---

## How It Works

```
Read current skill/prompt/tool ──► Generate eval dataset
                                        │
                                        ▼
                                   GEPA Optimizer ◄── Execution traces
                                        │                    ▲
                                        ▼                    │
                                   Candidate variants ──► Evaluate
                                        │
                                   Constraint gates (tests, size limits, regression)
                                        │
                                        ▼
                     Best variant ──► Proposal (review queue) ──► Human approves ──► PR
```

GEPA reads execution traces to understand *why* things fail (not just that they failed), then proposes targeted improvements. ICLR 2026 Oral, MIT licensed.

---

## Quick Start

```bash
# Install
git clone https://github.com/NousResearch/hermes-agent-self-evolution.git
cd hermes-agent-self-evolution
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Point at your hermes-agent repo
export HERMES_AGENT_REPO=~/.hermes/hermes-agent

# Dry-run a skill (no LLM cost, confirms plumbing)
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --dry-run --mode propose \
    --optimizer-model openai/cx/gpt-5.4 \
    --eval-model openai/cx/gpt-5.4

# Real propose-mode run (writes proposal for human review)
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --mode propose \
    --optimizer-model openai/cx/gpt-5.4 \
    --eval-model openai/cx/gpt-5.4
```

---

## Modes

| Mode | Behavior | When to use |
|------|----------|-------------|
| `propose` *(default, safe)* | Writes a candidate to `proposals/` for human review. Never touches source files. | Nightly cron, routine evolution |
| `auto` | Auto-merges the candidate if it beats baseline and passes all gates. Writes timestamped `.bak` backup before overwriting. | Only when you trust the metric |
| `--dry-run` | Exits before any LLM call. Validates CLI, skill discovery, frontmatter, proposals dir. ~2s, zero tokens. | Smoke testing, CI |

---

## Nightly Workflow

`nightly.sh` runs three phases end-to-end, logged to `logs/nightly-<ts>.log`:

1. **Smoke preflight** — `bash smoke_test.sh t1 + t5` (~10s, zero tokens).
   Fails fast if CLI, skill discovery, or frontmatter is broken.
2. **Evolve** — runs `evolve_skill` in the configured mode (default `propose`).
   Emits proposal to `proposals/<skill>-<ts>/`.
3. **Digest** — `evolution.review.digest` builds a human-readable markdown
   summary of all proposals in the last 24h to `logs/digests/YYYY-MM-DD.md`.
   Offline, deterministic, no LLM calls.

**Environment variables:**

| Var | Default | Purpose |
|-----|---------|---------|
| `SKILL` | `github-code-review` | Target skill slug |
| `ITERATIONS` | `10` | GEPA generations |
| `MODE` | `propose` | `propose` \| `auto` |
| `OPTIMIZER_MODEL` | `openai/cx/gpt-5.4` | Optimizer LLM |
| `EVAL_MODEL` | `openai/cx/gpt-5.4` | Eval LLM |
| `DIGEST_WINDOW_HOURS` | `24` | Digest lookback window |
| `SKIP_SMOKE` | `` | Set `1` to skip phase 1 |
| `SKIP_EVOLVE` | `` | Set `1` to skip phase 2 (digest-only runs) |
| `SKIP_DIGEST` | `` | Set `1` to skip phase 3 |

**Schedule:**

```bash
# Cron (Asia/Tehran, 07:00 local)
0 7 * * * cd ~/.hermes/self-evolution && bash nightly.sh
```

---

## Reviewing Proposals

Every propose-mode run writes a proposal bundle to
`proposals/{skill_name}/{YYYYMMDD_HHMMSS}/` containing:

- `baseline_skill.md` — original skill text
- `evolved_skill.md` — reassembled evolved skill
- `diff.patch` — unified diff
- `decision.json` — gate decision + scores + delta
- `constraints.json` — per-constraint results
- `review.md` — human-readable summary
- `STATUS` — `PENDING` | `APPROVED` | `REJECTED`

Use the `ProposalReviewer` CLI to triage:

```bash
# List all pending proposals (status + delta)
python -m evolution.review.proposal_reviewer list

# Show a specific proposal (prints review.md)
python -m evolution.review.proposal_reviewer show github-code-review 20260418_171916

# View the full diff
python -m evolution.review.proposal_reviewer diff github-code-review 20260418_171916

# Approve (writes evolved skill back to hermes-agent; timestamped .bak)
python -m evolution.review.proposal_reviewer approve github-code-review 20260418_171916

# Reject (marks STATUS=REJECTED, optional reason)
python -m evolution.review.proposal_reviewer reject github-code-review 20260418_171916 --reason "too verbose"
```

Approval reuses `write_back_skill(..., mode='auto', auto_merge=True)` so the
atomic overwrite + `.bak` safety path is identical to `auto` mode runs.

---

## Smoke Tiers

`smoke_test.sh` has five tiers — run subsets depending on what you're
validating:

| Tier | What it checks | Cost | Time |
|------|----------------|------|------|
| **t1** | Dry-run on top 5 skills — CLI, skill discovery | 0 tokens | ~8s |
| **t2** | Dataset builder end-to-end | 0 tokens | ~2s |
| **t3** | Constraint validator (size, frontmatter, drift) | 0 tokens | ~1s |
| **t4** | Full GEPA optimization + regression eval on one skill | ~$0.30 | ~4min |
| **t5** | Propose-mode structural (dry-run, proposals dir writable) | 0 tokens | ~2s |

```bash
bash smoke_test.sh           # default: t1 + t5 (zero-token, ~10s)
bash smoke_test.sh full      # all tiers including t4 (hits LLM)
bash smoke_test.sh t5        # single tier
```

The default `bash smoke_test.sh` is what nightly cron runs. Use `full`
weekly or before a release to exercise the expensive t4 path.

---

## What It Optimizes

| Phase | Target | Engine | Status |
|-------|--------|--------|--------|
| **Phase 1** | Skill files (`SKILL.md`) | DSPy + GEPA | ✅ Implemented |
| **Phase 2** | Tool descriptions | DSPy + GEPA | 🔲 Planned |
| **Phase 3** | System prompt sections | DSPy + GEPA | 🔲 Planned |
| **Phase 4** | Tool implementation code | Darwinian Evolver | 🔲 Planned |
| **Phase 5** | Continuous improvement loop | Automated pipeline | 🔲 Planned |

---

## Engines

| Engine | What It Does | License |
|--------|-------------|---------|
| **[DSPy](https://github.com/stanfordnlp/dspy) + [GEPA](https://github.com/gepa-ai/gepa)** | Reflective prompt evolution — reads execution traces, proposes targeted mutations | MIT |
| **[Darwinian Evolver](https://github.com/imbue-ai/darwinian_evolver)** | Code evolution with Git-based organisms | AGPL v3 (external CLI only) |

---

## Guardrails

Every evolved variant must pass:

1. **Full test suite** — `pytest tests/ -q` must pass 100% (218 tests).
2. **Size limits** — Skills ≤15KB, tool descriptions ≤500 chars.
3. **Frontmatter validation** — YAML front-matter preserved and valid.
4. **Semantic preservation** — Name/description slugs unchanged.
5. **Regression gate** — In `auto` mode, candidate must beat baseline on
   held-out eval set. In `propose` mode, regression does NOT fail the run —
   writing a proposal for review is the success path.
6. **Backups** — `auto` mode writes a timestamped `.bak` before overwrite.
7. **Human review** — `propose` mode requires explicit approval via the
   `ProposalReviewer` CLI.

---

## Documentation

- **[PLAN.md](PLAN.md)** — Full architecture plan, roadmap, design decisions.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Module map, data flow, extension points.
- **[docs/OPERATIONS.md](docs/OPERATIONS.md)** — Runbook: nightly cron, proposal review, troubleshooting.

---

## Layout

```
self-evolution/
├── evolution/
│   ├── core/            # Config, constraints, datasets, fitness, proposals, regression, write-back
│   ├── skills/          # Phase 1: skill evolver (evolve_skill.py, skill_module.py)
│   ├── review/          # Proposal reviewer CLI + nightly digest
│   ├── prompts/         # Phase 3 (planned)
│   ├── tools/           # Phase 2 (planned)
│   ├── code/            # Phase 4 (planned)
│   └── monitor/         # Phase 5 (planned)
├── tests/               # 218 tests, all green
├── smoke_test.sh        # 5-tier smoke harness
├── nightly.sh           # 3-phase nightly orchestrator
├── run_evolution.sh     # Legacy single-shot runner
└── proposals/           # Proposal review queue (gitignored)
```

---

## License

MIT — © 2026 Nous Research
