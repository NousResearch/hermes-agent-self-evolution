# 🧬 Hermes Agent Self-Evolution

**Evolutionary self-improvement for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Hermes Agent Self-Evolution uses DSPy + GEPA (Genetic-Pareto Prompt Evolution) to automatically evolve and optimize Hermes Agent's skills, tool descriptions, and system prompt sections — producing measurably better versions through reflective evolutionary search.

**No GPU training required.** Everything operates via API calls — mutating text, evaluating results, and selecting the best variants. ~$2-10 per optimization run.

## How It Works

```
Read artifact + supporting files ──► Build eval set from real sessions
                                             │
                                             ▼
                                        GEPA Optimizer ◄── Judge + size feedback
                                             │                      ▲
                                             ▼                      │
                                       Candidate variants ──────► Score
                                             │
                                   Pareto front (quality × size × tokens × tool calls)
                                             │
                                   Constraint gates (size, structure, references, tests)
                                             │
                                   A/B report vs a noise band ──► SHIP or HOLD
                                             │
                                   PR against hermes-agent ──► canary ──► auto-rollback
```

GEPA reads execution traces to understand *why* things fail (not just that they failed), then proposes targeted improvements. ICLR 2026 Oral, MIT licensed.

## Quick Start

```bash
# Install
git clone https://github.com/NousResearch/hermes-agent-self-evolution.git
cd hermes-agent-self-evolution
pip install -e ".[dev]"

# Point at your hermes-agent repo and your Hermes data directory.
# These are different things: the repo holds code and shipped skills, the
# data dir holds state.db, profiles, cron and your own skills.
export HERMES_AGENT_REPO=~/.hermes/hermes-agent
export HERMES_DATA_DIR=~/.hermes

# Evolve a skill against real session history
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source sessiondb

# Or synthesize an eval set when there is no history yet
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source synthetic
```

> **Running inside the Hermes container?** `HOME` is not where the data lives.
> Set `HERMES_DATA_DIR` to the bind-mounted data directory (e.g. `/opt/data`),
> or the run will tell you which paths it tried and stop.

## What It Optimizes

| Phase | Target | Objective | Status |
|-------|--------|-----------|--------|
| **Phase 1** | Skill files (SKILL.md) | Judge quality × size | ✅ Implemented |
| **Phase 2** | Tool descriptions | Tool-selection accuracy × catalog size | ✅ Implemented |
| **Phase 3** | System prompt sections | Judge quality × per-request cost | ✅ Implemented |
| **Phase 4** | Tool implementation code | Darwinian Evolver via AGPL sidecar | ✅ Implemented |
| **Phase 5** | Continuous improvement loop | Evidence-prioritized rotation | ✅ Implemented |

```bash
# Phase 4 — tool implementation code (needs the AGPL sidecar; see below)
python -m evolution.code.evolve_code --suggest
python -m evolution.code.evolve_code --target agent/tool_executor.py --iterations 5

# Phase 2 — tool descriptions, graded on which tool the agent actually used
hermes tools list --json > tools.json
python -m evolution.tools.evolve_tool --catalog tools.json --iterations 6

# Phase 3 — system prompt sections, read from the live install
python -m evolution.prompts.evolve_prompt --list
python -m evolution.prompts.evolve_prompt --section "Tool use" --iterations 6

# Phase 5 — one scheduled sweep, highest-need skills first
python -m evolution.monitor.run_rotation --skills-per-run 4 --create-pr
```

## Where Evaluation Data Comes From

| Source | What it provides |
|---|---|
| **`state.db`** | Real user/assistant exchanges, per-session tool counts, tokens and outcomes. Retrieval uses the FTS5 index Hermes already builds, so relevance filtering costs nothing. |
| **`verification_evidence.db`** | Exit codes from real test, build and lint runs. Ground truth, not a rubric score. |
| **`cron/executions.db`** | Scheduled-job outcomes, attributed to skills through `jobs.json` — a per-skill success rate measured in production. |
| **`sessions.system_prompt_hash`** | Groups sessions by the exact prompt they ran under, which is what makes canary comparison possible. |
| Claude Code / Copilot | Cold-start fallback when there is no Hermes history yet. |
| Synthetic | Last resort; an LLM writes test cases from the artifact itself. |

Every source reports itself. A source that is missing and a source that is present-but-empty are different failures with different fixes, and the run says which one it hit.

## Guardrails

Every evolved variant must pass:

1. **Size budget** — derived from the installed skill corpus (p90 by default), floored at the artifact's own size so no skill is disqualified for already being what it is. Size is also an objective *during* the search, not only a gate after it.
2. **Growth limit** — 20% for skills, 5% for system prompt sections, which every request pays for.
3. **Structural integrity** — valid frontmatter, non-empty body.
4. **Supporting-file references** — links to reference files must survive the rewrite, and invented paths are rejected.
5. **Test suite** — `pytest tests/` in hermes-agent must pass, with `--run-tests`.
6. **A verdict beyond the noise band** — a delta smaller than run-to-run variance is reported as noise, never as an improvement.
7. **PR review** — `--create-pr` opens a draft PR carrying the constraint report and A/B summary. Nothing is committed directly.

## Reporting

Every run writes `SUMMARY.md` and `report.json` in the readtool eval's format: both arms, observation counts, the noise band, an explicit **SHIP** or **HOLD** verdict, and caveats emitted automatically for unbalanced arms, single-repetition runs, excluded errors, and size growth.

```
| metric       | baseline | evolved |   delta |
|--------------|---------:|--------:|--------:|
| score        |    0.812 |   0.874 |  +0.062 |
| size (chars) |   13,218 |  13,940 |    +722 |

Observations: 6 baseline / 6 evolved. Noise band ±0.021.

**Verdict: SHIP.** +0.062 (+7.6%) beyond the ±0.021 noise band (measured)
```

## Phase 4 and the AGPL boundary

Code evolution uses [darwinian_evolver](https://github.com/imbue-ai/darwinian_evolver), which is AGPL-3.0. It **cannot be invoked as a plain external CLI**: `problems/registry.py` is a hardcoded dict and its CLI restricts `--problem` to that dict's keys, so defining a Hermes problem means subclassing its classes — importing AGPL code.

The AGPL-linked code therefore lives in a separate package, [hermes-evolver-problems](https://github.com/numandev1/hermes-evolver-problems), and the dependency runs one way:

```
hermes-evolver-problems (AGPL) ──imports──▶ darwinian_evolver (AGPL)
hermes-evolver-problems (AGPL) ──imports──▶ this package      (MIT)
this package            (MIT)  ──subprocess──▶ the sidecar
```

A test asserts that nothing under `evolution/` imports `darwinian_evolver`, because a stray import would relicense this project and would not otherwise fail anything.

```bash
git clone https://github.com/numandev1/hermes-evolver-problems
pip install -e ./hermes-evolver-problems     # needs Python >= 3.11
```

### Why code gets a stricter gate

A bad skill edit produces a worse answer; a bad code edit ships a defect into every agent that loads the tool. So a candidate is not scored until it is admitted:

- **Sandboxed.** Checks run against a copy, never the real checkout, with credentials stripped from the environment.
- **Held-out checks.** Visible failures go back to the mutator, because that is how it improves. The full suite and replayed real commands are sealed — they gate admission but their names and output never reach the mutator, so it cannot learn to satisfy the specific checks it can see.
- **Ground truth.** Commands recorded in `verification_evidence.db` are replayed with their real exit codes. Unsafe ones are never replayed.
- **Tests are not evolvable.** `tests/`, `setup.py`, `__init__.py` and migrations are refused as targets — a mutator that can edit the tests can pass any gate it likes.
- **No automatic deployment, ever.** Skill evolution can canary into a live install and roll back. Code cannot: a bad tool implementation is already executing inside the agent before any outcome signal exists. Phase 4 ends at a draft pull request.


## Engines

| Engine | What It Does | License |
|--------|-------------|---------|
| **[DSPy](https://github.com/stanfordnlp/dspy) + [GEPA](https://github.com/gepa-ai/gepa)** | Reflective prompt evolution — reads execution traces, proposes targeted mutations | MIT |
| **[Darwinian Evolver](https://github.com/imbue-ai/darwinian_evolver)** | Code evolution over git-based organisms | AGPL v3 (isolated in a separate sidecar package) |

## Models

Any LiteLLM-supported model works. `openai-codex/<model>` routes through Hermes' own OpenAI-Codex OAuth credentials:

```bash
hermes auth add openai-codex   # once
python -m evolution.skills.evolve_skill --skill my-skill \
    --optimizer-model openai-codex/gpt-5.6-luna \
    --eval-model openai-codex/gpt-5.6-luna
```

## Full Plan

See [PLAN.md](PLAN.md) for the complete architecture, evaluation data strategy, constraints, benchmarks integration, and phased timeline.

## License

MIT — © 2026 Nous Research
