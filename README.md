# 🧬 Hermes Agent Self-Evolution

**Evolutionary self-improvement for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Hermes Agent Self-Evolution uses DSPy + GEPA (Genetic-Pareto Prompt Evolution) to automatically evolve and optimize Hermes Agent's skills, tool descriptions, system prompts, and code — producing measurably better versions through reflective evolutionary search.

**No GPU training required.** Everything operates via API calls — mutating text, evaluating results, and selecting the best variants. ~$2-10 per optimization run.

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
                                   Constraint gates (tests, size limits, benchmarks)
                                        │
                                        ▼
                                   Best variant ──► PR against hermes-agent
```

GEPA reads execution traces to understand *why* things fail (not just that they failed), then proposes targeted improvements. ICLR 2026 Oral, MIT licensed.

## Quick Start

```bash
# Install
git clone https://github.com/NousResearch/hermes-agent-self-evolution.git
cd hermes-agent-self-evolution
pip install -e ".[dev]"

# Point at your hermes-agent repo
export HERMES_AGENT_REPO=~/.hermes/hermes-agent

# Evolve a skill (synthetic eval data)
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source synthetic

# Or use real session history from Claude Code, Copilot, and Hermes
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source sessiondb
```

## What It Optimizes

| Phase | Target | Engine | Status |
|-------|--------|--------|--------|
| **Phase 1** | Skill files (SKILL.md) | DSPy + GEPA | ✅ Implemented |
| **Phase 2** | Tool descriptions | DSPy + GEPA | 🟡 Candidate-only gate/report contract implemented |
| **Phase 3** | System prompt sections | DSPy + GEPA | 🔲 Planned |
| **Phase 4** | Tool implementation code | Darwinian Evolver | 🔲 Planned |
| **Phase 5** | Continuous improvement loop | Automated pipeline | 🔲 Planned |

## Engines

| Engine | What It Does | License |
|--------|-------------|---------|
| **[DSPy](https://github.com/stanfordnlp/dspy) + [GEPA](https://github.com/gepa-ai/gepa)** | Reflective prompt evolution — reads execution traces, proposes targeted mutations | MIT |
| **[Darwinian Evolver](https://github.com/imbue-ai/darwinian_evolver)** | Code evolution with Git-based organisms | AGPL v3 (external CLI only) |

## Guardrails

Every evolved variant must pass:
1. **Full test suite** — `pytest tests/ -q` must pass 100%
2. **Size limits** — Skills ≤15KB, tool descriptions ≤500 chars
3. **Caching compatibility** — No mid-conversation changes
4. **Semantic preservation** — Must not drift from original purpose
5. **PR review** — All changes go through human review, never direct commit

## Phase 2 Candidate-Only Report Contract

Phase 2 tool-description work currently emits review artifacts only. It does **not** patch active Hermes Agent tool schemas, registry entries, source files, or runtime configuration.

```bash
python -m evolution.tools.evolve_tool_descriptions \
    --hermes-repo ~/.hermes/hermes-agent \
    --output-dir output/tool-description/<run-name>
```

Each run writes:
- `inventory.json` — read-only Hermes tool inventory snapshot.
- `candidate_descriptions.json` — generated candidate descriptions plus normalized parameter descriptions.
- `candidate.diff` — baseline-vs-candidate description diff.
- `candidate_only_report.json` — canonical Phase 2 report intended for review, smoke checks, and CI/automation gates.

`candidate_only_report.json` has these top-level contract fields:

| Field | Contract |
|-------|----------|
| `phase` | Currently `"2D"` for candidate generation plus formal cross-tool gate. |
| `mode` | Always `"candidate-only"`. |
| `apply_ready` | Always `false`; the report must not contain an apply payload. |
| `summary` | Human-readable run summary. |
| `candidate_count` | Number of candidate tool descriptions evaluated. |
| `metrics` | Candidate-quality metrics only: selection accuracy, wrong-tool avoidance, cue coverage, constraint pass rate, per-case results, and candidate-quality warnings. |
| `candidates` | Baseline/candidate description pairs, normalized parameter descriptions, and `description_delta`. |
| `phase_index_executed` | Phase markers included in the run, currently `["2A", "2B", "2C", "2D"]`. |
| `phase2d_gate` | Formal pass/fail gate result. The candidate-generation CLI exits non-zero when `phase2d_gate.passed` is `false`; the schema smoke check validates structure and can accept a structurally valid failed-gate report. |
| `inventory_metadata` | Inventory/environment metadata, including import warnings that are **not** candidate-quality warnings. |
| `artifacts` | Paths to `inventory.json`, `candidate_descriptions.json`, and `candidate.diff`. |

The formal Phase 2D gate now uses the expanded 45-case golden set unless callers explicitly supply another case set. Default thresholds are:
- `min_case_count = 45`
- `min_selection_accuracy = 0.70`
- `min_wrong_tool_avoidance = 0.70`
- `max_per_tool_regression = 0.0`

Candidate-quality warnings and environment/import warnings must stay separate:
- `metrics.warnings` and `phase2d_gate.candidate_metrics.warning_count` count only candidate-quality issues.
- Optional dependency import issues, such as `tools.browser_dialog_tool` missing `websockets`, are recorded under `inventory_metadata.import_warnings` with `candidate_quality: false`.
- `inventory_metadata.candidate_quality_warnings_are_separate` must be `true` whenever inventory metadata is present.

Run the lightweight schema smoke check before wiring reports into downstream automation:

```bash
python -m evolution.tools.report_contract output/tool-description/<run-name>/candidate_only_report.json
# or, after package install:
hse-validate-tool-report output/tool-description/<run-name>/candidate_only_report.json
```

Phase 2E automation readiness is covered by `.github/workflows/phase2-tool-description-gate.yml`. The workflow runs the focused Phase 2 tool-description tests, builds a deterministic synthetic inventory from the 45-case default golden set, runs the candidate-only Phase 2D generator, validates `candidate_only_report.json`, and asserts `phase2d_gate.passed == true`, `min_case_count == 45`, and `apply_ready == false`. Negative tests also prove that the generator CLI exits non-zero on failed gates while structurally valid failed-gate reports remain readable by the report-contract smoke checker.

Phase 2E SessionDB mining is treated as a privacy-safe holdout source, not an automatic golden-set mutation. A local scan of Hermes `state.db` showed recurring shell/file-operation anti-patterns (`tail`/`head`/`cat`, `grep`/`rg`/`find`, `ls`/`tree`, `sed`/`awk`, and echo/heredoc writing), but these are mostly already represented by the 45-case default gate and raw session prompts can contain private context. The committed fixture `datasets/golden/tool-description/session_misfire_holdout.jsonl` therefore contains generalized, sanitized cases derived from those patterns. Use `load_tool_selection_cases(...)` to evaluate that holdout explicitly; the default Phase 2D gate remains the stable 45-case set.

## Full Plan

See [PLAN.md](PLAN.md) for the complete architecture, evaluation data strategy, constraints, benchmarks integration, and phased timeline.

## License

MIT — © 2026 Nous Research
