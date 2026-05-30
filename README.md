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
| **Phase 3** | System prompt sections | DSPy + GEPA | 🟡 Design + execution Seed draft + readiness manifest recorded |
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

Phase 2E automation readiness is covered by `.github/workflows/phase2-tool-description-gate.yml`. The workflow runs the focused Phase 2 tool-description tests, builds a deterministic synthetic inventory from the 45-case default golden set, runs the candidate-only Phase 2D generator, validates `candidate_only_report.json`, runs the SessionDB holdout review smoke, and asserts `phase2d_gate.passed == true`, `min_case_count == 45`, and `apply_ready == false`. Negative tests also prove that the generator CLI exits non-zero on failed gates while structurally valid failed-gate reports remain readable by the report-contract smoke checker.

Phase 2E SessionDB mining is treated as a privacy-safe holdout source, not an automatic golden-set mutation. A local scan of Hermes `state.db` showed recurring shell/file-operation anti-patterns (`tail`/`head`/`cat`, `grep`/`rg`/`find`, `ls`/`tree`, `sed`/`awk`, and echo/heredoc writing), but these are mostly already represented by the 45-case default gate and raw session prompts can contain private context. The committed fixture `datasets/golden/tool-description/session_misfire_holdout.jsonl` therefore contains generalized, sanitized cases derived from those patterns. Use `load_tool_selection_cases(...)` to evaluate that holdout explicitly; the default Phase 2D gate remains the stable 45-case set.

Run the held-out candidate improvement/no-regression review after a Phase 2D candidate-only run:

```bash
python -m evolution.tools.heldout_tool_selection_review \
    --inventory-json output/tool-description/<run-name>/inventory.json \
    --candidates-json output/tool-description/<run-name>/candidate_descriptions.json \
    --cases-jsonl datasets/golden/tool-description/session_misfire_holdout.jsonl \
    --output-json output/tool-description/<run-name>/heldout_review.json
# or, after package install:
hse-review-tool-holdout \
    --inventory-json output/tool-description/<run-name>/inventory.json \
    --candidates-json output/tool-description/<run-name>/candidate_descriptions.json \
    --cases-jsonl datasets/golden/tool-description/session_misfire_holdout.jsonl \
    --output-json output/tool-description/<run-name>/heldout_review.json
```

The review remains candidate-only and fails non-zero on aggregate primary-metric regression (`selection_accuracy` / `wrong_tool_avoidance`), any per-tool pass-rate regression, missing holdout tool coverage, inventory/candidate artifact mismatch, or candidate description/parameter length violations. It reports secondary metric deltas, including cue coverage, for human review.

Record the expanded holdout decision after the heldout review:

```bash
python -m evolution.tools.expanded_holdout_decision \
    --holdout-jsonl datasets/golden/tool-description/session_misfire_holdout.jsonl \
    --heldout-review-json output/tool-description/<run-name>/heldout_review.json \
    --output-json reports/phase2e_expanded_holdout_decision.json \
    --output-md reports/phase2e_expanded_holdout_decision.md
# or, after package install:
hse-decide-tool-holdout \
    --holdout-jsonl datasets/golden/tool-description/session_misfire_holdout.jsonl \
    --heldout-review-json output/tool-description/<run-name>/heldout_review.json \
    --output-json reports/phase2e_expanded_holdout_decision.json \
    --output-md reports/phase2e_expanded_holdout_decision.md
```

Current Phase 2E decision: the 45-case default gate plus the 9-case SessionDB holdout is sufficient for candidate-only Phase 2 closeout. A 100+ held-out quality slice is deferred until before any default-gate promotion, active tool-schema apply, or broader Phase 3/benchmark expansion requiring more lexical diversity. The committed decision artifacts are `reports/phase2e_expanded_holdout_decision.json` and `reports/phase2e_expanded_holdout_decision.md`.

Phase 2E benchmark gate decision is recorded in `reports/phase2e_benchmark_gate_decision.json` and `reports/phase2e_benchmark_gate_decision.md`. TBLite/YC-Bench are deferred until Phase 3 execution or active apply because the current candidate-only closeout is already covered by the 45-case formal gate, 9-case SessionDB holdout, heldout review, and CI smoke. Benchmark gates remain required before Phase 3 execution, active tool-schema apply, default-gate promotion, or system-prompt evolution acceptance.

Phase 2E human review checkpoint is recorded in `reports/phase2e_human_review_checkpoint.json` and `reports/phase2e_human_review_checkpoint.md`. Sunwoo authorized the checkpoint via `rec action GO`; active schema/source apply remains separate and requires a human-approved PR or patch.

## Phase 3 System Prompt Evolution Design Plan

Phase 3 system prompt evolution design plan artifacts are recorded in `reports/phase3_system_prompt_evolution_plan.json` and `reports/phase3_system_prompt_evolution_plan.md`, with the design-only Seed at `seeds/phase3_system_prompt_evolution_seed.yaml`.

This is not an execution/apply Seed. It only fixes the scope, acceptance criteria, benchmark gate reactivation boundaries, and non-evolvable sections before any Phase 3 GEPA/DSPy run. Active system-prompt/source apply remains separate and requires a later human-approved PR or patch.

Benchmark gates are not blocking this design-only plan, but TBLite/YC-Bench must be reactivated before Phase 3 execution, system-prompt evolution acceptance, active system-prompt apply, or default-gate promotion.

Phase 3 execution Seed draft artifacts are recorded in `reports/phase3_execution_seed_draft.json` and `reports/phase3_execution_seed_draft.md`, with the draft Seed at `seeds/phase3_system_prompt_evolution_execution_seed_draft.yaml`. The draft fixes benchmark command templates, rollback boundary, and human approval gate before execution. It still does not run GEPA/DSPy, does not run real benchmark commands, and does not approve active system-prompt/source apply.

## Phase 3 Candidate-Only Scaffold

The Phase 3 candidate-only scaffold prepares review artifacts under `output/phase3-system-prompt/<run-id>/` without running GEPA/DSPy, without running real benchmarks, and without editing Hermes Agent prompt source or active runtime configuration.

```bash
python -m evolution.prompts.phase3_candidate_scaffold \
    --baseline-prompt output/phase3-system-prompt/<run-id>/inputs/baseline_system_prompt.json \
    --candidate-prompt output/phase3-system-prompt/<run-id>/inputs/candidate_system_prompt.json \
    --output-dir output/phase3-system-prompt/<run-id>/review/ \
    --dry-run
```

It writes only `baseline_system_prompt.json`, `candidate_system_prompt.json`, `candidate_only_report.json`, and `review_packet.md` under the allowed Phase 3 output root, and rejects pre-existing write targets plus input/output path overlap so prompt inputs remain read-only. The report keeps `candidate_only=true`, `apply_ready=false`, `real_benchmarks_executed=false`, and rejects non-evolvable section changes. This is a candidate scaffold only; real benchmark execution remains required before optimizer acceptance, active apply, or default-gate promotion.

## Phase 3 Benchmark Adapter Contract

The TBLite/YC-Bench command templates now resolve to runnable **read-only dry-run fixture** adapters. These commands validate prompt artifacts and committed fixtures, write only the requested JSON report, make no external calls, and keep `apply_ready=false`. `--output-json` must resolve to a `.json` file under `output/phase3-system-prompt/`; path traversal outside that root is rejected.

```bash
python -m evolution.benchmarks.run_tblite \
    --baseline-prompt datasets/golden/benchmarks/phase3-system-prompt/baseline_system_prompt.json \
    --candidate-prompt datasets/golden/benchmarks/phase3-system-prompt/candidate_system_prompt.json \
    --fixtures-jsonl datasets/golden/benchmarks/phase3-system-prompt/tblite_cases.jsonl \
    --output-json output/phase3-system-prompt/dry-run/benchmarks/tblite.json \
    --dry-run

python -m evolution.benchmarks.run_yc_bench \
    --baseline-prompt datasets/golden/benchmarks/phase3-system-prompt/baseline_system_prompt.json \
    --candidate-prompt datasets/golden/benchmarks/phase3-system-prompt/candidate_system_prompt.json \
    --fixtures-jsonl datasets/golden/benchmarks/phase3-system-prompt/yc_bench_fast_test.jsonl \
    --preset fast_test \
    --output-json output/phase3-system-prompt/dry-run/benchmarks/yc_bench.json \
    --dry-run
```

Real benchmark execution remains deferred. The current adapter contract is a fixture-backed runnable smoke boundary that future Phase 3 execution must replace or supplement with real TBLite/YC-Bench results before optimizer acceptance, active apply, or default-gate promotion. The contract is hardened by tests that require `--output-json` to resolve to a fresh `.json` file under `output/phase3-system-prompt/`, reject non-`.json` output paths, reject output paths outside that root, reject resolved path traversal, reject pre-existing, symlinked, hardlinked, and input-overlapping output targets, and monkeypatch network/external-process APIs during in-process adapter main calls.

The execution draft keeps benchmark command templates and human approval gate explicit so future Phase 3 execution can fail closed: TBLite/YC-Bench adapters must be runnable and passing, rollback handles must be available, and separate human approval must be recorded before optimizer execution or active apply.

## Phase 3 Local Preflight Gate

The Phase 3 local preflight gate validates the candidate scaffold report plus dry-run TBLite/YC-Bench adapter reports and writes a local preflight report. It is still candidate-only: `phase3_execution_ready=false`, no real benchmarks are executed, and active prompt/source/runtime apply remains blocked.

```bash
python -m evolution.prompts.phase3_preflight_gate \
    --candidate-report output/phase3-system-prompt/<run-id>/review/candidate_only_report.json \
    --tblite-report output/phase3-system-prompt/<run-id>/benchmarks/tblite.json \
    --yc-bench-report output/phase3-system-prompt/<run-id>/benchmarks/yc_bench.json \
    --output-json output/phase3-system-prompt/<run-id>/preflight/phase3_preflight_report.json \
    --dry-run
```

The preflight gate checks candidate-only/apply-blocked flags, dry-run benchmark reports, hardened output constraints, and prompt artifact checksum consistency across reports. It can pass the local contract while still keeping real benchmarks and human approval blocking Phase 3 execution.

## Phase 3 Real Benchmark Readiness Manifest

The Phase 3 real benchmark readiness manifest is recorded in `reports/phase3_real_benchmark_readiness_manifest.json` and `reports/phase3_real_benchmark_readiness_manifest.md`. It is machine-readable preparation only: `real_benchmark_ready_now=false`, `active_apply_ready_now=false`, real TBLite/YC-Bench execution is not approved, and active system-prompt/source apply remains blocked.

The manifest fixes required inputs, environment requirements, cost/runtime caps, rollback requirements, and go/no-go conditions before any future switch from dry-run fixtures to real benchmark evidence.

## Full Plan

See [PLAN.md](PLAN.md) for the complete architecture, evaluation data strategy, constraints, benchmarks integration, and phased timeline.

## License

MIT — © 2026 Nous Research
