# Documentation Review Notes

A consistency + completeness pass over the docs in this directory.

## Verified accurate

- Module / package layout matches `find evolution -type f -name "*.py"`. Four subpackages now have content: `core/`, `skills/`, `tools/`, `validation/`. `prompts/`, `code/`, `monitor/` remain empty stubs.
- `EvolutionConfig` field defaults match `evolution/core/config.py`.
- `gate_decision.json` schema_version `"4"` matches both payload-writer sites in `evolution/skills/evolve_skill.py` and the test fixtures in `tests/skills/test_evolve_skill_validation_flow.py`.
- `ValidationReport` schema_version `"1"` matches `evolution/validation/validator.py` and the consumers in `evolution/core/closed_loop_feedback.py`.
- `_HEARTBEAT_TIERS` table matches `evolution/core/lm_timing_callback.py`.
- LM `request_timeout` / `num_retries` values per surface verified:
  - judge LM (`fitness.py`): `request_timeout=60, num_retries=5`
  - dataset gen LM (`dataset_builder.py`): `request_timeout=120, num_retries=5`
  - reflection LM (`evolve_skill.py`): `request_timeout=300, num_retries=2`
- Pinned dep ranges verified against `pyproject.toml`. Direct deps include `numpy>=1.24` and `pyyaml>=6.0`.
- 681 tests collected (`pytest tests/ -q` inside venv). 37 test files spanning `tests/{core,skills,tools,validation}/`.
- `generate_report.py` is a renderer that takes `--run output/<skill>/<ts>/ --prose reports/<phase>_prose.yaml --out reports/<phase>_validation_report.pdf`. Numbers come from the run dir's `gate_decision.json` + `metrics.json` + `run.log`; editorial prose + tables come from the YAML; the title-page logo is `assets/dna.png`.
- Closed-loop integration: `--closed-loop-during-evolution`, `--closed-loop-hermes-repo`, `--closed-loop-mode {feedback,trainset,both}`, `--closed-loop-in-valset`, `--closed-loop-saturation-threshold`, `--closed-loop-min-iters`, `--closed-loop-window-size` all wired in `evolve_tool.main()`. Symmetric flag on `evolve_skill.main()` raises `UsageError` until a `SkillFileInstaller` ships.
- `examples/hermes_tools_evolution_metadata.json` ships the confusable-neighbors sidecar users copy into `<hermes-agent>/tools/_evolution_metadata.json`.

## Minor inconsistencies in the codebase (worth tracking, not blockers)

### 4. Module-import-time `logging.basicConfig`
`evolution/skills/evolve_skill.py:30-34` calls `logging.basicConfig` at import. This is *idempotent* in stdlib (only first call wins) but means importing `evolve_skill` from another script silently configures the root logger. Documented in `interfaces.md` (Logging conventions) — flag if a future user wants to import `evolve()` from a notebook without the side effect.

### 5. `HermesSkillSource` env var name has changed
The `external_importers._load_skill_text` standalone CLI uses `~/.hermes/skills/`, but the `HermesSkillSource` adapter uses `~/.hermes/hermes-agent/skills/` (or `$SKILL_SOURCES_HERMES_REPO/skills/`). Different path under the same `~/.hermes/` prefix; could confuse a user who deletes one and expects both surfaces to break together.

### 6. CLI flag naming inconsistency
- `--bootstrap-resamples` (CLI) maps to `bootstrap_n_resamples` (Python) — note the `n_` prefix difference.
- All other CLI flags map straightforwardly.

### 7. Tier-3/4/5 packages are empty stubs
`evolution/{prompts,code,monitor}/` contain only `__init__.py`. They anchor the planned architecture but currently do nothing. Documented in `codebase_info.md` (implementation status table). Could confuse a new contributor expecting working code there.

### 8. Tool descriptions have a parallel-but-not-shared infrastructure
`evolution/tools/` mirrors `evolution/skills/` for the dataset, judge, proposer, and orchestration — the modules are intentionally duplicated rather than parameterized because the prompts differ enough that a shared base would be more abstract than helpful. `evolution/core/quality_gate.py` is the one piece that was extracted out and is now genuinely shared (preset table + gate-decision persistence).

### 9. Closed-loop signal is opt-in even when wired
`--closed-loop-during-evolution` is required to construct the `ClosedLoopFeedbackCache`; otherwise the metric's behavioral branch is dead code (no behavioral examples in trainset). Default `--closed-loop-mode feedback` keeps full backward compatibility with the pre-closed-loop CLI.

## Gaps that warrant future docs

### 1. No deployment / release docs
No `release.md`, `CONTRIBUTING.md`, `RELEASE.md`. Project is currently single-author with PR-based merges; if it scales, these would be needed.

### 2. No example `gate_decision.json` walkthrough
`data_models.md` shows the schema; a worked example narrating "the bootstrap CI lower bound was -0.06 so dual-check rejected" would help users reading their own decisions for the first time. Could be added if rejection diagnostics become a frequent user task.

### 3. No "how to add a new constraint" guide
`ConstraintValidator` is closed over a hardcoded set of checks. Adding a new one requires editing both the validator and (for the gate-payload integration) `evolve_skill.py`. Pattern is straightforward but undocumented; would be useful when Tier 2/3 lands and tool-description-specific constraints are added.

### 4. No GEPA-vs-MIPROv2 comparison
The fallback chain is implemented but the "when does GEPA underperform / when does MIPROv2" narrative isn't documented. The MIPROv2 path is a degraded mode (no knee-point, no `detailed_results`); user-facing implications are not surfaced beyond "knee_point.applied=false."

### 5. No "how to author a closed-loop task suite" guide
Users adopting `--closed-loop-during-evolution` need to write JSONL suites with calibrated `expected_tools` / `forbidden_tools` per task. The shape is documented in `data_models.md`; the *design heuristics* (how to choose tasks the agent's behavior is sensitive to) are not. The current shipped suites (`evolution/validation/suites/{patch,write_file,search_files}.jsonl`) are the de-facto examples.

## Recommended documentation maintenance

1. **Re-verify defaults on every release.** `EvolutionConfig` defaults are tuned often; doc table in `data_models.md` will drift.
2. **Re-collect test count when refactoring.** Currently ~680; bump if tests are added/removed.
3. **Update `gate_decision.json` schema docs on every schema bump.** When `schema_version` increments, both `data_models.md` and `interfaces.md` (test surfaces) need to mention the new fields.
4. **Update `ValidationReport` schema docs on every schema bump.** Currently `"1"`; fields are stable but new diagnostic fields will likely accumulate.
5. **Verify mermaid diagrams render.** GitHub renders mermaid in markdown; if a diagram breaks during edits, the rest of the page still renders, so silent breakage is possible. Spot-check on github.com after pushing.

## What's NOT documented (intentionally)

- **Per-PR rationale or change log.** That's `git log` + PR descriptions — not duplicated here.
- **Bug-fix recipes.** The fix is in the code; the commit message has the context.
- **Debugging output samples.** Run logs and `gate_decision.json` snapshots are user-specific and rot fast.
- **Style preferences.** Lives in `AGENTS.md`.
- **Experimental run results / findings docs.** Run outcomes belong in PR descriptions; the durable claims surface in `PLAN.md` deviations or in code-level docstrings where the constraint applies.
