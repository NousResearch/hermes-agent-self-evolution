# Components

Reference of the major modules in `evolution/`. Each entry: what it owns, the public surface, and the load-bearing implementation details that aren't obvious from the signatures.

## evolution/skills/evolve_skill.py — CLI + orchestrator

**Owns:** the end-to-end `evolve()` flow and the Click CLI (`main`).

**Public surface:**
- `main()` — Click command. CLI flags map 1:1 onto `evolve()` kwargs.
- `evolve(skill_name, ...)` — the orchestrator function. Importable and used directly by tests.
- Internal helpers tested directly: `_write_gate_decision`, `_dataset_payload`, `_knee_point_payload`, `_holdout_evaluate_with_metric`, `_resolve_budget`, `_default_gepa_runner`, `_default_mipro_runner`, `_print_fallback_banner`, `_build_optimizer_and_compile`.

**Phases inside `evolve()` (numbered headers in source):**
1. Find + load the skill via `find_skill(name, config.skill_sources)`.
2. Build or load eval dataset (synthetic / golden / sessiondb).
3. Validate baseline static constraints (warn-only — never blocks the run).
4. Configure DSPy LM + `LMTimingCallback`; build judge + GEPA-shaped metric.
5. Run GEPA (or MIPROv2 fallback) via `_build_optimizer_and_compile`.
5b. Knee-point Pareto selection across `optimized_module.detailed_results.candidates` (skipped on MIPROv2 fallback — no `detailed_results`).
6. Reassemble evolved frontmatter + body.
7. Static constraints on evolved artifact — short-circuit reject before holdout if any fail.
8. `dspy.Evaluate` baseline + evolved against holdout (≈ 2 × |holdout| judge calls).
9. `paired_bootstrap()` on per-example deltas → `validate_growth_with_quality()` → write `gate_decision.json`.
10. On deploy: write `evolved_skill.md`, `baseline_skill.md`, `metrics.json`.

**Quality-gate presets:** `_QUALITY_GATE_PRESETS` defines `strict`, `default`, `lenient`, `off`. CLI `--quality-gate` picks one; individual `--growth-free-threshold` / `--growth-quality-slope` / `--max-absolute-chars` flags override the preset.

**Output dir is created up-front** (right after the dry-run check) so the `FileHandler` captures dataset-gen LM calls + GEPA reflection + holdout eval all in one `run.log`.

## evolution/skills/skill_module.py — SKILL.md ↔ dspy.Module bridge

**Owns:** loading, parsing, wrapping, and reassembling SKILL.md files.

**Public surface:**
- `load_skill(path) -> dict` with keys `path`, `raw`, `frontmatter`, `body`, `name`, `description`.
- `find_skill(name, sources) -> Path | None` — walks the `SkillSource` list, first match wins.
- `SkillModule(skill_text)` — `dspy.Module` exposing `.skill_text` property and `forward(task_input)`.
- `reassemble_skill(frontmatter, evolved_body) -> str` — rejoin into a complete SKILL.md.

**Implementation note (load-bearing):** `SkillModule.__init__` constructs a `dspy.ChainOfThought(TaskWithSkill)` then immediately overwrites `self.predictor.predict.signature` via `with_instructions(skill_text)`. GEPA's `named_predictors()` walks `Predict.signature.instructions` to find mutable parameters, so the skill body has to live there. The `TaskWithSkill` docstring is intentionally a placeholder.

**Defensive strip:** `reassemble_skill` checks for a leading `---` block on the GEPA-mutated body and strips it (with a `WARNING` log) — the reflection LM occasionally mimics YAML frontmatter, which would otherwise produce a double-frontmatter file.

## evolution/skills/budget_aware_proposer.py — char-budget reflection prompt

**Owns:** the GEPA `instruction_proposer` that bakes a length budget into the reflection LM's prompt.

**Public surface:**
- `BudgetAwareProposer(baseline_chars, max_growth=0.2, safety_margin=0.10)` — call instance is `__call__(candidate, reflective_dataset, components_to_update) -> dict[str, str]`.

**Implementation notes:**
- Required because `gepa.optimize`'s `reflection_prompt_template` kwarg is unconditionally rejected when `DspyAdapter` is in use (`gepa/api.py:317-321`). DSPy's documented extension point is `instruction_proposer: ProposalFn` on `dspy.GEPA`.
- Prompt-engineering choices are deliberate (countdown framing, "at most N chars", loss-frame, one-shot tight example) — references are inline in the source.
- `safety_margin` (default 0.10) tightens the prompt's stated target relative to the validator's bar to compensate for observed ~8-9% LM overshoot. Default lands at +10pp prompt vs +20pp validator.
- Soft enforcement: if the LM overshoots, log `WARNING` but pass the proposal through — hard truncation would corrupt mid-sentence and could lose the very change that helped.

## evolution/skills/knee_point.py — Pareto-frontier knee-point selection

**Owns:** picking the most parsimonious candidate within ε of the best valset score.

**Public surface:**
- `select_knee_point(candidates, val_aggregate_scores, n_val, static_validator, gepa_default_idx, epsilon=None) -> CandidatePick`
- `CandidatePick` dataclass — frozen, carries the picked module + diagnostics needed to debug the choice (band size, ε, fallback reason, picked vs GEPA-default body chars, full band roster).

**Default ε:** `1 / n_val` — "one valset example's worth of disagreement." Override with caution; tightening it narrows the band and biases selection back toward the GEPA default.

**Iteration order:** ascending `body_chars`, tiebreak `-val_score`, then `idx`. First candidate whose `static_validator(text)` returns all-passed is picked. If every band candidate fails static, falls back to `gepa_default_idx` and records `fallback="static_failed_all"`.

**Why parsimony:** it's a legitimate regularizer (MDL / Occam) and is uncorrelated with the "lucky on N" noise that drives GEPA's overfit on small valsets.

## evolution/core/config.py — EvolutionConfig dataclass

**Owns:** the single source of truth for run parameters.

**Public surface:** `EvolutionConfig` dataclass (all fields documented inline, defaults included).

**Important:** `skill_sources` uses `field(default_factory=lambda: discover_skill_sources())`, so the discovery walk runs at config-construction time. Tests use a `_skill_source_env` autouse fixture (see `tests/core/test_constraints.py:9`) to point this at a fake repo.

## evolution/core/skill_sources.py — pluggable SKILL.md discovery

**Owns:** the layout-specific glue between agent frameworks and the optimizer.

**Public surface:**
- `SkillSource` Protocol with `name: str`, `find_skill(name) -> Path | None`, `list_skills() -> list[str]`.
- `HermesSkillSource(root)` — Hermes layout `<root>/skills/<category>/<name>/SKILL.md`. Direct dir-name match first, then frontmatter `name:` fuzzy match.
- `ClaudeCodeSkillSource(plugins_cache=~/.claude/plugins/cache)` — Claude Code layout `<vendor>/<plugin>/<version>/skills/<name>/SKILL.md`. Highest-version wins on collision.
- `LocalDirSkillSource(root)` — generic flat `<root>/<name>/SKILL.md`. Escape hatch for Codex / openclaw / custom.
- `discover_skill_sources(explicit_dirs=None) -> list[SkillSource]` — builds the priority-ordered default list.

## evolution/core/dataset_builder.py — eval dataset construction

**Owns:** generating + loading + saving eval datasets with train/val/holdout splits.

**Public surface:**
- `EvalExample` dataclass (`task_input`, `expected_behavior`, `difficulty`, `category`, `source`).
- `EvalDataset` dataclass (`train`, `val`, `holdout` lists). Supports `save(path)`, `load(path)`, `to_dspy_examples(split)`, `all_examples` property.
- `SyntheticDatasetBuilder(config).generate(artifact_text, artifact_type, num_cases)` — uses `dspy.ChainOfThought(GenerateTestCases)` against the judge model.
- `GoldenDatasetLoader.load(path, seed)` — loads `train/val/holdout.jsonl` if present, else single-file with auto-split.

**Load-bearing settings on the synthetic generator:**
- `max_tokens=16000` on the judge LM. Bumped from 4000 after `eval_dataset_size=60` truncated JSON output mid-string. Without this: JSONDecodeError → process exit.
- `request_timeout=120, num_retries=5` — dataset gen is a single bursty call; 5×120s = 10min worst case.

**Split logic** (synthetic): ratios from `EvolutionConfig.train_ratio` / `val_ratio` / `holdout_ratio` are normalized to actually sum to 1; holdout is no longer just "whatever's left." Default ratios `0.5/0.40/0.50` normalize to ≈ 0.36/0.29/0.36 of N (≈ 54/43/53 at the current `eval_dataset_size=150` default).

## evolution/core/external_importers.py — session-history mining

**Owns:** the `--eval-source sessiondb` path. Mines real usage from local AI tools.

**Public surface:**
- `ClaudeCodeImporter.extract_messages(limit)` — reads `~/.claude/history.jsonl` (user inputs only).
- `CopilotImporter.extract_messages(limit)` — reads `~/.copilot/session-state/*/events.jsonl` (user + assistant pairs).
- `HermesSessionImporter.extract_messages(limit)` — reads `~/.hermes/sessions/*.json` (OpenAI-format messages).
- `RelevanceFilter(model, seed).filter_and_score(messages, skill_name, skill_text, max_examples)` — two-stage filter: cheap heuristic, then LLM relevance scoring.
- `build_dataset_from_external(skill_name, skill_text, sources, output_path, model, max_examples, seed)` — orchestration entry point used by both the standalone CLI (`python -m evolution.core.external_importers`) and `evolve_skill.py`.

**Secret detection** (`SECRET_PATTERNS` regex) runs on every imported message. Anchored to known key formats (Anthropic, OpenRouter, OpenAI, GitHub, Slack, Notion, AWS, PEM private keys, common env-var names, common assignment patterns) to minimize false positives on prose. Matched messages are dropped silently.

**JSON parse hardening** (`_parse_scoring_json`): tries direct `json.loads`, falls back to balanced-brace extraction (not regex — `r'\{[^}]+\}'` breaks on nested braces).

## evolution/core/fitness.py — LLM-as-judge + GEPA metric

**Owns:** scoring agent outputs and producing GEPA-compatible metric callables.

**Public surface:**
- `FitnessScore` dataclass with `correctness`, `procedure_following`, `conciseness`, `length_penalty`, `feedback`, and a `composite` property (`0.5*c + 0.3*pf + 0.2*con - length_penalty`, clamped to [0,1]).
- `LLMJudge(config)` with `score(task_input, expected_behavior, agent_output, artifact_size=None, max_size=None) -> FitnessScore`. Uses `dspy.ChainOfThought(JudgeSignature)` against the eval model.
- `make_skill_fitness_metric(judge, baseline_skill_text, max_growth) -> callable` — closure returning the GEPA-shaped 5-arg metric.

**Why scores are typed `str`** in `JudgeSignature.OutputField`: scores arrive as text from the LLM and are clamped to `[0,1]` in `_clamp_to_unit()`. Declaring them as `str` keeps the typeguard quiet without per-field float-coercion ceremony.

**Empty-output handling:** if `prediction.output` is empty, the metric returns `score=0.0, feedback="Agent produced empty output"` and logs `WARNING`. Empty output is a real upstream failure signal (timeout, content filter, malformed prompt) that GEPA can't otherwise distinguish from a wrong answer.

**`_augment_feedback_with_pred_trace`** appends two blocks when `pred_trace` is set (predictor-level call site, not module-level):
- `[BUDGET]` line with current vs baseline chars + growth %, so the reflection LM sees when it's bloated.
- `[REASONING]` block quoting the predictor's chain-of-thought (truncated at 500 chars).

Score is **never** modified by `pred_trace` enrichment — GEPA enforces score equality across both call sites (warns and overrides if they diverge).

**`_augment_feedback_with_closed_loop`** appends a `[CLOSED_LOOP]` (or `[CLOSED_LOOP-NOISY]` when `|Δpass_rate| < 0.15`) block to the feedback string when a `ClosedLoopFeedbackCache` is wired in and the saturation gate fires (pred_trace-gated, same as the budget hint). The block carries the closed-loop decision, win/loss/tie counts, and per-task diffs for tasks whose verdict changed between baseline and evolved. Score is still untouched on this path — only feedback. Used when the metric is configured with `closed_loop_cache=` (see `closed_loop_feedback.py` and `--closed-loop-during-evolution` on `evolve_tool`).

**`_score_behavioral_example`** is the behavioral-example branch. The metric's top-of-closure check `if hasattr(prediction, "_closed_loop_task_id"):` routes a behavioral `dspy.Example` (one per closed-loop task, added to the trainset when `--closed-loop-mode trainset|both`) to this helper. It reads `_candidate_text` + `_closed_loop_task_id` off the Prediction (stuffed there by `ToolModule.forward` on the behavioral branch), asks the cache for the per-task verdict, and returns `Prediction(score=float(verdict.passed), feedback="[BEHAVIORAL] task <id>: pass|fail|abstain …")`. Judge is never called on behavioral examples. Score is deterministic over candidate text (cache key) so GEPA's predictor-vs-module byte-identity contract holds automatically.

**LM hardening:** `request_timeout=60, num_retries=5` on the judge LM. 60s = 6× P99 of slowest observed `gpt-4.1-mini` call (9.8s). 5×60s = 5min worst case before raising.

## evolution/core/behavioral_example.py — closed-loop task → dspy.Example builder

**Owns:** the function that turns a closed-loop `TaskSuite` into a list of `dspy.Example`s suitable for injection into GEPA's trainset (or valset).

**Public surface:**
- `build_behavioral_examples(suite: TaskSuite) -> list[dspy.Example]` — one example per task in the suite, stable-ordered by `task_id`. Each example carries a `closed_loop_task_id` marker the fitness metric routes on, plus the suite's original `user_message` as a placeholder `task` field (not consumed by the behavioral branch; kept for debuggability). Both fields are marked as input keys via `.with_inputs("task", "closed_loop_task_id")` so DSPy passes them to `ToolModule.forward()` via `program(**example.inputs())`.

**Why this exists:** behavioral examples are how closed-loop signal enters GEPA's `sum(minibatch_scores)` acceptance arithmetic. The metric's behavioral branch scores `float(verdict.passed)` and the result lands in the same minibatch sum the judge contributes to — so a candidate that wins a behavioral task its predecessor failed adds +1.0 to the sum and breaks judge ties on saturated baselines.

## evolution/core/closed_loop_feedback.py — closed-loop verdict cache + deterministic feedback rendering

**Owns:** the integration seam between `ClosedLoopValidator` and the GEPA loop — runs validator, caches results by candidate text, and renders verdicts as deterministic text blocks for the reflection LM's feedback channel.

**Public surface:**
- `ClosedLoopFeedbackCache(validator, suite, tool_name, baseline_description, saturation_threshold=0.95, min_iters=3, window_size=8, gate_mode="sampled")` — single instance per `evolve_tool`/`evolve_skill` run. Threading-locked so DSPy's parallel `Evaluate` workers don't race into `ConcurrentRunError` from the validator's cross-process flock. Tmp dir for baseline/evolved manifest JSON lives for the cache's lifetime; OS reclaims at process exit.
- `.record_judge_score(score)` — call from the metric on every invocation; feeds the saturation gate's recent-window history.
- `.should_run() -> bool` — the gate. `gate_mode="sampled"` (default, opportunistic feedback-only use): fire when `min(recent_window) >= saturation_threshold` OR `iters_since_last_run >= min_iters`. `gate_mode="always"` (selection-affecting trainset use): always open — every novel candidate must score every time.
- `.get_or_run(candidate_text) -> Optional[ValidationReport]` — cache key is `sha256(candidate + suite.sha256)`. Cache hit returns cached report; miss writes the candidate's description into a tmp JSON manifest and calls `validator.validate()`. Validator failures (`ConcurrentRunError`, `StaleBackupError`, `ChecksumDriftError`) log `WARNING` and return `None` — closed-loop failure must never take the GEPA run down.
- `.get_task_verdict(candidate_text, task_id) -> Optional[TaskResult]` — calls `get_or_run` and indexes `report.evolved.tasks` by `task_id`. Returns `None` if the gate is closed or the validator raised a swallowed error or the task isn't present.
- `.force_run(candidate_text) -> ValidationReport` — same shape as `get_or_run` but bypasses `should_run()` and propagates validator exceptions (instead of logging + returning `None`). Used by the saturation pre-flight (`evolution/core/saturation_check.py`) to fire the validator on the baseline once before any judge scores have been recorded — in default `gate_mode="sampled"`, `should_run()` returns `False` until either a judge score saturates or the periodic floor elapses, so `get_or_run` would silently no-op at preflight time. Preserves the "next `get_or_run` is allowed to fire immediately" guarantee by resetting `_iters_since_last_run` to `min_iters` (the same value `__init__` uses), so the saturation gate's first-fire allowance for downstream callers is intact.
- `render_feedback_block(report: ValidationReport) -> str` — module-level function. Renders the cached report as a deterministic `[CLOSED_LOOP]` block (or `[CLOSED_LOOP-NOISY]` when `|Δpass_rate| < 0.15`) with decision, decision_reasons, win/loss/tie counts, and per-task diffs for tasks whose verdict changed. Determinism is required because GEPA hashes reflective-dataset entries for caching.

**Two use modes**, both wired through `evolve_tool` CLI flags:

1. **Feedback enricher** (`--closed-loop-mode feedback`, default): the metric's `_augment_feedback_with_closed_loop` helper calls `get_or_run` on the candidate currently under reflection, then appends the rendered block to the metric's `dspy.Prediction.feedback`. Saturation-gated so it only fires when the judge has converged. Score is unchanged.
2. **Trainset score channel** (`--closed-loop-mode trainset`): `build_behavioral_examples(suite)` injects per-task `dspy.Example`s into the trainset. The metric's behavioral branch calls `get_task_verdict` on each behavioral example and returns the binary verdict as score. Behavioral wins contribute to `sum(minibatch_scores)`, breaking judge ties at acceptance.

## evolution/core/saturation_check.py — pre-flight that detects doomed runs

**Owns:** the pre-GEPA probe that scores the baseline on the holdout (and the closed-loop suite, if configured), classifies the result into one of four bands, and lets the call site decide whether to prompt for confirmation or default-deny. Independent of any GEPA-side change; mirrors the shape of `evolution/core/auth_check.py` (pure helper returns a structured report; rendering + exit handled by the call site).

**Public surface:**

- `SaturationBand: Literal["healthy", "no_headroom", "weak_signal", "uniform_failure"]` — the four-band classification.
- `DEFAULT_THRESHOLDS: dict[str, float]` — `no_headroom_synthetic=0.99`, `weak_signal_synthetic=0.95`, `no_headroom_closed_loop=0.95`, `uniform_failure_closed_loop=0.15`.
- `SaturationReport` dataclass — the contract between the helper and the call site. Carries the band, holdout score + per-example list (reused downstream for cache reuse), the closed-loop score + per-example list when present, the band-specific suggestion strings, and the thresholds that produced the band.
- `saturation_preflight(baseline_module, holdout_examples, metric, lm, closed_loop_cache=None, baseline_artifact_text=None, thresholds=None) -> SaturationReport` — pure function. Scores baseline via `_score_baseline_on_holdout` (a thin wrapper around `dspy.Evaluate` carved out so tests can patch the DSPy boundary), then fires `closed_loop_cache.force_run(baseline_artifact_text)` when the cache is provided. Raises `ValueError` on empty `holdout_examples` before any LM call.
- `render_saturation_panel(report, console=None) -> None` — emits a one-line dim acknowledgement for the `healthy` band, or a Rich `Panel` (yellow border) with band, score lines, and bulleted suggestions for the warn bands.
- `interactive_confirm(prompt="Continue anyway? [y/N] ") -> bool` — reads stdin; returns `True` only for `{y, yes}` case-insensitive. Catches `KeyboardInterrupt` and `EOFError`, returning `False` (treats as "n", no traceback noise).
- `is_non_interactive() -> bool` — `not sys.stdin.isatty()`. Call sites use it to decide between prompting and printing the override hint.

**Band classifier logic** (`_classify_band`, in priority order):

1. **`uniform_failure`** if `closed_loop_score is not None AND closed_loop_score <= 0.15` — validator agent too weak to use the artifact at all; signal isn't discriminating.
2. **`no_headroom`** if either:
   - `holdout_score >= 0.99 AND closed_loop_score is None` — only signal available is the judge, and it's pegged, OR
   - `closed_loop_score >= 0.95 AND holdout_score >= 0.95` — both signals effectively saturated. The `holdout_score >= 0.95` gate on this clause keeps `(synthetic=0.5, CL=1.0)` classified as `healthy` (there's real judge headroom even with behavioral pegged; usually means misconfigured eval rather than true saturation).
3. **`weak_signal`** if `holdout_score >= 0.95 AND 0.15 < closed_loop_score < 0.95` — judge saturating but closed-loop discriminates; GEPA's small-minibatch acceptance will struggle (per the deviation #8 finding); expect many proposals rejected.
4. **`healthy`** otherwise — no panel, just a one-line dim log.

**Call-site integration:** both `evolve_skill.py` and `evolve_tool.py` invoke the helper after the dataset is built and `baseline_module`/`metric`/`closed_loop_cache` are constructed but before GEPA setup. The `holdout_per_example` list from the report is stashed and reused at the post-GEPA `_holdout_evaluate_with_metric` site — so the baseline isn't re-scored at run end. Net cost: ~zero (the probe is the holdout eval shifted earlier). See `--no-saturation-check` / `--force-saturation-check` in `interfaces.md`.

## evolution/core/constraints.py — deploy gate

**Owns:** all constraint checks and the deploy gate's two-stage decision.

**Public surface:**
- `ConstraintResult` dataclass (`passed`, `constraint_name`, `message`, `details`).
- `ConstraintValidator(config)`:
  - `validate_static(artifact_text, artifact_type) -> list[ConstraintResult]` — size, non_empty, structure (skill only).
  - `validate_growth_with_quality(artifact_text, baseline_text, bootstrap_result) -> list[ConstraintResult]` — quality-gated growth + absolute char ceiling.

**Two-stage gate logic** (in `_check_growth_with_quality_gate`):
- `required_improvement = max(0, slope * (growth - free))`
- If `required == 0` (growth ≤ free threshold): **no_regression_only** branch — pass on `mean ≥ 0`.
- Else: **dual_check** branch — pass requires `mean ≥ required` AND `lower_bound > 0`.
- Negative growth (shorter artifact) always falls into the no_regression branch.
- Zero baseline length is treated as zero growth (avoids divide-by-zero).

**Absolute char ceiling** (`_check_absolute_chars`) is independent of growth. Backstops short baselines that legitimately need expansion — a 200-char baseline growing to 1500 is +650% but only 1500 chars absolute.

## evolution/core/stats.py — paired bootstrap

**Owns:** the bootstrap CI helper used by the deploy gate.

**Public surface:** `paired_bootstrap(baseline_scores, evolved_scores, *, confidence=0.90, n_resamples=2000, seed=42) -> dict` returning `mean`, `lower_bound`, `upper_bound`, `n_examples`, `n_resamples`, `confidence`.

**Method:** basic (reverse-percentile) bootstrap on the per-example improvement vector. Literature-recommended for small N (≤20). BCa is the upgrade path once N≥20 routinely.

**Inputs must be paired:** equal-length arrays where index `i` in both refers to the same example. Raises `ValueError` on length mismatch or empty input.

## evolution/core/lm_timing_callback.py — LM observability

**Owns:** per-LM-call timing, heartbeat warnings, and per-attempt litellm failure logging.

**Public surface:**
- `LMTimingCallback(timer_factory=threading.Timer)` — DSPy `BaseCallback` subclass. Register globally via `dspy.configure(callbacks=[LMTimingCallback()])`.
- `register_litellm_failure_callback()` — installs `_log_litellm_failure` into `litellm.failure_callback`. Idempotent + lock-guarded against TOCTOU on concurrent first-import.

**Heartbeat tiers** (`_HEARTBEAT_TIERS`):
- 60s → DEBUG (cold-cache calls cross this legitimately; WARNING here would train the user to ignore heartbeats)
- 180s → WARNING
- 300s → WARNING
- 600s → WARNING

**Why two surfaces:** `BaseCallback.on_lm_end` only fires once per logical call — it hides retries. Without `litellm.failure_callback`, a 5×60s retry loop on a flaky API looks like a single 5-minute LM call. The pair (callback + failure hook) gives both visibility into call duration and visibility into individual retry attempts.

**`timer_factory` is injectable** so tests use a `FakeTimer.advance(seconds)` double instead of monkeypatching intervals + `time.sleep(0.2)` — deterministic, sub-millisecond, not flaky on slow CI.

## evolution/core/quality_gate.py — artifact-agnostic deploy-gate helpers

**Owns:** the preset table, proposer-mode resolution, and gate-decision persistence shared across pipelines.

**Public surface:**
- `QUALITY_GATE_PRESETS` — `strict` / `default` / `lenient` / `off` / `non-inferiority`. Each preset bundles `growth_free_threshold`, `growth_quality_slope`, `max_absolute_chars`, and optionally `gate_mode` + `inferiority_tolerance`.
- `resolve_proposer_mode(fitness_profile) -> ProposerMode` — `growth` → `growth`, `balanced` → `balanced`, anything else (including unknown values) → `compression`.
- `write_gate_decision(output_dir, decision) -> Path` — writes `gate_decision.json` under `output_dir` and returns the path.

**Implementation note:** extracted from `evolve_skill.py` to be reusable by the tool pipeline. `evolve_skill.py` keeps back-compat aliases under the original underscored names (`_QUALITY_GATE_PRESETS`, `_resolve_proposer_mode`, `_write_gate_decision`) so existing test surfaces keep working.

## evolution/tools/tool_source.py — tool-manifest data model + discovery

**Owns:** the tool-pipeline analog of `skill_sources.py` — loading manifests, validating tool names, and modeling the manifest as an immutable dataclass.

**Public surface:**
- `ToolSource` Protocol with `name: str`, `supports(path) -> bool`, `find_manifest(path_or_name) -> ToolManifest | None`, and `apply_evolved(source_path, evolved_manifest, target_tool, new_description) -> None`. Read and write both live on the adapter so the orchestrator can dispatch by `supports()` without knowing which backing store it's hitting.
- `MCPManifestSource(root)` — reads a static JSON file in MCP `list_tools()` shape; paths are resolved against `root` if relative. `supports()` returns True for existing `.json` files. `apply_evolved()` rewrites the source manifest atomically (tempfile + `os.replace`), preserving every non-target tool's `description`, `inputSchema`, and any `_evolution_metadata` block.
- `ToolManifest` (frozen dataclass): `tools: tuple[ToolEntry, ...]`, `confusable_neighbors: dict[str, str]`, `dropped_tools: tuple[tuple[str, str], ...]`. Helpers `from_json_file(path)`, `find_tool(name)`, `confusable_neighbor_for(name)`, `replace_description(name, new_description) -> ToolManifest`.
- `ToolEntry` (frozen dataclass): `name`, `description`, `input_schema`, `source_kind`, `source_location`. The last two are optional adapter-state fields populated by source-walking adapters (HermesToolSource); they're `None` for JSON-backed manifests.
- `SentinelParseError(ValueError)` — raised by sentinel parsing in `tool_module.py` / `tool_proposer.py`; defined here so callers can import it without pulling DSPy.
- `discover_tool_sources(explicit_dirs=None) -> list[ToolSource]` — returns both `MCPManifestSource(d)` and `HermesToolSource(d)` per directory, with `MCPManifestSource` first (its `supports()` is the cheaper check).

**Implementation note (load-bearing):** tool names are validated against `^[a-zA-Z0-9_-]{1,128}$` at load. Names outside this set break sentinel parsing (regex metacharacters, embedded `-->`, etc.) and are rejected with a clear error. Normalization collisions (`read-file` vs `read_file`, which both lowercase + underscore-normalize to `read_file`) are also rejected at load — sentinel matching uses the original casing but lookup robustness relies on normalization being injective.

## evolution/tools/hermes_source.py — Python-source tool adapter

**Owns:** the read-and-write adapter for tools defined as Python `*_SCHEMA` / `*_SCHEMAS` dicts (Hermes Agent's pattern). Pure AST — no module execution — so it tolerates schemas whose sibling fields use Name refs, function calls, or lists.

**Public surface:**
- `HermesToolSource(root)` — `name = "hermes_source"`. Implements the full `ToolSource` Protocol.
- `supports(path) -> bool` — True iff `path` is a directory containing at least one top-level `Assign` whose target id matches `^_?[A-Z][A-Z0-9_]*_(SCHEMA|SCHEMAS)$`.
- `find_manifest(path) -> ToolManifest | None` — walks `path` recursively, collects every parseable schema, and returns a `ToolManifest` whose tools are alphabetically sorted. Skipped schemas appear in `manifest.dropped_tools` as `(name_hint, reason)` pairs.
- `apply_evolved(source_path, evolved_manifest, target_tool, new_description)` — splices the new description into the source file's bytes at the AST-derived span and atomically replaces the file. For `name_ref` tools, modifies the resolved constant's assignment rather than the schema dict's `"description"` key. Refuses to rewrite `joined_str` (f-string) descriptions.

**Tool shapes handled:**
1. `*_SCHEMA = {"name": "tool_a", "description": "...", ...}` — single tool dict.
2. `*_SCHEMAS = [{...}, {...}]` — list of tool dicts at module top level.
3. Schemas with non-literal sibling fields (e.g., `"input_schema": some_func()`) — captured as long as the `name` and `description` fields are statically reachable.
4. Name-ref descriptions — `"description": TOOL_DESCRIPTION` where `TOOL_DESCRIPTION = "..."` is a top-level string constant; `source_kind = "name_ref"` and `source_location` points at the constant.
5. Multi-line parenthesized concatenation — `("first " "second " "third")` collapsed by the parser into a single Constant; `source_location` spans the parenthesized block.
6. f-string descriptions — surfaced as `source_kind = "joined_str"` for read but rejected on write (the caller must convert to a literal first).

**`source_kind` annotation:** every emitted `ToolEntry` carries one of `"literal"`, `"name_ref"`, or `"joined_str"` so the write path can dispatch correctly. `source_location` is `(file_path, lineno, col_offset, end_lineno, end_col_offset)` of the description string node.

**Byte-precise write path:** `apply_evolved` does NOT serialize the dict back through `ast.unparse` (that would canonicalize formatting and lose comments). It computes byte offsets from `(lineno, col_offset)` into the source text and splices `repr(new_description)` (single-line) or a triple-quoted string with re-indented continuation lines (multi-line) into place. Bytes outside the target span are preserved verbatim.

**Sidecar metadata:** `<root>/_evolution_metadata.json` — if present, its `confusable_neighbors` mapping is loaded into the manifest. Missing or malformed sidecars are ignored (with a warning).

## evolution/tools/tool_module.py — DSPy module + sentinel rendering

**Owns:** the manifest-rendered prompt that lives in the predictor's signature instructions, plus the sentinel parser that recovers the target description from a mutated candidate.

**Public surface:**
- `ToolModule(target_tool_name, manifest, target_description)` — `dspy.Module` exposing one `selector` predictor (`dspy.ChainOfThought(ToolSelectionSignature)`). `.description_text` property reads the current target description from the predictor's instructions via sentinel parsing.
- `ToolSelectionSignature` — DSPy signature: `task` input, `reasoning` + `chosen_tool` outputs.
- `_render_manifest_for_prompt(manifest, target_name, target_description)` — alphabetical-by-name rendering with sentinel markers (`<!-- TARGET:<name> -->` … `<!-- /TARGET:<name> -->`) around the target description. All non-target slots are byte-identical regardless of which target description is plugged in.
- `_extract_description_from_sentinels(instructions, target_name)` — inverse parser. Raises `SentinelParseError` on missing, duplicated, or reversed markers.

**Implementation note (load-bearing):** like `SkillModule`, `ToolModule.__init__` overwrites `self.selector.predict.signature` via `with_instructions(rendered_manifest)` so GEPA's `named_predictors()` walk finds the manifest in `Predict.signature.instructions`. The signature's class docstring is intentionally a placeholder.

**Behavioral-example branch:** `forward()` accepts an optional `closed_loop_task_id: str` kwarg. When set, the module skips `self.selector(task=task)` entirely (no LM call) and returns a `Prediction` carrying `_closed_loop_task_id` and `_candidate_text=self.description_text`. The fitness metric's behavioral branch reads these fields from `pred` regardless of `pred_trace`, so behavioral-example scores are consistent across GEPA's Pareto-eval and reflective-feedback paths. This is the mechanism by which closed-loop verdicts enter the score channel without a custom DspyAdapter — the data is threaded through the `Prediction` object itself.

## evolution/tools/tool_proposer.py — sentinel-constrained reflection prompt

**Owns:** the GEPA `instruction_proposer` that mutates only the sentinel-delimited region of the rendered manifest.

**Public surface:**
- `BudgetAwareToolProposer(target_tool_name, manifest, target_description, baseline_chars, max_growth=0.2, safety_margin=0.10)` — subclasses `BudgetAwareProposer` for budget infrastructure but installs a tool-specific reflection template (sentinel-preservation hard constraint, sentinel-preserving BEFORE/AFTER one-shot, budget framed against the description not the full manifest).
- `extract_and_rebuild(candidate, manifest, target_name) -> str` — pure function: parse the sentinel-delimited region, re-render the full manifest with that description plugged in. Testable without LM mocks.

**Implementation note (load-bearing):** on `SentinelParseError`, `__call__` logs `WARNING`, increments `self.sentinel_failures`, and **re-raises**. GEPA's `reflective_mutation.py` catches the exception and skips the iteration. Returning a baseline-unchanged candidate would instead create a phantom duplicate-score entry that pollutes the knee-point Pareto pool.

## evolution/tools/tool_judge.py — tool-flavored LLM judge + fitness metric

**Owns:** the 3-dim judge specialized for tool-selection outputs, and the GEPA-shaped 5-arg metric closure.

**Public surface:**
- `ToolJudgeSignature` — DSPy signature with inputs `task`, `expected_tool`, `chosen_tool`, `reasoning`; outputs `correctness`, `procedure_following`, `conciseness`, `feedback` (all typed `str`, same rationale as `JudgeSignature`).
- `ToolJudge(config)` — LLM-as-judge wrapper around `ToolJudgeSignature`. `score(task, expected_tool, chosen_tool, reasoning) -> FitnessScore`. Mirrors `LLMJudge`'s contract but takes the four tool-selection input fields and always returns `length_penalty=0.0` (length pressure lives in the proposer's slope, not the judge).
- `make_tool_fitness_metric(judge, baseline_description, manifest, target_tool_name, max_growth, text_extractor=None) -> callable` — returns the GEPA-shaped 5-arg metric.

**Implementation note (load-bearing):** the metric parses `pred.chosen_tool` with generous normalization before reaching the judge — lowercased, stripped of quotes/backticks/whitespace, hyphens replaced with underscores. Short-circuits to `dspy.Prediction(score=0.0, feedback=...)` for unparseable outputs (blank or contains internal whitespace) and for outputs that parse to a name not in the manifest. The judge is only called on a parseable, in-manifest choice.

`text_extractor` is forwarded into `_augment_feedback_with_pred_trace` so the `[BUDGET]` reflection line measures the description region between sentinels rather than the full rendered manifest — without it, the budget framing is wrong by an order of magnitude on multi-tool manifests.

**Behavioral-example branch:** the metric's top-of-closure check is `if hasattr(pred, "_closed_loop_task_id"):` — a behavioral example produced by `build_behavioral_examples(suite)` gets routed to `_score_behavioral_example(pred, closed_loop_cache)`, which returns `Prediction(score=float(verdict.passed), feedback="[BEHAVIORAL] task <id>: pass|fail|abstain …")`. The judge is never called on behavioral examples. The `closed_loop_cache` is an optional `make_tool_fitness_metric` kwarg; when set, every metric invocation also calls `cache.record_judge_score(score.composite)` on the non-behavioral path so the saturation gate sees fresh history.

## evolution/tools/evolve_tool.py — CLI + orchestrator

**Owns:** the end-to-end `evolve()` flow and the Click CLI (`main`) for tool description evolution.

**Public surface:**
- `main()` — Click command. CLI flags map onto `evolve()` kwargs.
- `evolve(tool_name, manifest_path, ...) -> dict` — orchestrator function. Importable and used directly by tests.

**Phases inside `evolve()`** mirror `evolve_skill.evolve()`'s 10-step structure with three substituted steps:
1. Load manifest from JSON + pick target tool (replaces "find + load SKILL.md").
2. Build eval dataset via `SyntheticDatasetBuilder.generate_tool_selection` (three buckets: `target_correct`, `confusable_neighbor`, `regression_detection`).
3. Validate baseline static constraints (warn-only).
4. Configure DSPy LM + `LMTimingCallback`; build `ToolJudge` + tool fitness metric.
5. Run GEPA with `BudgetAwareToolProposer` as `instruction_proposer`.
5b. Knee-point Pareto selection — `text_extractor` measures description length, not full rendered-manifest length, so parsimony tracks the artifact the user actually evolves.
6. Reassemble evolved manifest (`ToolManifest.replace_description`).
7. Static constraints on the evolved description — short-circuit reject before holdout if any fail.
8. `dspy.Evaluate` baseline + evolved against holdout.
9. `paired_bootstrap` → `validate_growth_with_quality` → `write_gate_decision`. Growth + ceiling check is on the description, not the rendered manifest.
10. On deploy: write `evolved_manifest.json`, `baseline_manifest.json`, `metrics.json`. Optional `--apply` rewrites the source manifest in place (refuses to write into a Claude Code plugin cache); `--patch` emits a unified diff of the manifest to stdout.

**`gate_decision.json` additions:** every tool-path decision carries `artifact_type: "tool_description"`, `target_tool: <name>`, `manifest_neighbor_count: len(manifest.tools) - 1`, and `sentinel_failures: <count>` (number of reflection-LM outputs the proposer rejected for failing sentinel preservation).

**Closed-loop integration during evolution:** five flags wire `ClosedLoopFeedbackCache` into the GEPA loop without changing the metric signature or the proposer.

- `--closed-loop-during-evolution <suite.jsonl>` (+ required `--closed-loop-hermes-repo`): construct a `ClosedLoopFeedbackCache` against the validator stack and pass it into `make_tool_fitness_metric(closed_loop_cache=...)`.
- `--closed-loop-mode {feedback,trainset,both}` (default `feedback`):
  - `feedback`: the cache lives only on the metric's feedback path — `_augment_feedback_with_closed_loop` appends a `[CLOSED_LOOP]` block when the saturation gate fires. Score channel unchanged.
  - `trainset`: `_load_behavioral_examples_from_suite(suite_path)` injects per-task `dspy.Example`s into GEPA's trainset; cache is built with `gate_mode="always"` because behavioral scoring needs to run every time the example is sampled. Behavioral scores contribute to `sum(minibatch_scores)` acceptance.
  - `both`: trainset behavioral examples AND the `[CLOSED_LOOP]` feedback block on non-behavioral examples (most expensive).
- `--closed-loop-in-valset / --no-closed-loop-in-valset` (default off): when `mode != feedback`, also include behavioral examples in the valset — adds them to the Pareto frontier + holdout scoring at the cost of an extra full-eval pass over the suite per accepted candidate.
- `--closed-loop-saturation-threshold / --closed-loop-min-iters / --closed-loop-window-size` — only consumed in `feedback` mode; tune the saturation gate.

`main()` rejects `--closed-loop-during-evolution` without `--closed-loop-hermes-repo` as a `UsageError`, and `--closed-loop-mode != feedback` without `--closed-loop-during-evolution`. Local imports in `_maybe_build_closed_loop_cache` keep the validation stack out of cold-path runs.

**Cost ceiling + benchmark hook (shared with `evolve_skill`):** `--max-total-cost-usd` participates in the same `CostLedger` kill switch (see `lm_timing_callback.py`); `--benchmark-cmd` is a post-gate shell hook whose env vars include `EVOLVED_PATH` / `BASELINE_PATH` pointing at the rendered manifest JSONs and `ARTIFACT_TYPE="tool_description"`. Both write structured blocks into `gate_decision.json` — see `data_models.md`.

## evolution/prompts/evolve_prompt_section.py — CLI + orchestrator

**Owns:** the end-to-end `evolve_prompt_section()` flow and the Click CLI (`main`) for evolving a named system-prompt section. Two backends, dispatched on `--target`: `hermes` evolves a top-level string constant in Hermes `agent/prompt_builder.py` (e.g. `MEMORY_GUIDANCE`); `claude` evolves a sentinel-delimited region of a `--claude-md` CLAUDE.md. The phase-3 analogue of `evolve_tool`, but with a fundamentally different eval substrate: there is no cheap synthetic classification GEPA can score, so **every** candidate is run through a real agent subprocess (`hermes -z` or `claude -p`). The deploy gate is therefore a `ClosedLoopValidator` win/loss decision, not a paired-bootstrap CI. Backend swap is purely at the adapter layer — `--target claude` substitutes `ClaudeCodeAgentRunner` / `ClaudeCodePromptSource` / `ClaudeAppendPromptInstaller` (see the Claude Code backend section below) and the convention scorer; GEPA, the validator, and `score_task` are shared.

**Public surface:**
- `main()` — Click command. CLI flags map onto `evolve_prompt_section()` kwargs.
- `evolve_prompt_section(section_name, hermes_repo, tasks_path, ...) -> dict` — orchestrator function. Importable and used directly by tests.

**Integration model — in-place splice + atomic restore.** Unlike skills (separate writable workdir) there is no env-var hook or plugin seam: the section is a constant inside Hermes' own source, so the framework edits that file in place and restores it. The whole evolution runs inside `_prompt_builder_guard(target_path)` — a context manager that takes an atomic `.cl_backup` (`_BACKUP_SUFFIX`), grabs an exclusive `fcntl.flock` on `.cl_validation.lock` (`_LOCK_FILENAME`) in the target's parent dir, and byte-restores the original on exit (refusing to start on a stale backup or a held lock). These are the *same* lock + backup names `ClosedLoopValidator` uses, so the guard is sequenced *before* the deploy-gate validator, never nested. The deploy gate then re-acquires the lock itself.

**Phases inside `evolve_prompt_section()`:**
1. Resolve baseline: `HermesPromptSource.read(section_name)` validates the section is a top-level string constant, then reads its text — or `--baseline-override-file` supplies starting text (a deliberately-weakened baseline for headroom, or a regression ablation) while the *live* file is still backed up/restored and `--apply` still writes the live section.
2. Train/holdout split of the JSONL suite (`_split_train_holdout`, deterministic shuffle+seed, ≥1 task each side; suites with <2 tasks are rejected).
3. Build the eval stack: `SaveCallJudge` + a per-task Layer-2 factory (`_make_layer2_factory`, binds each task's `expected_save_content` rubric + message into a `score_task`-shaped scorer; returns `None` for tasks with no rubric) → `HermesPromptSectionInstaller` + `HermesAgentRunner` + a `make_memoizing_splice_scorer` over `install_candidate` / `score_task_id`, serialized under a `threading.Lock`.
4. `dspy.configure(lm=eval_lm)` sets the **global** default LM (not just `dspy.context`) so the passthrough predictor resolves an LM inside GEPA's worker threads — without it, `forward()`'s passthrough call raises "No LM is loaded" in those threads, yielding no trajectories and no proposal.
5. Inside `_prompt_builder_guard`: saturation pre-flight (baseline behavior on the holdout; aborts/denies on a non-`healthy` band unless `--force-saturation-check`, with non-interactive contexts refusing rather than prompting) followed by GEPA(`PromptModule`, `PromptSectionProposer`, `make_prompt_fitness_metric` + the memoizing splice scorer). Trainset/valset are `_behavioral_examples` (task message + `closed_loop_task_id`).
6. Select the evolved section via GEPA val-argmax (`detailed_results.best_idx`), reading the body back out of the winning candidate's sentinel region (`_section_text_from_candidate`).
7. Deploy gate: `ClosedLoopValidator.validate(...)` runs baseline vs evolved on the holdout suite (the same per-task Layer-2 factory + threshold threaded in). `report.decision == "pass"` is the deploy verdict.
8. Write `gate_decision.json`; on a passing gate `--apply` writes the evolved section back into `prompt_builder.py`. `baseline_section.txt` / `evolved_section.txt` are also emitted.

`_run_one_task_score` is the GEPA in-loop scorer: materialize the task fixture into a tmp dir, run the agent against whatever section is currently spliced, `score_task`, return 1.0/0.0 (in-loop abstentions score 0.0 — the deploy gate handles abstentions properly). Budget rides the shared `COST_LEDGER` + `CostCeilingExceeded` kill switch; the ceiling abort writes a `cost_ceiling_exceeded` gate decision.

**`gate_decision.json` additions:** `artifact_type: "prompt_section"`, `target_section: <name>`, `baseline_chars` / `evolved_chars` / `growth_pct`, a `closed_loop` block (the validator decision + pass rates + W/L/T), and `sentinel_failures` (proposer candidates rejected for losing the sentinels). `decision_signal` is always `"closed_loop"`. `--create-pr` is **deferred** for prompt sections (it would pollute the diff with the local override-hook commit) and is recorded as `skipped`; use `--apply` + a manual PR.

### Supporting modules (`evolution/prompts/`)

- `prompt_source.py` — `PromptSource` Protocol (`read` + `write` only, `runtime_checkable`) + `SectionDescriptor` (frozen metadata). The Protocol is deliberately minimal — the driver only reads a baseline and writes/splices an evolved value. `list_sections` is a concrete convenience on `HermesPromptSource` (a future `--list-sections` affordance), not part of the contract.
- `hermes_prompt_source.py` — `HermesPromptSource`, the splice primitive. `read` AST-walks top-level `NAME = "..."` string constants (v1 string-typed only; dict-typed constants like `PLATFORM_HINTS` raise `KeyError`). `write` splices by byte offset using `repr(new_text)` so the literal round-trips byte-equal regardless of embedded quotes/newlines, and `ast.parse`-guards the result before an atomic `os.replace` — it **refuses to write non-parseable Python**, leaving the user's Hermes startable.
- `prompt_module.py` — `PromptModule(section_name, candidate_text)`: a `dspy.Module` whose `ChainOfThought` passthrough predictor carries the candidate in `signature.instructions` between sentinel markers (`<!-- SECTION:name -->` … `<!-- /SECTION:name -->`). There is no cheap classification to score, so the predictor exists only as a mutation target. `forward()` **must** invoke the passthrough so GEPA captures a trace for `passthrough.predict` — without a traced predictor call, `make_reflective_dataset` finds "no valid predictions" and never proposes a mutation. It returns a placeholder response with `_closed_loop_task_id` + `_candidate_text` attached for the behavioral metric. GEPA discovers the target via `named_predictors()` → `"passthrough.predict"`.
- `prompt_proposer.py` — `PromptSectionProposer`, a sentinel-preserving GEPA `instruction_proposer` subclassing `BudgetAwareProposer` (inherits the char-budget infrastructure; see `budget_aware_proposer.py`). Runs the proposer LM, then passes the candidate through `extract_and_rebuild` so only the sentinel-delimited region survives. On a candidate that loses the sentinels it increments `sentinel_failures` and **re-raises** `SentinelParseError` rather than returning the parent unchanged — GEPA's reflective-mutation path skips the iteration instead of admitting a phantom identical-to-parent candidate into the selection pool.
- `prompt_judge.py` —
  - `SaveCallJudge` — LLM-as-judge scoring an individual memory-save's content against `MEMORY_GUIDANCE`'s rules (durable, declarative, fact-focused; not task progress / PR numbers / completed-work logs). Unparseable judge output falls back to a neutral 0.5 (logged so it's distinguishable from a real mediocre score).
  - `judge_save_calls` — the Layer-2 aggregate. Only judges `SAVE_ACTIONS = {add, replace}` (the real Hermes `memory` tool actions that carry a `content` payload; `remove` is not a save), caps judged calls at `MAX_JUDGED_CALLS_PER_TASK = 5` (excess score 0 each), and returns a vacuous 1.0 when there are no save calls or no judge/rubric is configured.
  - `make_prompt_fitness_metric` — the GEPA 5-arg metric. Routes purely behaviorally: a prediction missing `_closed_loop_task_id` is degenerate and scores 0 with a diagnostic; otherwise `closed_loop_scorer(task_id, candidate_text)` runs one closed-loop trial. Appends a `[BUDGET]` feedback line.
  - `make_memoizing_splice_scorer` — builds `closed_loop_scorer(task_id, candidate_text)` that splices **only when `candidate_text` changes** (consecutive tasks for one candidate reuse the live splice). Serialized under a `threading.Lock` because `dspy.Evaluate` is multi-threaded but `prompt_builder.py` is one shared mutable file — behavioral scoring is therefore effectively serial, an accepted v1 cost of splice-and-restore. Backup/restore is the caller's job (the guard wraps the whole run).

### Claude Code backend (`--target claude`)

The Claude backend evolves a CLAUDE.md region instead of a `prompt_builder.py` constant. It is three adapter classes behind the **existing** AgentRunner / PromptSource / ArtifactInstaller protocols — the agnostic core (GEPA, `ClosedLoopValidator`, `score_task`) is unchanged. Backend selection lives in a single factory, `build_backend` (`evolution/prompts/backend.py`), which returns a `PromptBackend` bundling `{source, installer, runner, baseline_text, deploy_target}` plus `install_candidate` (validation, via the installer) and `deploy` (`--apply`, via the source — a different file for claude). `evolve_prompt_section` carries **no** per-target branches; adding a third backend is a factory branch + adapter classes, nothing in the driver.

- `evolution/validation/claude_runner.py` — `ClaudeCodeAgentRunner` (implements `AgentRunner`). `run(ctx)` drives `claude -p <msg> --output-format stream-json --verbose --model <m> --no-session-persistence --strict-mcp-config --permission-mode bypassPermissions --add-dir <fixture> --allowedTools <list> [--append-system-prompt-file <file>]`. Runs hermetic: a fresh tmp `HOME` (no ambient `~/.claude` CLAUDE.md / plugins / memory). Filesystem confinement is **OS-enforced** — the subprocess is wrapped in macOS `sandbox-exec` with a write-restrict profile (`_macos_write_sandbox_profile`) that denies writes outside the fixture dir + per-run HOME + temp, confining the native Write/Edit tools (the Claude Code `sandbox` *setting* only confines Bash, so an agent escapes through Write — verified; it is deliberately not used). When OS sandboxing is unavailable the runner raises `SandboxUnavailableError` rather than executing an unconfined agent (override with `require_sandbox=False`). Parses the stream-json (`_parse_stream_json`): assistant `tool_use` content blocks → `tool_calls_seq` + `tool_calls_with_args`; the final `result` event → `total_cost_usd` + token usage. Auth via the `CLAUDE_CODE_OAUTH_TOKEN` env var (**not** `--bare`, which ignores the token). Integrates the shared `CostLedger` — `run()` records the agent cost and enforces the ceiling eagerly, mirroring `HermesAgentRunner`, because convention scoring makes no in-process LM call. Cost resolves actual(>0) → litellm token-pricing estimate (`computed`, so subscription runs that report $0 still meter) → `uncaptured`; an empty/degraded result stream (no tools, text, or tokens) abstains.
- `evolution/prompts/claude_prompt_source.py` — `ClaudeCodePromptSource` (implements `PromptSource` read/write). Reads/writes a sentinel-delimited region in a CLAUDE.md: `<!-- evolve:NAME start -->` … `<!-- evolve:NAME end -->`. `read(section)` returns the region body (raises `KeyError` if the markers are absent, `ValueError` if duplicated/reversed). `write(section, text)` replaces only the region body via an atomic write, preserving every other byte of the file; when the markers are absent it appends a fresh delimited block at EOF (first-time targets). Used to **seed** GEPA (read the baseline region) and to **deploy** on `--apply` — the only write to the user's real CLAUDE.md.
- `evolution/validation/artifact_installer.py` — `ClaudeAppendPromptInstaller` (implements `ArtifactInstaller`, alongside the existing installers). Mirrors `SkillFileInstaller`'s don't-touch-the-original contract: `target_path` is a throwaway `append_system_prompt.txt` in a caller-owned workdir; `install(source)` / `install_text(text)` write the candidate there and return its sha256; `verify_backup` requires UTF-8 + non-empty. The runner passes `target_path` to `--append-system-prompt-file`, so each candidate is injected statelessly — **validation never touches the user's CLAUDE.md** (only `--apply`, via the source, writes it).
- `evolution/validation/report.py` — `_score_convention` + the `expected_action == "convention"` branch in `score_task` (agent-agnostic). Convention-adherence verdict, no LLM judge: pass iff some `Bash` call's command used one of `required_cmd_substr` (the repo wrapper) AND no `Bash` command used any of `forbidden_cmd_substr` (the default tool to avoid). Matching is trailing-boundary aware, so forbidden `pytest` matches `python -m pytest` but not `pytest.ini`/`pytest_cache`, while required `bin/check` still matches `./bin/check`. Reads only `tool_calls_with_args`, so it is backend-independent. A convention task must declare a non-empty `required_cmd_substr` (validated at suite-parse time, else it would silently always-fail).
- `evolution/validation/task.py` — `Task` gained `required_cmd_substr: tuple[str, ...]` and `forbidden_cmd_substr: tuple[str, ...]`, parsed from the suite JSON (default empty tuples).
- `evolution/validation/suites/claude_conventions.jsonl` — example convention suite (4 tasks: custom test / run / fmt / lint wrappers). Each task is `expected_action:"convention"` with `required_cmd_substr` (the wrapper) + `forbidden_cmd_substr` (the default tool), and is bounded to `{fixture_dir}` so the sandboxed agent has no external referents.

Workflow (claude): read the seed region from the CLAUDE.md (or `--baseline-override-file`) → GEPA evolves the region, each candidate injected via `--append-system-prompt-file` and scored by running `claude -p` in the OS-sandboxed fixture (convention-adherence verdict, multi-rep) → deploy gate (`ClosedLoopValidator` on the holdout) → `--apply` splices the evolved region into the real CLAUDE.md via the source.

### Shared validation-stack changes that enable the prompt path

These let the prompt path reuse `ClosedLoopValidator` unchanged (see the validation section below for the base machinery):

- `HermesPromptSectionInstaller` (in `artifact_installer.py`) — implements the `ArtifactInstaller` Protocol. `target_path` = `agent/prompt_builder.py`; `install(text_file)` reads the candidate body and calls `HermesPromptSource.write`, returning the post-install `sha256`; `verify_backup` = `verify_python_parses`. Constraint: the section must be a top-level string constant.
- `ClosedLoopValidator` gained an optional `layer2_judge_factory` (per-task — prompt-section judging needs the task's `expected_save_content` rubric + message, which a single global fn couldn't carry) plus a `layer2_threshold`. When unset, scoring is Layer 1 only and the tool-description path is unchanged.
- `report.py`'s `score_task` gained the compound Layer 2: when a `layer2_judge_fn` is supplied a task passes only if Layer 1 (trigger membership) passes **and** the judge scores `>= layer2_threshold`. Layer 1 short-circuits — the judge is never called (no LLM cost) on a task that already failed the trigger test, and `test_command` mode ignores Layer 2. The judge receives the subset of `run.tool_calls_with_args` whose name is `memory`. `Task` gained `expected_save_content`; `AgentRunResult` gained `tool_calls_with_args`.
- `hermes_runner.py` (shared change): reads agent sessions from the SQLite `state.db` (`parse_session_from_db`) since the current one-shot `hermes -z` is ephemeral and no longer writes `session_*.json`. A row whose `tool_calls` column won't parse as JSON aborts with an `error` result (the task **abstains**) rather than being silently read as "no tools."

## evolution/validation/ — closed-loop validation against a real agent

Drives an actual agent (`HermesAgentRunner` via `hermes -z`) through a small task suite with baseline and evolved artifacts, scores real tool-selection behavior, compares. Orthogonal to skills/tools/prompts/code — measures agent behavior, not artifact production.

- `agent_runner.py` — `AgentRunner` Protocol + `AgentRunResult` dataclass + `TaskRunContext`. The Protocol shape leaves room for Claude Code or other backends without changing the validator — `ClaudeCodeAgentRunner` (`claude_runner.py`) is the second concrete implementation.
- `hermes_runner.py` — `HermesAgentRunner.run(ctx) -> AgentRunResult`. Subprocess `hermes -z "task"` with `HERMES_HOME` + `HOME` redirected to a per-task tmp dir and `cwd` set to the task's fixture dir. Scores from the session JSON only — `hermes -z` exit code is unreliable (returns 0 on agent-loop crashes). Tolerates both tool-call shapes Hermes emits (`{"function": {"name": ...}}` nested and flat `{"name": ...}`). Errors (timeout, no session JSON written, malformed JSON) become abstentions in the report.
- `task.py` — `Task` + `TaskSuite.from_jsonl`. JSONL with comment + blank-line tolerance; sha256 of the file bytes lands in the report so regression-by-curation (silently dropping a hard task) is auditable. Carries the convention-verdict fields `required_cmd_substr` / `forbidden_cmd_substr` (consumed by `score_task`'s `expected_action:"convention"` branch).
- `artifact_installer.py` — `ArtifactInstaller` Protocol + `HermesToolDescriptionInstaller`, `HermesPromptSectionInstaller`, `SkillFileInstaller`, and `ClaudeAppendPromptInstaller`. The tool/prompt installers splice into the live install via `apply_evolved` / `HermesPromptSource.write`; the skill and Claude installers write into a caller-owned workdir so the user's original is never touched (`ClaudeAppendPromptInstaller`'s `target_path` is the `--append-system-prompt-file` the runner injects). All return `sha256(target_path)` post-install for the validator's drift check. Helpers: `atomic_write_bytes`, `verify_python_parses`, `sha256_of`.
- `validator.py` — `ClosedLoopValidator.validate(inputs) -> ValidationReport`. The dangerous path: mutates the user's hermes-agent install in place. Three defenses:
  - `fcntl.flock` sentinel in the parent dir of the target tool file — concurrent harness runs (or a user running `hermes` interactively) fail fast with `ConcurrentRunError`.
  - `.cl_backup` written atomically and `ast.parse`-validated before being trusted for restore. The harness refuses to start if a stale, valid backup exists, naming the file so the user can `mv` it manually.
  - `sha256` verification after every task — a YOLO-mode agent that overwrites the spliced file mid-suite is caught with `ChecksumDriftError` before later tasks silently run a corrupt baseline.
- `report.py` — `ValidationReport` (JSON + Rich console rendering) plus the scoring (`score_task`), phase summary (`summarize_phase`), win/loss decomposition (`compute_win_loss`), and the two-condition decision rule (`decide`):
  - `evolved.pass_rate >= baseline.pass_rate`
  - `n_losses == 0` OR `n_wins >= 2 * n_losses`

  `score_task` supports four verdict modes, all agent-agnostic: trigger membership (`expected_tools` / `forbidden_tools`, optionally + Layer-2 content judge), `test_command` (skill-side), `expected_action:"patch"` (skill-patch), and `expected_action:"convention"` (`_score_convention` — Bash-command substring adherence, used by the Claude backend).
- `closed_loop.py` — Click CLI. Exit 0 on `pass`, 1 on `regression`. Drop-in compatible with `--benchmark-cmd`.
- `suites/*.jsonl` — JSONL task suites. `patch.jsonl` targets the `patch` vs `write_file` boundary; `write_file.jsonl` and `search_files.jsonl` cover the symmetric direction; `claude_conventions.jsonl` is the example Claude convention suite (`expected_action:"convention"` + `required_cmd_substr` / `forbidden_cmd_substr`, bounded to `{fixture_dir}`). Each task carries `task_id`, `user_message` (with optional `{fixture_dir}` placeholder), `expected_tools`, `forbidden_tools`, `fixture_setup` (relative-path → file-content dict materialized into a per-task tmp dir).

**During-evolution integration.** Beyond the standalone CLI, the same `ClosedLoopValidator` powers `evolution/core/closed_loop_feedback.py`'s `ClosedLoopFeedbackCache`. The cache writes the candidate description into a tmp manifest JSON, calls `validator.validate(ValidationInputs(...))` with it as `evolved_artifact`, and caches the returned `ValidationReport` by candidate text. The cache surfaces verdicts to the metric two ways: as a deterministic feedback block on the reflection path (`feedback` mode), or as per-task `TaskResult.passed` reads via `get_task_verdict(candidate, task_id)` for the behavioral-example branch (`trainset` mode). The validator itself doesn't know about the cache; it always sees a `ValidationInputs` with two artifacts and produces a `ValidationReport`.

## evolution/{code, monitor}/ — planned, empty

These packages exist as empty stubs anchoring the planned tier-4/5 work. See `PLAN.md` for the design. (`prompts/` is now implemented — see the phase-3 section above.)
