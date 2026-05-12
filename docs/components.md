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

**LM hardening:** `request_timeout=60, num_retries=5` on the judge LM. 60s = 6× P99 of slowest observed `gpt-4.1-mini` call (9.8s). 5×60s = 5min worst case before raising.

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

## evolution/{prompts, code, monitor}/ — planned, empty

These packages exist as empty stubs anchoring the planned tier-3/4/5 work. See `PLAN.md` for the design.
