# Data Models

The dataclasses, schemas, and on-disk formats the framework uses.

## EvolutionConfig (`evolution/core/config.py:18`)

```python
@dataclass
class EvolutionConfig:
    skill_sources: list[SkillSource] = field(default_factory=lambda: discover_skill_sources())

    # Optimization
    iterations: int = 10
    population_size: int = 5

    # LLMs
    optimizer_model: str = "openai/gpt-4.1"
    reflection_model: Optional[str] = None  # falls back to optimizer_model
    eval_model: str = "openai/gpt-4.1-mini"
    judge_model: str = "openai/gpt-4.1"

    # Static constraints
    max_skill_size: int = 15_000            # absolute deployment-cost backstop
    max_tool_desc_size: int = 500
    max_param_desc_size: int = 200

    # Quality-gated growth curve
    growth_free_threshold: float = 0.20     # required(growth) = max(0, slope*(growth-free))
    growth_quality_slope: float = 0.30
    max_absolute_chars: int = 5000          # hard ceiling, independent of growth %
    gate_mode: str = "no_regression"        # "no_regression" | "non_inferiority"
    inferiority_tolerance: float = 0.0      # only used when gate_mode == "non_inferiority"

    # Bootstrap CI
    bootstrap_confidence: float = 0.90
    bootstrap_n_resamples: int = 2000

    # Eval dataset
    eval_dataset_size: int = 150
    train_ratio: float = 0.5
    val_ratio: float = 0.40
    holdout_ratio: float = 0.50
    min_holdout_size: int = 10              # hard refuse-to-gate threshold

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    create_pr: bool = True

    # Determinism
    seed: int = 42                          # forwarded to dataset shuffle + GEPA/MIPROv2
```

`skill_sources` runs `discover_skill_sources()` at construction time. Tests use a `_skill_source_env` autouse fixture to point this at a fake repo so they don't pick up real `~/.hermes` or `~/.claude` installations.

`val_ratio + holdout_ratio + train_ratio` is **deliberately not 1.0** — the synthetic builder normalizes them to sum to 1, so changing any one shifts the others proportionally. Default normalizes to ≈ 0.36/0.29/0.36 of N (≈ 54 train / 43 val / 53 holdout at the default `eval_dataset_size=150`).

## EvalExample (`evolution/core/dataset_builder.py:21`)

```python
@dataclass
class EvalExample:
    task_input: str                # what the user asks
    expected_behavior: str         # rubric — what a good response looks like
    difficulty: str = "medium"     # easy | medium | hard
    category: str = "general"      # for stratified eval (not currently used)
    source: str = "synthetic"      # synthetic | sessiondb | golden | (any string)
```

`source` is consumed by `_dataset_payload()` to bucket per-source counts in `gate_decision.json`. Empty/None source is bucketed as `"unknown"`.

`category` defaults to `"general"` on skill datasets. Tool-selection datasets produced by `SyntheticDatasetBuilder.generate_tool_selection` populate it with one of `"target_correct"` (target is the correct tool), `"confusable_neighbor"` (a known-confusable neighbor is the correct tool — surfaces cross-tool regressions), or `"regression_detection"` (an unrelated tool is correct — guards against the evolved description "stealing" selections). Bucketed per-category counts are recorded in `gate_decision.json.dataset.categories` on tool-path runs.

`to_dict()` and `from_dict()` round-trip through JSONL. The on-disk format is one example per line:

```jsonl
{"task_input": "Find all .py files modified in last week", "expected_behavior": "Use find with -mtime -7 and -name '*.py'", "difficulty": "easy", "category": "filesystem", "source": "synthetic"}
```

## EvalDataset (`evolution/core/dataset_builder.py:43`)

```python
@dataclass
class EvalDataset:
    train: list[EvalExample] = field(default_factory=list)
    val: list[EvalExample] = field(default_factory=list)
    holdout: list[EvalExample] = field(default_factory=list)
```

- `all_examples` property: train + val + holdout.
- `save(path)` writes `{train,val,holdout}.jsonl` under `path/`.
- `load(path)` reads them back.
- `to_dspy_examples(split)` converts to `dspy.Example` objects with `with_inputs("task_input")`.

On-disk layout:
```
datasets/skills/<skill>/
├── train.jsonl
├── val.jsonl
└── holdout.jsonl
```

## FitnessScore (`evolution/core/fitness.py:18`)

```python
@dataclass
class FitnessScore:
    correctness: float = 0.0           # 0-1
    procedure_following: float = 0.0   # 0-1
    conciseness: float = 0.0           # 0-1
    length_penalty: float = 0.0        # 0-1, 0 = no penalty
    feedback: str = ""                 # judge's natural-language critique

    @property
    def composite(self) -> float:
        raw = 0.5*correctness + 0.3*procedure_following + 0.2*conciseness
        return max(0.0, raw - length_penalty)
```

Composite is what GEPA's metric returns as `score`. Length penalty ramps from 0 at 90% of `max_size` to 0.3 at 100%+ of `max_size` (ratio capped at 0.3).

## ConstraintResult (`evolution/core/constraints.py:15`)

```python
@dataclass
class ConstraintResult:
    passed: bool
    constraint_name: str    # "size_limit" | "non_empty" | "skill_structure" |
                            # "growth_quality_gate" | "absolute_char_ceiling" | "test_suite"
    message: str            # human-readable summary
    details: Optional[str] = None
```

Constraint names are stable strings — tests assert against them, and `gate_decision.json.failed_constraints` is a list of these names.

## CandidatePick (`evolution/skills/knee_point.py:27`)

```python
@dataclass(frozen=True)
class CandidatePick:
    module: Any                   # the picked candidate module (SkillModule)
    skill_text: str               # extracted from module.skill_text
    body_chars: int               # parsimony metric
    val_score: float              # picked candidate's val aggregate
    val_rank_in_band: int         # 1-indexed; 1 = highest val in band
    band_size: int                # candidates within ε of best
    epsilon: float                # ε used for the band
    fallback: str                 # "knee" | "static_failed_all"
    picked_idx: int               # index into the original candidates list
    gepa_default_idx: int         # for comparison telemetry
    gepa_default_body_chars: int  # for comparison telemetry
    band_roster: list[dict]       # [{"idx", "val_score", "body_chars"}, ...]
```

Frozen for safety — once selected, the pick + diagnostics shouldn't be mutated. `band_roster` is sorted by descending `val_score`, ties broken by `idx`.

## ToolEntry (`evolution/tools/tool_source.py`)

```python
@dataclass(frozen=True)
class ToolEntry:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    source_kind: Literal["literal", "name_ref", "joined_str"] | None = None
    source_location: tuple[Path, int, int, int, int] | None = None
```

Validated at load by `ToolEntry.from_dict()`: `name` must match `^[a-zA-Z0-9_-]{1,128}$`. Names outside this set break sentinel parsing (regex metacharacters, embedded `-->`) and are rejected with a clear error. `from_dict` reads `inputSchema` (MCP shape) into `input_schema`.

`source_kind` and `source_location` are **optional adapter-state fields** populated by source-walking adapters (`HermesToolSource`). They tell the write path how to splice the evolved description back in:
- `source_kind` is `"literal"` for plain string-constant descriptions, `"name_ref"` for descriptions that reference a top-level string constant (the write path edits the constant rather than the schema dict), and `"joined_str"` for f-string descriptions (write path refuses; the caller must convert to a literal first).
- `source_location` is `(file_path, lineno, col_offset, end_lineno, end_col_offset)` of the description string node. For `name_ref`, it points at the resolved constant.

Both fields are `None` for JSON-backed manifests (`MCPManifestSource`) — the MCP write path doesn't need a span because it round-trips the full JSON dict.

Frozen at the dataclass level — the attribute itself can't be rebound — but `input_schema` is not deep-frozen. Mutating it in place corrupts any other `ToolEntry` / `ToolManifest` that shares the reference (by design: `ToolManifest.replace_description` preserves the original reference).

## ToolManifest (`evolution/tools/tool_source.py`)

```python
@dataclass(frozen=True)
class ToolManifest:
    tools: tuple[ToolEntry, ...]
    confusable_neighbors: dict[str, str] = field(default_factory=dict)
    dropped_tools: tuple[tuple[str, str], ...] = ()
```

Helpers:
- `from_json_file(path) -> ToolManifest` — reads an MCP `list_tools()`-shape file. `_evolution_metadata.confusable_neighbors` is optional metadata for cross-tool regression evaluation.
- `find_tool(name) -> ToolEntry` — raises `KeyError` listing available tools on miss.
- `confusable_neighbor_for(name) -> str | None`.
- `replace_description(name, new_description) -> ToolManifest` — returns a new manifest with the named tool's description swapped; every other tool's `description` and `input_schema` are preserved by reference.

`dropped_tools` is a tuple of `(name_hint, reason)` pairs surfaced by source-walking adapters for schemas they couldn't parse statically (e.g., dicts built from function calls, or descriptions that aren't literal strings, name refs, or f-strings). Empty for JSON manifests; populated by `HermesToolSource` when an `*_SCHEMA` assignment can't be reached via pure AST. The pairs are echoed into `gate_decision.json.dataset.dropped_tools` so users see what was excluded.

Rejects at load: an empty `tools` tuple, and normalization collisions (`read-file` vs `read_file`, which both lowercase + underscore-normalize to `read_file`). Sentinel matching uses original casing but lookup robustness relies on normalization being injective.

## Closed-loop validation types

These live under `evolution/validation/` and are produced by `ClosedLoopValidator.validate()` (the standalone CLI's primary output) and consumed by `ClosedLoopFeedbackCache` (the during-evolution integration).

### Task (`evolution/validation/task.py`)

```python
@dataclass(frozen=True)
class Task:
    task_id: str
    user_message: str           # may contain {fixture_dir} placeholder
    expected_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    fixture_setup: dict[str, str] = field(default_factory=dict)
    test_command: Optional[str] = None          # skill-side: pass iff this exits 0 in fixture_dir
    expected_save_content: Optional[str] = None  # Layer-2 rubric for memory-save content
    skills_src: Optional[str] = None
    expected_action: Optional[str] = None        # "patch" | "convention" — action-level verdict mode
    target_skill: Optional[str] = None           # for expected_action == "patch"
    stale_token: Optional[str] = None            # for expected_action == "patch"
    required_cmd_substr: tuple[str, ...] = ()    # for expected_action == "convention"
    forbidden_cmd_substr: tuple[str, ...] = ()   # for expected_action == "convention"
```

`fixture_setup` is a `relative_path → file_content` map materialized into the task's per-task tmp dir before the agent runs. `user_message.format(fixture_dir=...)` substitutes the placeholder.

Scoring rule (`score_task` in `report.py`) — returns `(passed: bool, abstained: bool)`; abstention if the runner errored (timeout, no session, parse failure), neither evidence for nor against the artifact. Otherwise the verdict mode is chosen by which fields are set, in priority order:
- **`expected_action == "patch"`** (with `target_skill` + `stale_token`): pass iff the agent called `skill_manage(action in {patch, edit})` on `target_skill` and the call actually touched the stale token.
- **`expected_action == "convention"`** (Claude convention suites; `_score_convention`): pass iff some `command_tool` call (default `Bash`) used one of `required_cmd_substr` (the agent used the repo wrapper) AND no such command used any of `forbidden_cmd_substr` (it didn't fall back to the default tool). Substring matching is trailing-boundary aware (forbidden `pytest` won't match `pytest.ini`). No LLM judge — reads only `tool_calls_with_args`, so the verdict is agent-backend-independent. A convention task must declare a non-empty `required_cmd_substr` (rejected at parse time otherwise — it could never pass).
- **`test_command` set** (skill-side): pass iff the command exits 0 in `fixture_dir`.
- **Default (trigger membership):** passes iff `expected_tools` were invoked AND no `forbidden_tools` were invoked. Empty `expected_tools` short-circuits to true; empty `forbidden_tools` is no-op. When a `layer2_judge_fn` is supplied (prompt-section save suites), a passing Layer 1 additionally requires the content judge to score `>= layer2_threshold`.

### TaskSuite (`evolution/validation/task.py`)

```python
@dataclass(frozen=True)
class TaskSuite:
    path: Path
    sha256: str           # of the file bytes — lands in ValidationReport for audit
    tasks: tuple[Task, ...]
```

`TaskSuite.from_jsonl(path)` reads the file, skipping blank lines and `#`-prefixed comment lines, raising `ValueError` with `path:lineno` on parse errors. The sha256 is included in every `ValidationReport.task_suite_sha256` so a silent "drop a hard task" is auditable at code review time.

### TaskResult (`evolution/validation/report.py`)

```python
@dataclass(frozen=True)
class TaskResult:
    task_id: str
    passed: bool
    abstained: bool
    tool_calls_seq: list[str]
    duration_seconds: float
    model_name: Optional[str] = None
    error: Optional[str] = None
```

`tool_calls_seq` is the ordered list of tool names invoked during the agent's response. The behavioral-example metric in `_score_behavioral_example` reads `passed` directly to produce a binary score; `tool_calls_seq` lands in `render_feedback_block` for the reflection LM.

### PhaseResult / WinLoss / ValidationReport (`evolution/validation/report.py`)

```python
@dataclass(frozen=True)
class PhaseResult:
    pass_rate: float
    n_passed: int
    n_failed: int
    n_abstained: int
    tasks: list[TaskResult]

@dataclass(frozen=True)
class WinLoss:
    n_wins: int                # per-task: evolved passed, baseline failed
    n_losses: int              # per-task: evolved failed, baseline passed
    n_ties: int
    pass_rate_change: float    # evolved.pass_rate - baseline.pass_rate

@dataclass(frozen=True)
class ValidationReport:
    schema_version: str        # currently "1"
    tool: str
    task_suite_path: str
    task_suite_sha256: str
    baseline: PhaseResult
    evolved: PhaseResult
    delta: WinLoss
    decision: str              # "pass" | "regression"
    decision_reasons: list[str]
```

Two-condition decision rule (`decide()`):
1. `evolved.pass_rate >= baseline.pass_rate` (aggregate no-regression).
2. `n_losses == 0` OR `n_wins >= 2 * n_losses` (per-task: wins need to overwhelm losses 2:1).

Both must hold to return `"pass"`; else `"regression"`. The 2:1 win-loss ratio is the threshold for "robustly better" given LM non-determinism — observed at ~15-20% per-task flip rate on borderline tasks, so a single flip should not dominate.

`ValidationReport.to_dict()` round-trips to `validation_report.json` written under `output/validation/<tool>/<timestamp>/`.

## SaturationReport (`evolution/core/saturation_check.py`)

In-memory only. Built by `saturation_preflight(...)` before GEPA setup, consumed by the call site in `evolve_skill` / `evolve_tool` to decide whether to abort or proceed. Not currently serialized to disk — the `holdout_per_example` list flows directly into the post-GEPA `_holdout_evaluate_with_metric` baseline-cache reuse path.

```python
@dataclass
class SaturationReport:
    band: SaturationBand                          # "healthy" | "no_headroom" | "weak_signal" | "uniform_failure"
    holdout_score: float                          # baseline mean on holdout
    holdout_n: int                                # number of holdout examples scored
    holdout_per_example: list[float]              # per-example scores (reused at post-GEPA evaluation)
    closed_loop_score: Optional[float] = None     # None when no --closed-loop-during-evolution suite
    closed_loop_n: Optional[int] = None           # number of behavioral tasks scored
    closed_loop_per_example: Optional[list[float]] = None
    suggestions: list[str] = field(default_factory=list)   # band-specific user-facing strings
    thresholds: dict[str, float] = field(default_factory=dict)   # snapshot of values that produced the band
```

`SaturationBand` is a `Literal` of four strings. `DEFAULT_THRESHOLDS` ships as `{no_headroom_synthetic: 0.99, weak_signal_synthetic: 0.95, no_headroom_closed_loop: 0.95, uniform_failure_closed_loop: 0.15}`. See `components.md`'s `saturation_check.py` section for the classifier logic.

## Evolved manifest output JSON

`output/tools/<tool>/<timestamp>/evolved_manifest.json` (deploy) and `evolved_FAILED.json` (reject) have the same shape as the input MCP-shape manifest:

```json
{
  "tools": [
    {"name": "search_files", "description": "<evolved description>", "inputSchema": {...}},
    {"name": "grep_in_terminal", "description": "<unchanged>", "inputSchema": {...}}
  ],
  "_evolution_metadata": {
    "confusable_neighbors": {"search_files": "grep_in_terminal"}
  }
}
```

Only the target tool's `description` is changed; every other tool's `description`, `inputSchema`, and any `_evolution_metadata` block are preserved verbatim. With `--apply`, the source manifest file is rewritten in place with the same preservation guarantees. With `--patch`, a unified diff of (baseline → evolved) manifest JSON is written to stdout.

## gate_decision.json (schema_version "5")

The structured deploy-gate decision, written to `output/<skill>/<timestamp>/gate_decision.json` on every run regardless of outcome. The schema is the **calibration substrate** — `tests/skills/test_evolve_skill_validation_flow.py:TestGrowthGateDecisionSchema` locks the field list so future calibration scripts (`jq -s '...' output/*/*/gate_decision.json`) don't break.

### Static-failure variant

Written when any `validate_static` check fails on the evolved artifact (short-circuits before holdout):

```json
{
  "schema_version": "5",
  "decision": "reject",
  "reason": "static_constraint_failure",
  "failed_constraints": ["non_empty"],
  "messages": ["Artifact is empty"],
  "knee_point": { "applied": false, "reason": "no_detailed_results" },
  "dataset": { "size_total": 60, "size_train": 21, "size_val": 17, "size_holdout": 22, "sources": {"synthetic": 60} }
}
```

### Benchmark-hook block (opt-in, present when `--benchmark-cmd` was set)

When the run uses `--benchmark-cmd`, the gate decision carries an extra `benchmark` block regardless of pass/fail:

```json
"benchmark": {
  "command": "pytest -k smoke",
  "exit_code": 0,                 // null on timeout / spawn error
  "duration_seconds": 12.4,
  "stdout_tail": "...4096 chars max...",
  "stderr_tail": "...4096 chars max...",
  "passed": true,
  "reason": "ok"                  // "ok" | "exit_nonzero" | "timeout" | "command_error"
}
```

When `passed=false`, the top-level `decision` is `"reject"` and `reason` is `"benchmark_failed"`. The benchmark hook only runs when the framework's own deploy gate would deploy — if `growth_quality_gate` already rejected, the hook is skipped and the `benchmark` block is absent (no point spending the user's CI budget on a variant we already decided not to ship).

### Cost-ceiling-abort variant

Written when `--max-total-cost-usd` is set and cumulative LM cost exceeds the ceiling. The next LM call after the ceiling trips raises `CostCeilingExceeded` from `LMTimingCallback.on_lm_start`, which the orchestrator catches at top level. Worst-case overshoot is one LM call past the ceiling — `cost_at_abort_usd` shows what was actually spent.

```json
{
  "schema_version": "5",
  "decision": "aborted",
  "reason": "cost_ceiling_exceeded",
  "cost_ceiling_usd": 0.50,
  "cost_at_abort_usd": 0.524,
  "cost_summary": {                              // mirrors metrics.json.cost
    "total_usd": 0.524,
    "by_model": {
      "openai/gpt-4.1-mini": {
        "tokens_in_uncached": 12000,
        "tokens_in_cached": 0,
        "tokens_out": 800,
        "reasoning_tokens": 0,
        "cost_usd": 0.524,
        "calls": 28,
        "cache_hit_rate": 0.0
      }
    }
  },
  "run_inputs": { /* same shape as the deploy/reject variant */ },
  // Tool-path runs additionally include:
  "artifact_type": "tool_description",
  "target_tool": "search_files"
}
```

`decision="aborted"` is a third value alongside `"deploy"` and `"reject"` — calibration scripts that group by decision should add it to the dimension. The schema test (`TestGrowthGateDecisionSchema`) doesn't enforce this variant since it only fires on opt-in `--max-total-cost-usd` and an additive third decision value doesn't break the existing required-field set.

### Growth-quality-gate variant (deploy or reject)

```json
{
  "schema_version": "5",
  "decision": "deploy",                          // or "reject"
  "reason": "passed",                            // or "growth_quality_gate"
  "decision_rule_used": "dual_check",            // or "no_regression_only" | "non_inferiority"
  "gate_mode": "no_regression",                  // "no_regression" | "non_inferiority"
  "inferiority_tolerance": 0.0,                  // only meaningful when gate_mode == "non_inferiority"
  "growth_pct": 0.242,                           // (evolved_chars - baseline_chars) / baseline_chars
  "required_improvement": 0.013,                 // max(0, slope * (growth - free))
  "baseline_chars": 1264,
  "evolved_chars": 1570,
  "absolute_char_ceiling": 5000,                 // static config value (EvolutionConfig.max_absolute_chars)
  "effective_absolute_char_ceiling": 5000,       // max(static_floor, 1.5 × baseline_chars) — what was actually enforced
  "growth_free_threshold": 0.20,
  "growth_quality_slope": 0.30,
  "bap_max_growth": 0.20,                        // BudgetAwareProposer's prompt target for the reflection LM
  "bap_safety_margin": 0.10,                     // BAP's safety cushion (default 0.10; lower for calibration)
  "fitness_profile": "balanced",                 // "balanced" | "compression" | "growth"
  "proposer_mode": "balanced",                   // "compression" | "balanced" | "growth" — which BudgetAwareProposer template ran
  "baseline_per_example": [0.5, 0.6, /* ... */],  // float per holdout example
  "evolved_per_example":  [0.51, 0.61, /* ... */],
  "avg_baseline": 0.6,
  "avg_evolved":  0.605,
  "bootstrap": {
    "mean":         0.005,
    "lower_bound": -0.020,
    "upper_bound":  0.030,
    "n_examples":   22,
    "n_resamples":  2000,
    "confidence":   0.90
  },
  "win_loss": {
    "n_wins":             14,                     // count of holdout examples where evolved > baseline
    "n_losses":            5,
    "n_ties":              3,
    "worst_regression":  -0.30,                   // most-negative per-example delta (evolved - baseline)
    "worst_improvement":  0.45                    // most-positive per-example delta
  },
  "run_inputs": {
    "seed":                42,
    "iterations":          10,
    "optimizer_model":     "openai/gpt-4.1",
    "reflection_model":    "openai/gpt-5-mini",
    "eval_model":          "openai/gpt-4.1-mini",
    "eval_dataset_size":   150,
    "holdout_ratio":       0.50,
    "quality_gate_preset": "default",
    "eval_source":         "synthetic"
  },
  "failed_constraints": [],                       // names from ConstraintResult.constraint_name
  "messages": [],                                 // human-readable summaries
  "knee_point": {
    "applied":                  true,
    "fallback":                  "knee",          // or "static_failed_all"
    "epsilon":                   0.0588,          // 1/n_val by default
    "band_size":                 4,
    "picked_idx":                12,
    "picked_val_score":          0.95,
    "picked_val_rank_in_band":   3,               // 1-indexed
    "picked_body_chars":         412,
    "gepa_default_idx":          5,
    "gepa_default_body_chars":   1572,
    "band_roster": [                              // sorted by val_score desc
      {"idx": 5,  "val_score": 0.997, "body_chars": 1572},
      {"idx": 12, "val_score": 0.95,  "body_chars": 412}
    ]
  },
  "dataset": {
    "size_total":   60,
    "size_train":   21,
    "size_val":     17,
    "size_holdout": 22,
    "sources":     {"synthetic": 60}              // or {"sessiondb_claude_code": 12, "golden": 8, ...}
  }
}
```

### Effective absolute-char ceiling

`absolute_char_ceiling` records the static `EvolutionConfig.max_absolute_chars` value (default 5000). `effective_absolute_char_ceiling` records `max(static_floor, 1.5 × baseline_chars)` — what the validator actually enforced. The two are equal for skills with baseline ≤ ~3300 chars; for larger skills the effective ceiling scales with baseline so artifacts that already exceed 5000 don't auto-reject pre-gate.

### BAP-related fields

`bap_max_growth` is the proposer's prompt target for the reflection LM (decoupled from `growth_free_threshold`; see `EvolutionConfig.bap_max_growth`, default 0.20). `bap_safety_margin` is the cushion the proposer subtracts from `max_growth` to absorb the LM's overshoot tendency (default 0.10).

### `fitness_profile`

Records which composite-weighting profile the LLM judge used: `balanced` (default, 0.5/0.3/0.2 for correctness/procedure/conciseness), `compression` (0.4/0.2/0.4), or `growth` (0.6/0.4/0.0).

### `proposer_mode`

Which `BudgetAwareProposer` prompt template generated the candidates: `compression` (cut redundancy, stay under budget), `balanced` (direction-agnostic, soft target with ±20% tolerance), or `growth` (add only what feedback identifies as missing). Mapping: `fitness_profile == "growth"` → `proposer_mode == "growth"`; `fitness_profile == "balanced"` → `proposer_mode == "balanced"`; `fitness_profile == "compression"` (and any unrecognized value) → `proposer_mode == "compression"`. Recorded explicitly so historical runs stay analysable if the mapping changes in the future.

### `win_loss` block

Per-example win/loss decomposition computed from `baseline_per_example` and `evolved_per_example`. The deploy/reject logic does not consume this block — it's pure information for users who want to see the distribution behind the aggregate mean (e.g., "60% wins / 40% losses" vs. "100% small wins" can have the same mean lift but very different operational risk).

### `run_inputs` block

Records the inputs to the run so a third party with the artifact alone can reproduce the result: seed, iterations, model versions (optimizer/reflection/eval), dataset size + holdout ratio, the resolved `--quality-gate` preset name, and the eval source.

### Decision rule mapping
- `decision_rule_used == "no_regression_only"` ⟺ `required_improvement == 0.0` AND `gate_mode == "no_regression"` (default). Pass requires `bootstrap.mean >= 0`.
- `decision_rule_used == "non_inferiority"` ⟺ `required_improvement == 0.0` AND `gate_mode == "non_inferiority"`. Pass requires `bootstrap.lower_bound > -inferiority_tolerance` (Decagon-style; recommended for compression-focused runs at small N).
- `decision_rule_used == "dual_check"` ⟺ `required_improvement > 0`. Pass requires `bootstrap.mean ≥ required_improvement` AND `bootstrap.lower_bound > 0`.

### Knee-point applied/skipped
- `knee_point.applied: false` lands when MIPROv2 fallback fired (no `detailed_results` on the optimized module).
- `knee_point.applied: true` always carries the full diagnostic block.

### Tool-path additions (`artifact_type == "tool_description"`)

Runs of `evolution.tools.evolve_tool` write the same schema with four extra top-level fields:

| Field | Type | Notes |
|---|---|---|
| `artifact_type` | `"skill" \| "tool_description"` | Present on every gate decision. Skill runs always set `"skill"`. |
| `target_tool` | `str \| None` | Set only when `artifact_type == "tool_description"`. The tool whose description was optimized. |
| `manifest_neighbor_count` | `int \| None` | Set only on tool runs. Equals `len(manifest.tools) - 1` — the number of confusable peers the selector had to disambiguate against. |
| `sentinel_failures` | `int \| None` | Set only on tool runs. Count of reflection-LM outputs the proposer rejected for failing sentinel preservation. A high count signals the reflection LM is struggling with the constraint and the run may be wasting iterations. |

`dataset.categories` namespace is **disjoint per eval source** on the tool path. Synthetic runs produce `{"target_correct": N, "confusable_neighbor": N, "regression_detection": N}` from the three-bucket generator. SessionDB runs produce `{"agreed": N, "misselection": N}` from the confidence-banded judge: `"agreed"` means the judge concurred with the agent's actual tool choice, `"misselection"` means it disagreed at confidence ≥0.85 and `expected_behavior` was flipped to the judge's pick. Skill runs use `{"general": N}`. Calibration scripts should branch on `run_inputs.eval_source` before interpreting the category mix.

`dataset.dropped_tools` is a list of `[name_hint, reason]` 2-lists naming schemas the source adapter saw but couldn't parse statically (e.g., dicts built from function calls, descriptions that aren't literal strings / name refs / f-strings). Empty `[]` on the MCP-JSON path where nothing is dropped; populated on the Hermes Python-source path so users see what was excluded from evaluation.

#### SessionDB-only fields (`run_inputs.eval_source == "sessiondb"`)

| Field | Type | Notes |
|---|---|---|
| `dataset.sessiondb_drops` | `dict[str, int]` | Per-reason drop counts across the two pipeline stages. Importer keys: `short_task`, `slash_command`, `secret`, `no_tool_calls`, `non_manifest`. Judge keys: `judge_irrelevant`, `judge_error`, `noisy_middle`, `low_confidence`, `unknown_correct_tool`. Judge keys are absent when zero candidates reached the judge stage. |
| `dataset.dropped_non_manifest_count` | `int` | Pulled out of `sessiondb_drops["non_manifest"]` as a top-level int so calibration scripts don't have to know the inner key set. Counts session invocations of tools that exist in the historical session but not in the current manifest under evolution. |

### Schema v5 additions

v5 adds always-present `decision_signal` and `pr_created` fields, plus a closed-loop-primary field group that is present only when the deploy gate was decided on closed-loop signal rather than the synthetic holdout.

| Field | Type | Notes |
|---|---|---|
| `decision_signal` | `"synthetic" \| "closed_loop"` | Always present. Which signal the deploy gate actually decided on. `"closed_loop"` lands when the run executed CL-primary scoring (closed-loop tasks gained ≥ `cl_required_gain` AND synthetic non-inferiority held); `"synthetic"` otherwise. Calibration scripts should branch on this before interpreting `bootstrap` vs `cl_tasks_gained`. |
| `pr_created` | `dict` | Always present. Shape-stable across `--create-pr` on/off and across success/failure. Keys: `status` (`"created" \| "skipped" \| "failed" \| "disabled"`), `reason` (`str \| None`), `branch` (`str \| None`), `commit_sha` (`str \| None`), `url` (`str \| None`). `"disabled"` is the default when `--create-pr` is off. |

#### Closed-loop-primary fields (`decision_signal == "closed_loop"`)

Written by `evolution/core/quality_gate.py::append_cl_decision_fields` when the gate decision is taken on closed-loop signal.

| Field | Type | Notes |
|---|---|---|
| `baseline_closed_loop_per_example` | `list[float]` | Cached per-task closed-loop scores for the baseline artifact (0.0/1.0 per task). |
| `evolved_closed_loop_per_example` | `list[float]` | Per-task closed-loop scores for the evolved artifact (0.0/1.0 per task). Same length and task order as `baseline_closed_loop_per_example`. |
| `evolved_closed_loop_errored_tasks` | `list` | Task identifiers (or empty) for closed-loop evaluations that errored rather than scored. Empty list is the common case. |
| `cl_tasks_gained` | `int` | `int(sum(evolved)) - int(sum(baseline))` — the net delta of tasks passing closed-loop. The CL-primary gate requires this to meet `cl_required_gain`. |
| `cl_required_gain` | `int` | The CL-primary threshold the run had to clear, computed from `growth_pct` via the CL-primary slope/free-threshold constants. At least `1` for any non-zero growth. |
| `synthetic_sanity_check` | `dict` | The non-inferiority guard that runs alongside CL-primary. Keys: `tolerance` (float), `baseline_mean` (float), `evolved_mean` (float), `passed` (bool — `(evolved - baseline) >= -tolerance`). |
| `evolved_cl_eval_cost_usd` | `float` | LM cost in USD attributable to the evolved closed-loop evaluation pass — surfaces the CL-primary path's incremental spend. |
| `band_trigger_score` | `dict` | Pre-flight scores that decided whether CL-primary fired. Keys: `holdout` (`float \| None`), `closed_loop` (`float \| None`). |
| `validator_agent_model` | `str` | The LiteLLM model id used for the closed-loop validator agent. Recorded so historical decisions stay analysable if the default changes. |

### Prompt-section additions (`artifact_type == "prompt_section"`)

Runs of `evolution.prompts.evolve_prompt_section` (Phase 3) write the same `schema_version` "5" envelope but a **deliberately different field set** from the skill/tool variant, because the deploy gate is a closed-loop pass-rate / win-loss decision, **not** a paired-bootstrap one. There is no synthetic classification signal for a system-prompt section — every candidate is scored behaviorally by a real `hermes -z` against a curated suite — so the bootstrap substrate doesn't apply.

```json
{
  "schema_version": "5",
  "artifact_type": "prompt_section",
  "target_section": "MEMORY_GUIDANCE",
  "decision": "deploy",                          // "deploy" | "reject" | "denied" | "dry_run" | "aborted"
  "decision_signal": "closed_loop",              // always "closed_loop" on this path
  "baseline_chars": 1840,
  "evolved_chars": 2104,
  "growth_pct": 0.143,                           // (evolved_chars - baseline_chars) / baseline_chars
  "closed_loop": {
    "decision": "pass",                          // "pass" | "regression" (ValidationReport.decision)
    "decision_reasons": ["pass_rate 0.92 >= baseline 0.75", "n_wins 4 >= 2*n_losses 0"],
    "baseline_pass_rate": 0.75,
    "evolved_pass_rate": 0.92,
    "n_wins": 4,
    "n_losses": 0,
    "n_ties": 8
  },
  "sentinel_failures": 1,                         // reflection-LM outputs the proposer rejected for breaking sentinel preservation
  "elapsed_seconds": 412.6,
  "cost": { /* same shape as cost_summary: total_usd + by_model */ },
  "run_inputs": { /* seed, iterations, model versions, suite path/sha, validator_agent_model, ... */ },
  "pr_created": { "status": "skipped", "reason": "prompt_section_pr_unsupported", "branch": null, "commit_sha": null, "url": null }
}
```

**Fields this variant carries** (and the tool/skill variant does not, or differs on):

| Field | Type | Notes |
|---|---|---|
| `artifact_type` | `"prompt_section"` | Disjoint from `"skill"` / `"tool_description"`. |
| `target_section` | `str` | The `prompt_builder.py` constant whose text was evolved (e.g. `MEMORY_GUIDANCE`). |
| `decision` | `"deploy" \| "reject" \| "denied" \| "dry_run" \| "aborted"` | `"denied"` lands on a saturation pre-flight default-deny; `"dry_run"` when the run was asked to evaluate without splicing; `"aborted"` on cost-ceiling / interrupt. |
| `decision_signal` | `"closed_loop"` | Always `"closed_loop"` here — the synthetic value never appears on this path. |
| `baseline_chars` / `evolved_chars` / `growth_pct` | int / int / float | Size telemetry; growth informs the closed-loop required-gain threshold but is not gated on a bootstrap. |
| `closed_loop` | `dict` | `{decision, decision_reasons, baseline_pass_rate, evolved_pass_rate, n_wins, n_losses, n_ties}` — the deploy gate's primary evidence (sourced from `ValidationReport` over the behavioral suite). |
| `sentinel_failures` | `int` | Count of reflection-LM proposals rejected for failing sentinel preservation (same meaning as the tool path). |
| `elapsed_seconds` / `cost` | float / dict | Wall-clock + per-model cost ledger. |
| `run_inputs` | `dict` | Reproduction inputs (seed, iterations, models, suite path + sha, `validator_agent_model`). |
| `pr_created` | `dict` | Shape-stable with the skill/tool path, but the prompt-section path currently emits a `status: "skipped"` block (PR automation for in-place `prompt_builder.py` splices is not wired). |

**Fields the prompt-section variant deliberately OMITS.** A reader or calibration script must not assume these are present — they exist only on the skill/tool (paired-bootstrap) path:

- `bootstrap` — no per-example bootstrap CI; the gate is win-loss, not a resampled mean.
- `avg_baseline` / `avg_evolved` — no synthetic holdout mean. The analogous numbers live inside `closed_loop` as `baseline_pass_rate` / `evolved_pass_rate`.
- `dataset` — there is no synthetic eval dataset and no `dataset` block with per-source/per-category counts; the behavioral suite is the JSONL passed via `--tasks`. `run_inputs` records the run config (models, seed, iterations, holdout-ratio, `eval_source: "closed_loop"`), not the suite path or sha.
- `knee_point` — Pareto knee-point selection over a synthetic valset doesn't apply; candidates are chosen on behavioral score.

#### Saturation-denied variant (prompt section)

When the saturation pre-flight default-denies (non-healthy band, non-interactive context, no `--force-saturation-check`), the prompt-section gate writes `decision: "denied"` and carries a `saturation_band` field naming the band that triggered the denial:

```json
{
  "schema_version": "5",
  "artifact_type": "prompt_section",
  "target_section": "MEMORY_GUIDANCE",
  "decision": "denied",
  "decision_signal": "closed_loop",
  "saturation_band": "no_headroom",              // "healthy" never lands here; one of no_headroom | weak_signal | uniform_failure
  "baseline_chars": 1840,
  "run_inputs": { /* ... */ },
  "pr_created": { "status": "skipped", "reason": "prompt_section_pr_unsupported", "branch": null, "commit_sha": null, "url": null }
}
```

`saturation_band` appears only on the `"denied"` decision (it records why the run never started); it is absent on `deploy` / `reject` / `dry_run`.

## metrics.json (deploy-only summary)

Written to `output/<skill>/<timestamp>/metrics.json` only on deploy. Top-level summary for quick scanning:

```json
{
  "skill_name": "obsidian",
  "timestamp": "20260428_165005",
  "iterations": 10,
  "optimizer_model": "openai/gpt-4o-mini",
  "eval_model": "openai/gpt-4o-mini",
  "baseline_score": 0.93,
  "evolved_score": 0.9083333333333333,
  "improvement": -0.021666666666666723,
  "baseline_size": 1172,
  "evolved_size": 438,
  "train_examples": 9,
  "val_examples": 4,
  "holdout_examples": 6,
  "elapsed_seconds": 80.44,
  "constraints_passed": true
}
```

Note: `gate_decision.json` is the source of truth for the deploy decision and contains far richer detail — `metrics.json` is a convenience summary for scanning runs.

## SKILL.md format

The framework expects `SKILL.md` files to have YAML frontmatter delimited by `---` markers, then a markdown body:

```markdown
---
name: github-code-review
description: Review GitHub pull requests with structured feedback
version: 1.0
---

# GitHub Code Review

You are reviewing a pull request. For each changed file:

1. Identify the intent of the change
2. Check correctness against the file's existing patterns
3. ...
```

`load_skill()` parses the frontmatter into `name` + `description` strings (other fields are preserved verbatim in the `frontmatter` string but not parsed). `reassemble_skill()` rejoins frontmatter + evolved body. Skills missing frontmatter, `name:`, or `description:` fail the `skill_structure` constraint.

## paired_bootstrap return dict (`evolution/core/stats.py`)

```python
{
    "mean":         float,    # sample mean of (evolved - baseline) per-example diffs
    "lower_bound":  float,    # (1-confidence)/2 percentile of bootstrap means
    "upper_bound":  float,    # (1+confidence)/2 percentile of bootstrap means
    "n_examples":   int,      # len(baseline_scores)
    "n_resamples":  int,      # bootstrap iterations (default 2000)
    "confidence":   float,    # two-sided confidence level (default 0.90)
}
```

This dict is consumed verbatim by `validate_growth_with_quality()` and serialized into `gate_decision.json` under `bootstrap`. Calibration scripts depend on these key names.

## Tier-4/5 artifacts (code repair + sentinel)

The code-repair (Tier 4) and triage-sentinel (Tier 5) paths write their own on-disk artifacts, distinct from the GEPA `gate_decision.json` above. Write locations: `output/code/<tool-stem>/<ts>/` (single-tool repair), `output/code_campaign/<ts>/` (measurement campaign), `output/monitor/<ts>/` (sentinel); the noise/discrimination sidecars sit next to their suite as `<suite>.jsonl.noise.json` / `.discrimination.json`.

### Code-path `gate_decision.json` (`artifact_type == "code"`)

Written by `evolution/code/evolve_code.py`. Unlike the GEPA variants (bootstrap CI, knee-point, synthetic dataset) this gate is **deterministic-test / oracle** based — no judge, no bootstrap.

```json
{
  "schema_version": "1",
  "artifact_type": "code",
  "decision_signal": "deterministic_test",
  "decision": "deploy",
  "reason": "visible+held-out pass, surface frozen, regression floor green",
  "target_tool": "tools/foo.py",
  "visible_test": "tests/tools/test_foo_a.py",
  "holdout_test": "tests/tools/test_foo_b.py",
  "repair":  {"fixed": true, "fixed_round": 1, "rounds_used": 1},
  "guards": {
    "repair_passed_visible": true,
    "freeze_ok": true, "freeze_violations": [],
    "changed_files": ["tools/foo.py"], "file_scope_ok": true,
    "holdout": {"passed": true, "exit_code": 0, "duration_seconds": 0.35},
    "floor": {"ran": ["tests/tools"], "new_failures": [], "base_failure_count": 0,
              "repaired_failure_count": 0, "duration_seconds": 1.2, "is_full_suite": false}
  },
  "run_inputs": {},
  "full_suite": {}
}
```

`decision_signal` is `deterministic_test` (held-out-split model) or `oracle_match` (campaign/measurement model). `holdout` / `holdout_test` / `visible_test` belong to the held-out gate; the **oracle** variant instead carries `test` (the full fix-commit test file), `bug_tests` (list), `oracle_failure_count`, `guards.bug_tests_passed`, and a `guards.oracle_match` block (`new_vs_oracle`, `oracle_failure_count`) — there `guards.floor` is often `null` because oracle-match over the full file *is* the regression check. `full_suite` is the optional `--benchmark-cmd` block (and may add `downgraded_from: "deploy"` if it demoted a pass).

### `repair_trace.json`

Per-round repair record for human review (`evolution/code/trace.py`). No per-hunk attribution.

```json
{
  "tool": "tools/foo.py",
  "visible_test": "tests/tools/test_foo_a.py",
  "holdout_test": "tests/tools/test_foo_b.py",
  "fixed": true, "fixed_round": 1, "rounds_used": 1,
  "rounds": [
    {"round": 1, "proposed": true, "freeze_violations": [], "test_passed": true, "output_tail": "PASSED"}
  ],
  "final_diff": "--- live_baseline\n+++ deployed\n@@ ... @@\n-    return 0\n+    return a + b"
}
```

### `campaign_ledger.jsonl` + `campaign_report.json`

The campaign ledger is append-only/resumable; one line per organism (or skip reason). `deploy_reachable` is a majority of seeds.

```jsonl
{"status": "organism", "tool": "tools/approval.py", "fix_sha": "7f1b2b45…", "seeds": [true, false, false], "deploy_reachable": false}
{"fix_sha": "934fbe3c…", "tool": "tools/ansi_strip.py", "status": "source_missing"}
```

Skip `status` ∈ `source_missing` / `too_large` / `worktree_failed` / `not_valid`. `campaign_report.json` reduces the ledger to cluster-honest, **organism-level** estimands (`campaign_report.py`):

```json
{
  "n_organisms": 20,
  "deploy_reachable": {"k": 12, "n": 20, "fraction": 0.60,
                        "wilson": [0.387, 0.781],
                        "cluster_bootstrap": {"mean": 0.602, "ci_low": 0.40, "ci_high": 0.80, "p_below_kill": 0.0}},
  "icc": 0.326, "design_effect": 1.65, "effective_n": 36.3,
  "pooled_per_seed_rate_FOR_CONTRAST": {"k": 41, "n": 60, "rate": 0.683, "wilson_DISHONEST": [0.558, 0.787]},
  "kill_line": 0.10, "verdict": "GREEN", "aborted_on_cost": false,
  "cost_summary": {}
}
```

The `_FOR_CONTRAST` / `_DISHONEST` key suffixes are intentional — the pooled per-seed rate ignores seed correlation (ICC) and overstates precision, so it is recorded only for contrast, never as the headline. `cost_summary` has the same `total_usd` + `by_model{…}` shape as `metrics.json`. (Real instances: `reports/asymmetry_campaign_report.json` and `_n46.json`.)

### `triage_queue.json` + `triage_report.md`

Sentinel scan output (`evolution/monitor/queue.py`); `--attempt-top` annotates rows in place via `attempt.py`.

```json
{
  "schema_version": "1", "repo": "/path/to/repo", "since_days": 90,
  "n_candidates": 2, "by_kind": {"dependency_regression": 1, "bug_fix": 1},
  "candidates": [
    {"rank": 1, "kind": "dependency_regression", "tool": "tools/foo.py", "test": "tests/tools/test_foo.py",
     "fix_sha": "7f1b2b45…", "parent_sha": "6855d177…", "committed_at": "2026-05-23T02:59:13-07:00", "score": 2.0,
     "attempt": {"status": "attempted", "correct_seeds": 2, "seeds": 3, "deploy_reachable": true}}
  ],
  "cost_summary": {}
}
```

`attempt.status` ∈ `attempted` / `cost_ceiling` / `source_missing` / `too_large` / `worktree_failed` / `not_valid` (the `correct_seeds`/`seeds`/`deploy_reachable` fields appear only for `attempted`); `cost_summary` is present only when `--attempt-top` ran. `triage_report.md` renders a ranked markdown table (`#`, `kind`, `tool`, short `fix` sha, `committed` date) plus the propose-only disclaimer and the ready-to-run attempt command.

### `lineage.json` + `dossier.md` (GEPA runs)

`lineage.json` (`evolution/core/lineage.py`) persists a GEPA run's candidate ancestry so the deployed diff is reviewable; absent on the MIPROv2 fallback (no `parents`).

```json
{
  "schema_version": "1", "deployed_idx": 2, "best_idx": 1, "n_candidates": 3,
  "seed_text": "…", "live_baseline_text": "…",
  "selection": {"method": "knee_point"}, "suite_sha256": "…",
  "candidates": [
    {"idx": 2, "parents": [0], "val_aggregate": 0.65, "val_subscores": [0.7, 0.6],
     "discovery_eval_count": 8, "text": "…", "is_best": false, "is_deployed": true}
  ]
}
```

`deployed_idx` is explicit because the knee-point selector may pick a candidate other than GEPA's `best_idx`; `seed_text` vs `live_baseline_text` separates pre-GEPA drift from search changes. `dossier.md` (`evolution/core/dossier.py`) renders that lineage as a maintainer-local review: selection rationale (val_aggregate vs seed, candidate position, discovery count, lineage depth), an optional pre-GEPA-drift diff (when `seed_text != live_baseline_text`), and the live-baseline → deployed diff. Local artifact only — never a PR body, no per-hunk attribution.

### Suite sidecars: `<suite>.jsonl.noise.json` / `.discrimination.json`

Written next to a suite by `evolution/validation/noise_calibration.py` and `suite_discrimination.py`.

```json
// <suite>.jsonl.noise.json — A/A noise floor
{"spurious_strict_win_rate": 0.0, "spurious_regression_rate": 0.25, "mean_per_task_flip": 0.15,
 "per_task_flip": {"task_a": 0.2}, "runs": 4, "reps": 1, "suite_sha256": "…", "agent_model": "gpt-5-mini",
 "aborted": false, "n_scored": 8, "n_abstained": 0, "scored_fraction": 1.0, "is_degenerate": false}

// <suite>.jsonl.discrimination.json — per-task discrimination labels
{"labels": {"task_1": "discriminative"}, "baseline_rates": {"task_1": 0.3}, "ceiling_rates": {"task_1": 0.8},
 "flips": {"task_1": 0.05}, "summary": {"discriminative": 1}, "reps": 2, "suite_sha256": "…",
 "agent_model": "sonnet", "recommendation": "…",
 "per_task": {"task_1": {"baseline_rate": 0.3, "ceiling_rate": 0.8, "flip": 0.05, "label": "discriminative"}}}
```

`is_degenerate` is `true` when `scored_fraction < 0.5` — an all-abstain probe measures nothing and must not be read as a perfectly-stable suite. Discrimination labels: `too_easy` / `discriminative` / `unfillable` / `noise_limited` / `baseline_fails`.
