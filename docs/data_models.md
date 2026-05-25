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
```

`fixture_setup` is a `relative_path → file_content` map materialized into the task's per-task tmp dir before the agent runs. `user_message.format(fixture_dir=...)` substitutes the placeholder.

Scoring rule (`score_task` in `report.py`):
- Returns `(passed: bool, abstained: bool)`.
- Abstention if the runner errored (timeout, no session JSON, parse failure) — neither evidence for nor against the artifact.
- Else: passes iff `expected_tools` were invoked AND no `forbidden_tools` were invoked. Empty `expected_tools` short-circuits to true; empty `forbidden_tools` is no-op.

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
