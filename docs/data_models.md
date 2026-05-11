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

    # Benchmarks
    run_pytest: bool = True
    run_tblite: bool = False                # opt-in (expensive)
    tblite_regression_threshold: float = 0.02

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
```

Validated at load by `ToolEntry.from_dict()`: `name` must match `^[a-zA-Z0-9_-]{1,128}$`. Names outside this set break sentinel parsing (regex metacharacters, embedded `-->`) and are rejected with a clear error. `from_dict` reads `inputSchema` (MCP shape) into `input_schema`.

Frozen at the dataclass level — the attribute itself can't be rebound — but `input_schema` is not deep-frozen. Mutating it in place corrupts any other `ToolEntry` / `ToolManifest` that shares the reference (by design: `ToolManifest.replace_description` preserves the original reference).

## ToolManifest (`evolution/tools/tool_source.py`)

```python
@dataclass(frozen=True)
class ToolManifest:
    tools: tuple[ToolEntry, ...]
    confusable_neighbors: dict[str, str] = field(default_factory=dict)
```

Helpers:
- `from_json_file(path) -> ToolManifest` — reads an MCP `list_tools()`-shape file. `_evolution_metadata.confusable_neighbors` is optional metadata for cross-tool regression evaluation.
- `find_tool(name) -> ToolEntry` — raises `KeyError` listing available tools on miss.
- `confusable_neighbor_for(name) -> str | None`.
- `replace_description(name, new_description) -> ToolManifest` — returns a new manifest with the named tool's description swapped; every other tool's `description` and `input_schema` are preserved by reference.

Rejects at load: an empty `tools` tuple, and normalization collisions (`read-file` vs `read_file`, which both lowercase + underscore-normalize to `read_file`). Sentinel matching uses original casing but lookup robustness relies on normalization being injective.

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

## gate_decision.json (schema_version "4")

The structured deploy-gate decision, written to `output/<skill>/<timestamp>/gate_decision.json` on every run regardless of outcome. The schema is the **calibration substrate** — `tests/skills/test_evolve_skill_validation_flow.py:TestGrowthGateDecisionSchema` locks the field list so future calibration scripts (`jq -s '...' output/*/*/gate_decision.json`) don't break.

### Static-failure variant

Written when any `validate_static` check fails on the evolved artifact (short-circuits before holdout):

```json
{
  "schema_version": "4",
  "decision": "reject",
  "reason": "static_constraint_failure",
  "failed_constraints": ["non_empty"],
  "messages": ["Artifact is empty"],
  "knee_point": { "applied": false, "reason": "no_detailed_results" },
  "dataset": { "size_total": 60, "size_train": 21, "size_val": 17, "size_holdout": 22, "sources": {"synthetic": 60} }
}
```

### Growth-quality-gate variant (deploy or reject)

```json
{
  "schema_version": "4",
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

`dataset.categories` (`{"target_correct": N, "confusable_neighbor": N, "regression_detection": N}`) is populated on tool runs; on skill runs it's `{"general": N}`.

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
