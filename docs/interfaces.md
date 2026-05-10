# Interfaces

The public APIs, CLIs, and data contracts the codebase exposes — both to users and to the integration points it depends on.

## CLI: `python -m evolution.skills.evolve_skill`

The primary user-facing interface.

### Required flags
| Flag | Purpose |
|---|---|
| `--skill <name>` | Skill name to evolve (resolved via `SkillSource` walk). |

### Optimizer / iteration
| Flag | Default | Notes |
|---|---|---|
| `--budget {light,medium,heavy}` | `light` (via `--iterations`) | GEPA budget. **Prefer this over `--iterations`.** |
| `--iterations <int>` | `10` | DEPRECATED. Maps `1→light`, `2→medium`, `3→heavy`; anything else collapses to `light`. |
| `--no-fallback` | off | Re-raise GEPA exceptions instead of falling back to MIPROv2. Debug only. |
| `--seed <int>` | `42` | RNG seed for dataset shuffles + DSPy optimizer. |

### Models
| Flag | Default | Notes |
|---|---|---|
| `--optimizer-model <name>` | `openai/gpt-4.1` | Default LM bound to `dspy.configure` (eval LM). |
| `--reflection-model <name>` | `openai/gpt-5-mini` | Drives the GEPA instruction proposer. Reasoning models require `max_tokens >= 16000` (we set 32000). |
| `--eval-model <name>` | `openai/gpt-4.1-mini` | Judge model for scoring + dataset gen. |

### Dataset
| Flag | Default | Notes |
|---|---|---|
| `--eval-source {synthetic,golden,sessiondb}` | `synthetic` | Where eval examples come from. |
| `--dataset-path <dir>` | — | Required for `golden`; optional override for `sessiondb` output dir. |
| `--skill-source-dir <path>` | — | Repeatable. Adds a `LocalDirSkillSource` ahead of auto-discovered sources. |
| `--eval-dataset-size <int>` | `EvolutionConfig.eval_dataset_size` (150) | Total examples generated; train/val/holdout splits derived via the configured ratios. The synthetic generator treats this as a soft target — actual count may be lower for small skills. |
| `--holdout-ratio <float>` | `EvolutionConfig.holdout_ratio` (0.50) | Fraction of the dataset reserved for the holdout split (after train/val are taken). Higher → tighter bootstrap CI, smaller train+val. |

### Quality gate
| Flag | Default | Notes |
|---|---|---|
| `--quality-gate {strict,default,lenient,off,non-inferiority}` | `default` | Preset bundling free threshold + slope + abs ceiling + gate mode. `non-inferiority` ships variants statistically not-worse-than-baseline by more than `--inferiority-tolerance` (recommended for compression runs). `off` is misnamed — it disables the slope/ceiling but still enforces `mean >= 0`; emits a warning and recommends `non-inferiority` instead. |
| `--inferiority-tolerance <float>` | (preset; `0.02` for `non-inferiority`) | Tolerance for the non-inferiority gate: pass when `bootstrap.lower_bound > -tolerance`. Only meaningful with `--quality-gate non-inferiority`. |
| `--growth-free-threshold <float>` | (preset) | Override growth % below which no improvement justification required. |
| `--growth-quality-slope <float>` | (preset) | Override linear coefficient on required improvement. |
| `--max-absolute-chars <int>` | (preset) | Override absolute char ceiling. |
| `--bootstrap-confidence <float>` | `0.90` | Two-sided CI confidence for the holdout improvement bootstrap. |
| `--bootstrap-resamples <int>` | `2000` | Bootstrap iterations. |
| `--knee-point-epsilon <float>` | `1/n_val` | ε for knee-point Pareto band. Override only with calibrated reason. |
| `--knee-point-strategy {val-best,smallest}` | `val-best` | Within the ε-band, which candidate to pick. `val-best` (default): highest val score wins, smallest body as tiebreak. `smallest`: greedy parsimony — picks the smallest body in the band regardless of val cost; available for users explicitly chasing compression. |
| `--fitness-profile {balanced,compression,growth}` | `balanced` | Composite fitness weighting profile for the LLM judge. `balanced` (0.5/0.3/0.2 for correctness/procedure/conciseness) is general-purpose. `compression` (0.4/0.2/0.4) upweights conciseness for shrink-direction work. `growth` (0.6/0.4/0.0) drops conciseness so the optimizer doesn't punish necessary additions. Also selects the `BudgetAwareProposer` template: `compression` → compression-mode (cut redundancy under a tight budget), `balanced` → balanced-mode (direction-agnostic, soft ±20% target), `growth` → growth-mode (add only what feedback identifies as missing). Both the profile and the resolved proposer mode are recorded in `gate_decision.json`. |

### Proposer
| Flag | Default | Notes |
|---|---|---|
| `--bap-max-growth <float>` | `EvolutionConfig.bap_max_growth` (0.20) | `BudgetAwareProposer`'s prompt target for the reflection LM — the growth fraction the proposer asks the LM to aim for. Decoupled from `--growth-free-threshold` so the gate parameter and the proposer's prompt can be tuned independently. A user-supplied `0.0` is preserved as "no headroom" (the proposer floors at zero). |
| `--bap-safety-margin <float>` | `0.10` | Cushion subtracted from `max_growth` to absorb the reflection LM's overshoot tendency (~+8-9% observed empirically). Lower this (e.g. `0.0`) when explicitly calibrating against the gate's bar. |

### Delivery
| Flag | Default | Notes |
|---|---|---|
| `--apply` | off | On a deploy decision, copy `evolved_skill.md` over the source `SKILL.md` in place. No git operations — leaves workflow to the user. No-op (with warning) when the skill source is read-only (Claude Code plugin cache under `~/.claude/plugins/cache`). |
| `--patch` | off | On a deploy decision, emit a unified diff of (baseline → evolved) to stdout, labelled with the source path. Pipe to `patch`, `git apply`, or a code-review tool. |

Both delivery flags are no-ops on a reject decision and emit a one-line stderr notice in that case. Both default off; they only fire when the user opts in.

### Misc
| Flag | Default | Notes |
|---|---|---|
| `--run-tests` | off | Run target repo's pytest suite as a constraint gate (not used by default). |
| `--dry-run` | off | Validate setup; don't run optimization. |
| `--evaluate-band-on-holdout / --no-evaluate-band-on-holdout` | off | Calibration telemetry: after the picked candidate is selected, re-evaluate every candidate in the knee-point band on the holdout and write `band_holdout.json` alongside `gate_decision.json`. Adds judge calls proportional to band size × holdout examples (subsampled to ≤100). Off by default to keep production runs cheap. |

### Exit conditions
- `sys.exit(1)` if skill not found across all `SkillSource`s — prints available skills per source.
- `sys.exit(1)` if `eval_source` requires `--dataset-path` but none provided.
- `sys.exit(1)` if `sessiondb` finds no relevant examples.
- `sys.exit(1)` if holdout split has fewer than `min_holdout_size` (default 10) examples.
- Returns normally (rejection path) if static or growth-quality gate fails — `evolved_FAILED.md` + `gate_decision.json` are written.

## CLI: `python -m evolution.core.external_importers`

Standalone session-history importer. Useful for previewing what `--eval-source sessiondb` would produce without running the full evolution.

| Flag | Default | Notes |
|---|---|---|
| `--source {claude-code,copilot,hermes,all}` | `all` | Which session source(s) to mine. |
| `--skill <name>` | required | Target skill name. |
| `--output <dir>` | `datasets/skills/<skill>/` | Where to write `train/val/holdout.jsonl`. |
| `--model <name>` | `openrouter/google/gemini-2.5-flash` | LiteLLM model for relevance scoring. |
| `--max-examples <int>` | `50` | Cap on generated eval examples. |
| `--dry-run` | off | Show source counts without LLM scoring. |

**Note:** the standalone CLI uses `_load_skill_text(skill_name)` which expects skills under `~/.hermes/skills/`. The `evolve_skill.py` `--eval-source sessiondb` path uses the same `build_dataset_from_external` orchestration but resolves the skill via `SkillSource` instead.

## Python API: `evolve()`

```python
from evolution.skills.evolve_skill import evolve

evolve(
    skill_name="github-code-review",
    iterations=10,
    eval_source="synthetic",          # synthetic | golden | sessiondb
    dataset_path=None,
    optimizer_model="openai/gpt-4.1",
    eval_model="openai/gpt-4.1-mini",
    reflection_model="openai/gpt-5-mini",
    skill_source_dirs=None,           # list[str]
    run_tests=False,
    dry_run=False,
    seed=42,
    budget=None,                      # "light" | "medium" | "heavy"
    no_fallback=False,
    quality_gate="default",           # "strict" | "default" | "lenient" | "off" | "non-inferiority"
    growth_free_threshold=None,
    growth_quality_slope=None,
    max_absolute_chars=None,
    inferiority_tolerance=None,       # float, only meaningful with quality_gate="non-inferiority"
    bootstrap_confidence=None,
    bootstrap_n_resamples=None,
    knee_point_epsilon=None,
    knee_point_strategy="val-best",   # "val-best" | "smallest"
    bap_safety_margin=None,           # None falls back to BAP's 0.10 default
    bap_max_growth=None,              # None falls back to EvolutionConfig.bap_max_growth (0.20)
    eval_dataset_size=None,           # None falls back to EvolutionConfig.eval_dataset_size (150)
    holdout_ratio=None,               # None falls back to EvolutionConfig.holdout_ratio (0.50)
    evaluate_band_on_holdout=False,
    fitness_profile="balanced",       # "balanced" | "compression" | "growth"
    apply_in_place=False,             # --apply: copy evolved over source SKILL.md on deploy
    emit_patch=False,                 # --patch: emit unified diff to stdout on deploy
)
```

Returns `None`. All side effects go to `output/<skill>/<timestamp>/`. Failures are surfaced via `sys.exit(1)` (printed banner) or via the rejection-path artifacts (`evolved_FAILED.md` + `gate_decision.json`).

## SkillSource Protocol

```python
from typing import Protocol, runtime_checkable
from pathlib import Path

@runtime_checkable
class SkillSource(Protocol):
    name: str
    def find_skill(self, skill_name: str) -> Path | None: ...
    def list_skills(self) -> list[str]: ...
```

Implementations live in `evolution/core/skill_sources.py`. To plug in a new agent framework:

1. Create a class that satisfies the protocol (set `name`, implement both methods).
2. Either pass it explicitly into `EvolutionConfig.skill_sources`, or extend `discover_skill_sources()` to sniff for it.

## Output artifacts

Per-run directory: `output/<skill_name>/<YYYYMMDD_HHMMSS>/`. Contents vary by outcome:

| File | When | Contents |
|---|---|---|
| `run.log` | always | All `INFO`+ logs from the run, including `LMTimingCallback` start/end + heartbeats + litellm retries. |
| `gate_decision.json` | always (deploy + reject paths) | Structured decision payload. See [data_models.md](data_models.md). |
| `evolved_skill.md` | deploy only | Full reassembled SKILL.md with new body + original frontmatter. |
| `baseline_skill.md` | deploy only | Baseline SKILL.md verbatim (for diffing). |
| `metrics.json` | deploy only | Top-level run metrics (skill name, scores, sizes, timing). |
| `evolved_FAILED.md` | reject only | The proposed body that failed; saved for post-hoc inspection. |
| `gepa_failure.log` | only on GEPA→MIPROv2 fallback | Path is `output/<skill>/gepa_failure.log` (not per-timestamp). Contains GEPA exception + traceback. |

## DSPy integration points

- **`dspy.configure(lm=..., callbacks=[LMTimingCallback()], warn_on_type_mismatch=False)`** — done once in `evolve()`. The callback gives end-to-end LM observability; `warn_on_type_mismatch=False` silences spam from signatures that pass empty/None into `str` inputs.
- **`dspy.LM(model, ..., request_timeout, num_retries)`** — `request_timeout` and `num_retries` are forwarded to litellm's tenacity layer. Three different timeout regimes:
  - judge LM (`fitness.py`): `request_timeout=60, num_retries=5`
  - dataset gen LM (`dataset_builder.py`): `request_timeout=120, num_retries=5`
  - reflection LM (`evolve_skill.py:_default_gepa_runner`): `request_timeout=300, num_retries=2` (lower retries to fast-fail and trigger MIPROv2 fallback)
- **`dspy.GEPA(metric, auto, reflection_lm, seed, track_stats=True, instruction_proposer=...)`** — `track_stats=True` is required for knee-point selection; `instruction_proposer=BudgetAwareProposer(...)` is required for the char budget.
- **`dspy.MIPROv2(metric, auto="light", init_temperature=0.5, seed)`** — fallback only. Requires the `[miprov2]` extra (lazy `optuna` import).
- **`dspy.Evaluate(devset, metric, num_threads=4, provide_traceback=True, max_errors=...)`** — used for holdout evaluation. Returns `EvaluationResult(score=mean*100, results=[(ex, pred, score), ...])`.

## litellm integration points

- **`litellm.failure_callback`** — module-level list. `register_litellm_failure_callback()` appends `_log_litellm_failure` if not already present. Idempotent + lock-guarded.
- The callback signature is litellm's documented `(kwargs, exception, start_time, end_time)` shape. Logged at `WARNING`.

## Test surfaces locked by tests (don't break without versioning)

These are technically internal but tested directly because downstream calibration scripts depend on them:

- `_write_gate_decision(output_dir, payload) -> Path` — keep filename `gate_decision.json`.
- `gate_decision.json` schema fields — `tests/skills/test_evolve_skill_validation_flow.py:TestGrowthGateDecisionSchema` and `TestStaticValidationShortCircuitsBeforeHoldout` lock `schema_version="4"` plus the full key list. See [data_models.md](data_models.md).
- `_dataset_payload(dataset)` — `size_total`, `size_train`, `size_val`, `size_holdout`, `sources` (per-source counter; "unknown" bucket for `source=""`). Locked by `TestDatasetPayloadHelper`.
- `_knee_point_payload(pick)` — applied/skipped shapes both locked by `TestKneePointPayloadHelper`.
- `paired_bootstrap()` return shape — `mean`, `lower_bound`, `upper_bound`, `n_examples`, `n_resamples`, `confidence`. Calibration scripts depend on these key names.
- `SyntheticDatasetBuilder` LM construction — `tests/core/test_dataset_builder.py:TestSyntheticGeneratorLMConfig` asserts `max_tokens=16000` (regression guard against the JSON-truncation bug at `eval_dataset_size=60`).

## Environment variables

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required by litellm for the OpenAI models in the defaults. |
| `SKILL_SOURCES_HERMES_REPO` | Points `HermesSkillSource` at a custom repo location. Falls back to `~/.hermes/hermes-agent` then a sibling `hermes-agent/` checkout. |

`HERMES_AGENT_REPO` (without the `SKILL_SOURCES_` prefix) is a legacy alias seen in older shell snippets — only `SKILL_SOURCES_HERMES_REPO` is read by current code.

## Logging conventions

`evolution/skills/evolve_skill.py:30` calls `logging.basicConfig(level=INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")` at module import. Format is matched by the per-run `FileHandler`.

Logger names follow the module path: `evolution.skills.budget_aware_proposer`, `evolution.core.lm_timing_callback`, etc. Filter on these to isolate signal in `run.log`.
