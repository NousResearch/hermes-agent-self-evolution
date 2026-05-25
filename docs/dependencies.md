# Dependencies

External packages the framework depends on, what each is used for, and how they're constrained.

Last verified against `pyproject.toml` + `uv.lock` on 2026-05-25.

## Hard runtime dependencies

### `dspy>=3.2.0,<3.3`

The optimization engine. DSPy 3.2.0 is the floor; in practice the lock pins it at 3.2.0 because that's the version whose `gepa[dspy]==0.0.27` transitive constraint is what forced the gepa override below. Used pervasively:

| Module | Usage |
|---|---|
| `dspy.Module` | Base class for `SkillModule` (`evolution/skills/skill_module.py`). |
| `dspy.Signature`, `dspy.InputField`, `dspy.OutputField` | Declarative LM signatures (judge, dataset gen, relevance scoring, budget-aware proposer). |
| `dspy.ChainOfThought`, `dspy.Predict` | Wrapped predictors. |
| `dspy.Example` | Train/val/holdout examples (via `EvalDataset.to_dspy_examples`). |
| `dspy.Prediction` | Metric return type (`score`, `feedback`). |
| `dspy.LM` | All LM calls. `request_timeout` and `num_retries` kwargs are forwarded to litellm. |
| `dspy.configure` | Sets the global LM + callbacks. |
| `dspy.context(lm=...)` | Per-block LM override (used in dataset gen, judge, relevance scoring). |
| `dspy.GEPA` | The reflective optimizer. `track_stats=True` for `detailed_results`; `instruction_proposer=` for `BudgetAwareProposer`. |
| `dspy.MIPROv2` | Fallback optimizer (lazy `optuna` import). |
| `dspy.Evaluate` | Holdout scoring with `num_threads=4`. |
| `dspy.utils.callback.BaseCallback` | Subclassed by `LMTimingCallback`. **Internal-ish API.** |

**Why pinned to `<3.3`:** `BaseCallback` lives at `dspy.utils.callback.BaseCallback` and is not in `dspy.__all__`. A 3.3 minor bump could move or rename it. Same for the `DspyAdapter` / `propose_new_texts` interaction in `gepa/api.py:317-321` that forces our use of `instruction_proposer` instead of `reflection_prompt_template`.

### `gepa @ git+https://github.com/gepa-ai/gepa.git@5e24ee5c8e1857a62a1ba19731de9da45ffb6f1b`

GEPA itself is pinned to an exact git SHA — the merge commit for the upstream `acceptance_criterion` API. The framework's `--gepa-acceptance` flag forwards `acceptance_criterion="improvement_or_equal"` to `dspy.GEPA(..., gepa_kwargs={...})`; the latest released `gepa==0.1.1` on PyPI predates that merge and doesn't accept the kwarg.

**Why a git pin instead of a version range:** the version that ships with the merged API (`0.1.2`) hadn't been published at the time the framework adopted it. When `0.1.2` (or newer) lands on PyPI, swap the dep to `gepa>=0.1.2,<0.2` and drop the `[tool.uv]` override below.

**`[tool.uv] override-dependencies`:** DSPy 3.2.0 hard-pins `gepa[dspy]==0.0.27` as a transitive constraint. Without the override, `uv sync` would resolve to that pre-`acceptance_criterion` build and the `improvement-or-equal` default would error at GEPA construction time. The override block forces every resolver path — direct dep, transitive via `dspy[optuna]`, transitive via anything else — onto the same git SHA:

```toml
[tool.uv]
override-dependencies = [
    "gepa @ git+https://github.com/gepa-ai/gepa.git@5e24ee5c8e1857a62a1ba19731de9da45ffb6f1b",
]
```

This is uv-specific (PEP 735 doesn't standardize overrides). Users installing with pip pick up the direct git dep but not the override — if they also install `dspy[optuna]`, pip's resolver will report a conflict. `uv sync` is the supported install path.

### `litellm>=1.82.0,<2.0`

The provider-agnostic LLM client that DSPy wraps. Used directly only in `evolution/core/lm_timing_callback.py`:

- `litellm.failure_callback` — module-level list mutation (not a documented public API). The framework registers `_log_litellm_failure` here so per-attempt retries are visible (DSPy's `BaseCallback.on_lm_end` only fires once per logical call).

**Why pinned:** `failure_callback` is stable at 1.82 but not marked public. A future minor bump could move or change its protocol. The pin protects observability without rolling our own retry layer.

### `openai>=1.0.0`

The OpenAI SDK that litellm uses for `openai/*` model strings. Default models in `EvolutionConfig` are all OpenAI:
- `optimizer_model` / `judge_model` defaults: `openai/gpt-4.1`
- `eval_model` default: `openai/gpt-4.1-mini`
- `reflection_model` CLI default: `openai/gpt-5-mini`

Note: OpenAI's `_base_client` adds its own retry layer underneath litellm's tenacity layer. Visible as `openai._base_client: Retrying request` in `run.log`. This means `request_timeout × num_retries` is a bound, not a hard ceiling — observed worst case has been ~910s vs claimed 600s.

Models can be swapped to any provider litellm supports (Anthropic, OpenRouter, Together, etc.) via `--optimizer-model anthropic/claude-3-5-sonnet-20241022` etc. — no code changes required.

### `click>=8.0`

Used for the two CLIs:
- `evolution/skills/evolve_skill.py:main` — primary `evolve` CLI.
- `evolution/core/external_importers.py:main` — standalone session importer.

Standard `@click.command()` + `@click.option()` patterns; no plugins or custom types.

### `rich>=13.0`

Terminal UI. Two surfaces:
- `Console` — colored output throughout `evolve_skill.py` (cyan headers, green checks, red rejects).
- `Panel` — fallback banner when GEPA fails.
- `Table` — final results table (baseline / evolved / change).
- `Progress` — progress bars in `external_importers.py` for Copilot session reads + relevance scoring.

### `reportlab>=4.0`

Used only by `generate_report.py` (top-level, not part of `evolution/`) to render `reports/phase1_validation_report.pdf` from `reports/phase1_prose.yaml` plus the per-run `output/<skill>/<ts>/{gate_decision.json,metrics.json,run.log}` artifacts. Not exercised by the optimization pipeline.

### `numpy>=1.24`

Used by `evolution/core/stats.py:paired_bootstrap` for the resample matrix and percentile computation (`np.array`, `np.percentile`, `np.random`). Floor 1.24 predates the 2.0 ABI break and stays compatible with both numpy 1.x and 2.x. No upper bound — only stable APIs are touched.

### `pyyaml>=6.0`

Used only by `generate_report.py` to load `reports/<phase>_prose.yaml` (editorial content for the validation report). Not exercised by the optimization pipeline. Was previously a transitive dep through dspy/litellm; promoted to a direct dep when the report became data-driven.

## Dev dependency group — `pytest>=7.0`, `pytest-asyncio>=0.21`

Declared under `[dependency-groups].dev` (PEP 735), so it stays out of the published package's extras. `uv sync` installs it by default; `uv sync --no-dev` skips it.

`pytest-asyncio` is declared but not currently exercised by the test suite (no `@pytest.mark.asyncio` tests). Reserved for future async work.

## Optional extras

### `[miprov2]` — `dspy[optuna]>=3.2.0,<3.3`

Required only when GEPA fails and the MIPROv2 fallback fires. The import is lazy inside `_default_mipro_runner()` — install on demand:

```bash
uv pip install "agent-self-evolution[miprov2]"
```

If missing, the fallback raises `ImportError` (re-raised with the GEPA failure preserved as `__cause__` so the user sees both).

> Tier 4 (code-evolution) will introduce a `darwinian-evolver` dependency when that tier is built. Not declared in `pyproject.toml` until the package is published and `evolution/code/` actually imports it.

## Implicit dependencies (not in pyproject.toml)

### `threading` (stdlib)

`evolution/core/lm_timing_callback.py` uses `threading.Timer` (heartbeats) and `threading.Lock` (in-flight bookkeeping + idempotent litellm hook registration). The `timer_factory` constructor kwarg lets tests inject a `FakeTimer` that doesn't actually sleep.

### `pathlib`, `dataclasses`, `typing` (stdlib)

Used pervasively. `typing.Protocol` + `runtime_checkable` for `SkillSource`.

## External services

### OpenAI API

The default LM provider. Three call patterns with different cost profiles:

| Caller | Model (default) | Volume per run | Notes |
|---|---|---|---|
| Dataset gen | `openai/gpt-4.1` | 1 call (60-example JSON output, 16K max_tokens) | Synthetic mode only |
| GEPA reflection | `openai/gpt-5-mini` | ~iterations × 1-2 calls | Reasoning model; long P99 (103s observed max) |
| GEPA per-iteration scoring | `openai/gpt-4.1-mini` | ~iterations × |trainset| × judge calls | Bulk |
| Holdout eval | `openai/gpt-4.1-mini` | 2 × |holdout| judge calls | Done twice (baseline + evolved) |
| Knee-point per-candidate static | none | 0 LM calls | Pure-Python validation |

Typical run cost (light budget on a small skill): **$0.50 - $2.00**. Heavy budget on a large skill: **$5 - $15**.

### Local filesystem reads (sessiondb path)

| Source | Path | Format |
|---|---|---|
| Claude Code | `~/.claude/history.jsonl` | JSONL, one user message per line |
| Copilot CLI | `~/.copilot/session-state/<id>/{events.jsonl, workspace.yaml}` | JSONL events + YAML metadata |
| Hermes Agent | `~/.hermes/sessions/*.json` | OpenAI-format message list |

All reads are best-effort — missing files / dirs result in empty message lists, not errors.

### Skill source filesystem reads

| Source | Path | Layout |
|---|---|---|
| Hermes | `$SKILL_SOURCES_HERMES_REPO` or `~/.hermes/hermes-agent` | `<root>/skills/<category>/<name>/SKILL.md` |
| Claude Code | `~/.claude/plugins/cache` | `<vendor>/<plugin>/<version>/skills/<name>/SKILL.md` |
| LocalDir | passed via `--skill-source-dir` | `<root>/<name>/SKILL.md` |

Discovery walks all roots that exist; sources with missing roots are silently skipped.

## Build + tooling

### `setuptools>=68`, `wheel`

Build backend declared in `pyproject.toml`. Standard PEP 517 build.

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["evolution*"]
```

### `pytest` config

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

No pytest plugins are required at runtime. The `_skill_source_env` autouse fixture is defined per-test-module rather than centrally — see `tests/core/test_constraints.py:9` for the pattern.

## What this codebase does NOT depend on

Worth noting because they often appear in similar projects:

- **No PyTorch / TensorFlow / JAX.** No GPU training; everything is API calls. `BootstrapFinetune` (the only DSPy component that trains weights) is explicitly excluded by the project plan.
- **No Pydantic.** Pure stdlib `dataclasses` + `typing.Protocol`. DSPy uses Pydantic internally but no Pydantic models are exposed by this codebase.
- **No FastAPI / Flask / web framework.** CLI-only.
- **No database.** All state is files (`output/`, `datasets/`, `~/.claude/history.jsonl`, etc.).
- **No async runtime.** `dspy.Evaluate(num_threads=4)` uses threads, not asyncio. `pytest-asyncio` is declared but unused.
- **No Docker / container scaffolding.** Plain `uv sync` workflow.
- **No CI config in repo.** Tests are run locally via `pytest`.
