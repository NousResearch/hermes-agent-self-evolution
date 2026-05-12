# AGENTS.md

Project-specific context for AI coding assistants. Read this first when picking up the codebase. For deeper material, jump from the [Knowledge Base Index](docs/index.md).

## Table of contents

- [What this project is](#what-this-project-is) — `[meta:project-overview]`
- [Repo layout at a glance](#repo-layout-at-a-glance) — `[meta:dir-structure]`
- [How a run works (5-line summary)](#how-a-run-works-5-line-summary) — `[meta:flow]`
- [What lives where](#what-lives-where) — `[meta:component-map]`
- [Coding conventions](#coding-conventions) — `[meta:style]`
- [Testing](#testing) — `[meta:test-workflow]`
- [Running the framework locally](#running-the-framework-locally) — `[meta:local-dev]`
- [Output artifacts on disk](#output-artifacts-on-disk) — `[meta:on-disk-formats]`
- [Things that look weird but aren't](#things-that-look-weird-but-arent) — `[meta:gotchas]`
- [PR / commit conventions](#pr--commit-conventions) — `[meta:vcs]`
- [What's planned vs. what exists](#whats-planned-vs-what-exists) — `[meta:roadmap]`
- [When to consult which doc](#when-to-consult-which-doc) — `[meta:doc-routing]`

---

## What this project is

`[meta:project-overview]` Cross-ref: [docs/codebase_info.md](docs/codebase_info.md), [docs/architecture.md](docs/architecture.md), [README.md](README.md)

`agent-self-evolution` evolves agent SKILL.md files via DSPy + GEPA (a reflective prompt optimizer). The whole pipeline is API calls — no GPU training, no model weights touched. A SKILL.md body is wrapped as a `dspy.Module`, GEPA mutates the instruction text using execution-trace feedback, candidates are scored by an LLM-as-judge, and the winner has to clear a paired-bootstrap quality gate on a held-out split before being accepted.

Framework-agnostic at the optimizer layer: any agent that emits SKILL.md files (Hermes Agent, Claude Code skills, custom local layouts) is supported via the `SkillSource` Protocol in `evolution/core/skill_sources.py`.

For *why this differs from raw DSPy + GEPA* (the small-N selection layer, the paired-bootstrap deploy gate, the composite judge fitness) see [docs/framework_advantages.md](docs/framework_advantages.md).

Tiers 1 and 2 are implemented: skill files in `evolution/skills/`, tool descriptions in `evolution/tools/` (with `MCPManifestSource` for MCP `list_tools()`-shape JSON and `HermesToolSource` for Python `*_SCHEMA` declarations resolved via AST). Tiers 3-5 (prompt sections, code, continuous loop) exist as empty package stubs. See `PLAN.md` for each phase's "Deviations from plan" subsection.

## Repo layout at a glance

`[meta:dir-structure]`

```
agent-self-evolution/
├── evolution/           # the package (only thing installed into the venv)
│   ├── core/            # framework-agnostic infrastructure
│   ├── skills/          # Tier 1 — skill-file evolution
│   ├── tools/           # Tier 2 — tool-description evolution
│   ├── prompts/         # Tier 3 stub
│   ├── code/            # Tier 4 stub
│   └── monitor/         # Tier 5 stub
├── tests/
│   ├── core/            # mirrors evolution/core/
│   ├── skills/          # mirrors evolution/skills/
│   └── tools/           # mirrors evolution/tools/
├── datasets/
│   ├── skills/<name>/   # train.jsonl, val.jsonl, holdout.jsonl
│   └── tools/           # tool eval data is generated per-run, not cached here
├── output/<skill>/<ts>/ # per-run artifacts (git-ignored)
├── reports/             # validation PDFs + reports/<phase>_prose.yaml
├── assets/              # logo PNGs used by the report (e.g. dna.png)
├── docs/                # the knowledge base — start at docs/index.md
├── generate_report.py   # renderer: --run output/<skill>/<ts>/ --prose reports/<phase>_prose.yaml --out reports/<phase>_validation_report.pdf
├── PLAN.md              # full roadmap
├── README.md            # quick start
└── AGENTS.md            # you are here
```

The `evolution/<tier>/` directories form **a clean layering**: `evolution/core/` has no imports from any tier package. Tier packages may import from core but never from each other.

## How a run works (5-line summary)

`[meta:flow]` Full mermaid: [docs/workflows.md Workflow 1](docs/workflows.md)

1. CLI resolves `--skill <name>` to a `SKILL.md` via the `SkillSource` walk.
2. Eval dataset is built (synthetic LM gen / golden file / sessiondb mining).
3. Skill body wrapped as `dspy.Module`; GEPA optimizes it with `BudgetAwareProposer` injecting a char budget into the reflection prompt.
4. Knee-point Pareto selection walks the candidates within ε of the best valset score in `--knee-point-strategy` order. Default `val-best`: highest val first, smallest body as tiebreak. `smallest` (greedy parsimony) is available via the flag for users explicitly chasing compression.
5. Static constraints + paired-bootstrap growth-quality gate decide deploy vs. reject; both outcomes write `gate_decision.json`. The default rule is `no_regression` (`mean >= 0`); `--quality-gate non-inferiority` switches to `lower_bound > -inferiority_tolerance` (recommended for compression-focused runs at small N where the bootstrap CI swamps tiny effects).

## What lives where

`[meta:component-map]` Cross-ref: [docs/components.md](docs/components.md)

| Concern | File |
|---|---|
| CLI + orchestration | `evolution/skills/evolve_skill.py` |
| `EvolutionConfig` dataclass | `evolution/core/config.py` |
| `SkillSource` Protocol + 3 impls | `evolution/core/skill_sources.py` |
| SKILL.md ↔ DSPy bridge (`SkillModule`) | `evolution/skills/skill_module.py` |
| Char-budget reflection prompt | `evolution/skills/budget_aware_proposer.py` |
| Knee-point Pareto picker | `evolution/skills/knee_point.py` |
| Synthetic dataset gen + golden loader | `evolution/core/dataset_builder.py` |
| Sessiondb mining (Claude Code, Copilot, Hermes) | `evolution/core/external_importers.py` |
| LLM-as-judge + GEPA-shaped metric | `evolution/core/fitness.py` |
| Deploy gate (static + growth-quality) | `evolution/core/constraints.py` |
| Paired-bootstrap CI | `evolution/core/stats.py` |
| LM observability (timing, heartbeats, retries) | `evolution/core/lm_timing_callback.py` |

## Coding conventions

`[meta:style]`

Inferred from the existing source — follow these unless you have a specific reason to deviate.

### Comments
- **Comments should be rare and only when necessary to explain why an otherwise unintuitive decision was made.** If it's just explaining how the code works, that should be left to thoughtful variable naming and function docstrings.
- **Don't reference task IDs, PR numbers, internal change designators, or "in May 2026"-style dates anywhere user-visible** — not in source comments, not in docs, not in commit messages, not in PR bodies. Git log is authoritative for "when" and "why"; describe the *current state* or the *behavior change*, not the PR-numbered journey.
- **No "removed X" or "deprecated Y" placeholder comments.** Just delete.

### Imports
- Stdlib first, third-party second, local last — separated by blank lines.
- `from __future__ import annotations` at module top in newer files (`stats.py`, `skill_sources.py`, `lm_timing_callback.py`, `knee_point.py`, `budget_aware_proposer.py`). Older files don't. Don't add it to existing files just for consistency.

### Types
- Type hints on every function signature and dataclass field.
- `X | None` over `Optional[X]` in newer files; `Optional[X]` is also used (mixed). Don't refactor existing files just to switch styles.
- `typing.Protocol` + `runtime_checkable` for plug-in interfaces (see `SkillSource`).

### Dataclasses
- Plain `@dataclass` for state objects; `@dataclass(frozen=True)` for things that shouldn't mutate after construction (e.g., `CandidatePick`).
- `field(default_factory=...)` for mutable defaults — never bare mutable defaults.

### Logging
- Use module-level `logger = logging.getLogger(__name__)` then `logger.info/warning/...`.
- `evolve_skill.py:30-34` calls `logging.basicConfig(level=INFO)` at module import; importing this module from a notebook **will** configure the root logger — flag if undesirable.
- `LMTimingCallback` covers LM call observability — don't roll your own.

### CLI flags
- Use `click.option` with explicit `default=`, `type=`, `help=`.
- For optional advanced flags (overrides), `default=None` and check for `None` inside the callee. See `--growth-free-threshold` etc.
- Document deprecated flags in `--help` (see `--iterations`).

### Errors
- Validate at boundaries (`paired_bootstrap` raises `ValueError` on shape mismatch).
- `_clamp_to_unit` parses LLM-emitted score strings and falls back to 0.5 (neutral) on malformed input rather than raising — loud failure here would crash a run over a single noisy judge call. **Pattern: at boundaries, fail loudly; downstream of boundaries, degrade gracefully when the failure mode is "noisy LM output."**

### Imports / dep boundaries
- `evolution.core.*` must not import from `evolution.skills.*` or any other tier. The reverse is allowed.
- Tier packages must not import from each other.

## Testing

`[meta:test-workflow]` Cross-ref: [docs/interfaces.md](docs/interfaces.md), [docs/workflows.md Workflow 8](docs/workflows.md)

- Run all tests: `pytest tests/ -q` from the repo root, **inside the venv** (`source .venv/bin/activate`). Currently 282 tests.
- All tests use mocks for LM calls — no API keys required.
- The `_skill_source_env` autouse fixture (defined per-module, e.g., `tests/core/test_constraints.py:9`) sets `SKILL_SOURCES_HERMES_REPO` to a `tmp_path` fake repo so discovery doesn't pick up real `~/.hermes` / `~/.claude` installs. Add this fixture to any new test that touches `EvolutionConfig`.

### What to test
- New helper functions in `evolve_skill.py` go in `tests/skills/test_evolve_skill_helpers.py` (or `_validation_flow.py` if they're part of the post-optimization flow).
- New `gate_decision.json` fields → add a regression test in `TestGrowthGateDecisionSchema` so the calibration substrate doesn't silently drift.
- New constraints → add to `TestSizeConstraints` / `TestNonEmpty` / new test class in `test_constraints.py`. Lock both pass and fail paths.
- New `SkillSource` impl → add to `tests/core/test_skill_sources.py` with a `tmp_path`-based fake layout.

### What not to test
- Don't write tests that mock `dspy.GEPA.compile` end-to-end. Test the helpers around it (`_build_optimizer_and_compile` fallback chain, knee-point on a fake `detailed_results`).
- Don't test by hitting real LMs. Use `MagicMock`/`SimpleNamespace` for `dspy.Predict`, `dspy.Evaluate`, etc. (see `test_evolve_skill_validation_flow.py:TestHoldoutEvaluate` for the pattern).
- Don't add per-PR "changelog test" cases. The schema test locks the keys; that's the contract.

### Test naming
- `tests/<package>/test_<module>.py`.
- Test classes are `TestXxx`. Method names are `test_<behavior>` describing the assertion.

## Running the framework locally

`[meta:local-dev]` Cross-ref: [README.md](README.md), [docs/interfaces.md](docs/interfaces.md)

```bash
# install (creates .venv, installs project + dev group from uv.lock)
uv sync

# fast smoke test (no real LM calls)
uv run python -m evolution.skills.evolve_skill --skill <name> --dry-run

# real run (uses OpenAI by default; need OPENAI_API_KEY)
uv run python -m evolution.skills.evolve_skill \
    --skill <name> \
    --budget light \
    --eval-source synthetic
```

For Hermes Agent skills: `export SKILL_SOURCES_HERMES_REPO=~/.hermes/hermes-agent` (or set to wherever your checkout lives).
For Claude Code skills: nothing needed — `~/.claude/plugins/cache/` is auto-discovered.

Cost rough cuts: light budget on a small skill = $0.50-2.00; heavy budget on a large skill = $5-15.

## Output artifacts on disk

`[meta:on-disk-formats]` Cross-ref: [docs/data_models.md](docs/data_models.md)

Per-run dir: `output/<skill>/<YYYYMMDD_HHMMSS>/`. Contents vary by outcome:

| File | When | Purpose |
|---|---|---|
| `run.log` | always | Every LM call (start, end, heartbeats), every retry |
| `gate_decision.json` | always | Structured deploy decision (schema_version `"4"`) |
| `evolved_skill.md` | deploy only | New SKILL.md ready to ship |
| `baseline_skill.md` | deploy only | Original (for diffing) |
| `metrics.json` | deploy only | Top-level run summary |
| `evolved_FAILED.md` | reject only | The proposed body that failed (for inspection) |

`output/<skill>/gepa_failure.log` is written *only* when GEPA exceptions trigger MIPROv2 fallback. Path is per-skill, not per-timestamp.

`datasets/skills/<skill>/{train,val,holdout}.jsonl` are reused across runs — delete to force regeneration.

## Things that look weird but aren't

`[meta:gotchas]`

- **Empty `evolution/{tools,prompts,code,monitor}/`** — these are stubs anchoring the planned tier 2-5 work. See [docs/codebase_info.md](docs/codebase_info.md) status table.
- **`logging.basicConfig` at module import** — `evolve_skill.py:30-34` configures the root logger when imported. Side effect, intentional for the CLI; surprising if you `from evolution.skills.evolve_skill import evolve` in a notebook.
- **`val_ratio + holdout_ratio + train_ratio = 1.40`** — looks like a bug; isn't. `split_examples()` normalizes the three ratios so they sum to 1; the synthetic, sessiondb, and golden paths all go through the same helper.
- **`max_tokens=16000` on dataset gen LM** — load-bearing. At `eval_dataset_size>=60` the JSON output truncates mid-string with anything lower; the current default `eval_dataset_size=150` makes this even more critical. Locked by `TestSyntheticGeneratorLMConfig`.
- **`eval_dataset_size=150` is the current default**, sized for a ~53-example holdout that's tight enough on the bootstrap CI to detect ±2% effects. Per-skill calibration runs at smaller N are no longer authoritative.
- **Reflection LM `request_timeout=300, num_retries=2`** (vs `=5` for judge) — deliberate fast-fail. A reflection-LM `TimeoutError` triggers MIPROv2 fallback rather than burning more time on a stuck call.
- **Knee-point reads `optimized_module.detailed_results`** — only present when GEPA succeeded (and `track_stats=True`). MIPROv2 fallback path skips knee-point cleanly. `gate_decision.json.knee_point.applied=false` with `reason="no_detailed_results"` is the signal.
- **`SkillModule.TaskWithSkill` docstring is a placeholder** — `__init__` overwrites the signature instructions per-instance via `with_instructions(skill_text)`. Don't rely on the class-level docstring.
- **`reassemble_skill` strips a leading `---` block** — defensive against the reflection LM mimicking YAML frontmatter (would otherwise produce a double-frontmatter file). Logged at WARNING when it fires; see if the prompt needs tightening.
- **Test uses both `~/.hermes/skills/` and `~/.hermes/hermes-agent/skills/`** — `external_importers._load_skill_text` (standalone CLI only) reads the former; `HermesSkillSource` (the optimizer's path) reads the latter. Same prefix, different paths.
- **`generate_report.py` is a renderer, not a content-store.** Numbers come from a run dir's `gate_decision.json` + `metrics.json` + `run.log` (auto-extracted); editorial prose + table contents come from `reports/<phase>_prose.yaml`; the title-page logo is `assets/dna.png` (Twemoji 14.0). Updating the report after a run is `python generate_report.py --run output/<skill>/<new-ts>/` — no script edit needed unless the prose itself changes. Run.log only carries `lm_timing_callback` lines; `dspy.teleprompt.gepa` "Iteration N" lines are not in the per-run artifact (they're stdout-only) so the report's pipeline-step text doesn't reference an iteration count.

## PR / commit conventions

`[meta:vcs]`

- Conventional Commits: `feat:`, `fix:`, `refactor:`, `chore:`, `docs:` (no scope). Example: `feat: knee-point Pareto selection on GEPA candidate front (#8)`.
- Commits include the PR number at the end of the subject line on merge — `git log --oneline` is the running history.
- **Don't attribute reasoning to "senior reviewer" / "senior critique" in PR descriptions, commit messages, or code comments.** State the rationale directly. (Memory: `feedback_no_senior_reviewer_attribution.md`.)
- Branch from `main`. Single-PR-per-feature; PR titles match the merged commit subject.
- Never `--no-verify`, never `--no-gpg-sign` unless explicitly authorized.
- Use HEREDOC for multi-line commit messages to preserve formatting.

PR description template (loose, but the existing PRs follow it):
1. **What changed** (1-2 sentences, the *why*)
2. **Why now** (the calibration / incident / blocker that motivated the change)
3. **Verification** (tests added, e2e run if applicable)
4. **Honest scope statement** when the PR is bigger or smaller than the title implies — call out what it doesn't fix

## What's planned vs. what exists

`[meta:roadmap]` Cross-ref: [PLAN.md](PLAN.md), [docs/codebase_info.md](docs/codebase_info.md)

| Tier | Target | Status |
|---|---|---|
| 1 | Skill files (`SKILL.md`) | ✅ implemented (`evolution/skills/`) |
| 2 | Tool descriptions | ✅ implemented (`evolution/tools/`) — MCP-JSON and Hermes-Python-AST adapters; one target tool per run |
| 3 | System prompt sections | 🔲 stub (`evolution/prompts/`) |
| 4 | Tool implementation code | 🔲 stub (`evolution/code/`); needs `[darwinian]` extra |
| 5 | Continuous improvement loop | 🔲 stub (`evolution/monitor/`) |

Open questions deferred to future PRs (per `PLAN.md` deviation notes):
- GEPA Pareto-frontier checkpointing (so a `TimeoutError` mid-run doesn't lose all candidates)
- Cost ceiling kill switch (`max_total_cost_usd`)
- Skill-size-based reflection-LM timeout scaling
- BCa bootstrap upgrade once N≥20 routinely

## When to consult which doc

`[meta:doc-routing]` Cross-ref: [docs/index.md](docs/index.md)

| Question | Doc |
|---|---|
| What does flag X do? | [docs/interfaces.md](docs/interfaces.md) |
| Where is component Y? | [docs/components.md](docs/components.md) |
| Why was decision Z made? | [docs/architecture.md](docs/architecture.md) (decisions section) |
| What's in this JSON file? | [docs/data_models.md](docs/data_models.md) |
| How does the deploy gate decide? | [docs/architecture.md](docs/architecture.md) + [docs/components.md](docs/components.md) (`constraints.py`) |
| What dependency do I add for X? | [docs/dependencies.md](docs/dependencies.md) |
| What are the documented flow paths? | [docs/workflows.md](docs/workflows.md) |
| Where am I supposed to put new tests? | This file (Testing section) |
| What does the long-term roadmap look like? | [PLAN.md](PLAN.md) |
| Why use this over raw DSPy + GEPA? | [docs/framework_advantages.md](docs/framework_advantages.md) |
| Recent project state / commit history | `git log --oneline` |
