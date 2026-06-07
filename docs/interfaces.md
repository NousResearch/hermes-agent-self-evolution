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
| `--gepa-minibatch-size <int>` | `3` | GEPA's reflective minibatch size. Default matches GEPA's own default. Bump to ~8 when the saturation pre-flight flags `weak_signal` (wider sampling window makes discriminating examples appear in ~68% of minibatches vs ~34% at default). Aborts at startup if value exceeds trainset size. |
| `--gepa-acceptance {strict-improvement,improvement-or-equal}` | `improvement-or-equal` | GEPA's acceptance criterion under the `sum(minibatch_scores)` gate. `improvement-or-equal` (default) allows plateau-equal candidates through — the literature-recommended fix for noisy LM-judge fitness where strict acceptance rejects ~50% of true-equal mutations. `strict-improvement` is the legacy `gepa<0.1.2` default; pass it only to reproduce strict-acceptance behavior for comparison runs. Mapped to GEPA's underlying `acceptance_criterion` kwarg with hyphens converted to underscores. |

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
| `--knee-point-epsilon <float>` | `1/n_val` | ε for knee-point Pareto band. Only consulted by `--knee-point-strategy smallest`; the default `val-best` path defers to GEPA's val-argmax and ignores ε. Override only with calibrated reason. |
| `--knee-point-strategy {val-best,smallest}` | `val-best` | How to pick the deployed candidate from GEPA's output. `val-best` (default): defer to GEPA's `detailed_results.best_idx` — empirical calibration showed the ε-band walker picked GEPA's default on every observed run, so this path skips the band entirely. `smallest`: walk the ε-band in ascending body-char order (greedy parsimony) for users explicitly chasing compression even at val cost. |
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
| `--create-pr / --no-create-pr` | off | On a deploy decision, branch the source repo, atomically copy the evolved artifact in, commit, push, and open a GitHub PR via `gh pr create`. Skips cleanly when the source isn't git-backed (e.g. Claude Code plugin cache). Skips when the working tree is dirty unless `--pr-allow-dirty` is also set. Requires `gh` on `$PATH`. |
| `--pr-base-branch <name>` | `main` | Target branch for the PR opened by `--create-pr`. The PR's head branch is created from `origin/<base>`. |
| `--pr-branch-prefix <prefix>` | `evolve/` | Prefix for the PR's head branch under `--create-pr`. Branch names become `{prefix}{artifact}-{timestamp}-{hex}`. |
| `--pr-draft` | off | Open the `--create-pr` PR as a draft. Recommended for personal automation pipelines that want a human review gate before merge. |
| `--pr-allow-dirty` | off | Override `--create-pr`'s dirty-tree refusal. Default behavior skips PR creation when the source repo has uncommitted changes, to avoid sweeping unrelated edits into the evolution PR. |

`--apply`, `--patch`, and `--create-pr` are all no-ops on a reject decision and emit a one-line stderr notice in that case. All three default off; they only fire when the user opts in.

### Misc
| Flag | Default | Notes |
|---|---|---|
| `--dry-run` | off | Validate setup; don't run optimization. |
| `--evaluate-band-on-holdout / --no-evaluate-band-on-holdout` | off | Calibration telemetry: after the picked candidate is selected, re-evaluate every candidate in the knee-point band on the holdout and write `band_holdout.json` alongside `gate_decision.json`. Adds judge calls proportional to band size × holdout examples (subsampled to ≤100). Off by default to keep production runs cheap. |
| `--max-total-cost-usd FLOAT` | off | Safety net: abort cleanly when cumulative LM cost exceeds this dollar amount. Worst-case overshoot is one LM call past the ceiling (the cost callback fires AFTER the call returns; the next call aborts at start). 0 is accepted (aborts on first call). Negatives rejected. Writes a `decision="aborted"` `gate_decision.json` with `cost_at_abort_usd`, `cost_ceiling_usd`, and the full `cost_summary` block. |
| `--benchmark-cmd "<shell command>"` | off | Deploy-gate hook: shell command run AFTER the framework's own deploy gate passes; nonzero exit flips the decision to `reject` with `reason="benchmark_failed"`. Receives `EVOLVED_PATH`, `BASELINE_PATH`, `RUN_DIR`, `TARGET_NAME`, `ARTIFACT_TYPE` via env. Runs under `/bin/sh -c`; aliases and shell functions from your interactive shell are not available. Trust boundary: the command string is yours; do not pass strings you didn't write. Adds a `benchmark` block to `gate_decision.json`. |
| `--benchmark-timeout-seconds INT` | `600` | Wall-clock cap for the `--benchmark-cmd` hook. Timeout treated as a benchmark fail with `reason="timeout"`. |
| `--closed-loop-during-evolution <suite.jsonl>` | off | Path to a closed-loop JSONL task suite. The skill-side flow drives `hermes -z` against a temporary working copy of the resolved `SKILL.md` for each task; same `--closed-loop-mode`/`--closed-loop-in-valset` semantics as `evolve_tool`. The full closed-loop flag family (`--closed-loop-mode`, `--closed-loop-saturation-threshold`, `--closed-loop-min-iters`, `--closed-loop-window-size`, `--closed-loop-in-valset`, `--closed-loop-agent-model`, `--closed-loop-task-timeout-seconds`) is wired symmetrically with `evolve_tool`. |
| `--no-saturation-check` | off | Skip the saturation pre-flight (`evolution/core/saturation_check.py`). By default, the framework scores the baseline on the holdout (and the closed-loop suite, if `--closed-loop-during-evolution` is set) BEFORE GEPA starts; non-`healthy` bands prompt for confirmation (interactive) or default-deny (non-interactive) with a `--force-saturation-check` override. Pass `--no-saturation-check` to skip the probe entirely. |
| `--force-saturation-check` | off | Run the saturation pre-flight, render the panel, but proceed regardless of band. Required to override a non-`healthy` verdict in non-interactive contexts (no TTY on stdin). Without this in such a context, the framework exits cleanly without spending GEPA budget. |

### Exit conditions
- `sys.exit(1)` if skill not found across all `SkillSource`s — prints available skills per source.
- `sys.exit(1)` if `eval_source` requires `--dataset-path` but none provided.
- `sys.exit(1)` if `sessiondb` finds no relevant examples.
- `sys.exit(1)` if holdout split has fewer than `min_holdout_size` (default 10) examples.
- Returns normally (rejection path) if static or growth-quality gate fails — `evolved_FAILED.md` + `gate_decision.json` are written.

## CLI: `python -m evolution.tools.evolve_tool`

Evolves one tool's top-level `description` field inside an MCP-shape manifest. The agent sees the full rendered manifest at evaluation time, so cross-tool regressions surface through the deploy gate.

### Required flags
| Flag | Purpose |
|---|---|
| `--tool <name>` | Tool name to evolve. Must match a `name` in the manifest. |
| `--manifest <path>` | Path to the MCP-`list_tools()`-shape JSON file. |

### Optional flags
| Flag | Default | Notes |
|---|---|---|
| `--iterations <int>` | `5` | GEPA `max_full_evals`. |
| `--fitness-profile {compression,balanced,growth}` | `balanced` | Same composite-weighting profile as `evolve_skill`. Maps to `BudgetAwareToolProposer` mode via `resolve_proposer_mode`. |
| `--quality-gate {strict,default,lenient,off,non-inferiority}` | `default` | Same preset semantics as `evolve_skill`. |
| `--max-absolute-chars <int>` | preset value | Override the description's absolute-length ceiling. |
| `--gepa-minibatch-size <int>` | `3` | GEPA's reflective minibatch size; same meaning as the skill-path flag. Bump alongside `--iterations` when the saturation pre-flight flags `weak_signal`. Aborts at startup if value exceeds trainset size. |
| `--gepa-acceptance {strict-improvement,improvement-or-equal}` | `improvement-or-equal` | Same meaning as the skill-path flag. `improvement-or-equal` (default) lets plateau-equal candidates through GEPA's acceptance gate. `strict-improvement` is the legacy `gepa<0.1.2` default; pass it only to reproduce strict-acceptance behavior for comparison runs. |
| `--apply` | off | Rewrite the source manifest file in place with the evolved description on a deploy decision. Preserves every non-target tool's description, `inputSchema`, and any `_evolution_metadata` block. No-op (with stderr notice) when the manifest is under `~/.claude/plugins/cache`. Mutually exclusive with `--patch`. |
| `--patch` | off | Emit a unified diff of (baseline → evolved) manifest JSON to stdout. Mutually exclusive with `--apply`. |
| `--create-pr / --no-create-pr` | off | On a deploy decision, branch the source repo, atomically copy the evolved manifest in, commit, push, and open a GitHub PR via `gh pr create`. Skips cleanly when the source isn't git-backed. Skips when the working tree is dirty unless `--pr-allow-dirty` is also set. Requires `gh` on `$PATH`. |
| `--pr-base-branch <name>` | `main` | Target branch for the PR opened by `--create-pr`. |
| `--pr-branch-prefix <prefix>` | `evolve/` | Prefix for the PR's head branch. Branch names become `{prefix}{tool}-{timestamp}-{hex}`. |
| `--pr-draft` | off | Open the `--create-pr` PR as a draft. |
| `--pr-allow-dirty` | off | Override `--create-pr`'s dirty-tree refusal. |
| `--seed <int>` | `42` | RNG seed for dataset splitting. |
| `--eval-source {synthetic,sessiondb}` | `synthetic` | Where the eval dataset comes from. `synthetic` runs the three-bucket generator (50%/30%/20% target-correct / confusable-neighbor / regression-detection). `sessiondb` mines Hermes session JSON for `(task, invoked_tool)` pairs and re-judges them against the current manifest; misselections at judge confidence ≥0.85 become flipped-label training examples. Claude Code and Copilot logs aren't mined (no tool-call data). |
| `--dry-run` | off | Build the eval dataset and stop. Useful for confirming sessiondb discovery before spending judge + GEPA budget. Returns `{"decision": "dry-run", "dataset_size": N}`. |
| `--max-total-cost-usd FLOAT` | off | Same as the skill-path flag: abort cleanly when cumulative LM cost (dataset gen + judge + GEPA + holdout eval) exceeds this dollar amount. Worst-case overshoot is one LM call. Writes a `decision="aborted"` `gate_decision.json` with `cost_at_abort_usd`, `cost_ceiling_usd`, `cost_summary`, plus the tool-path `artifact_type` and `target_tool` fields for grouping by surface. |
| `--benchmark-cmd "<shell command>"` | off | Same as the skill-path hook: shell command run AFTER deploy gate passes; nonzero exit flips to `reject`. Env vars `EVOLVED_PATH` and `BASELINE_PATH` point at the rendered manifest JSONs in the run dir. `ARTIFACT_TYPE` is `"tool_description"`. |
| `--benchmark-timeout-seconds INT` | `600` | Wall-clock cap for the hook. |
| `--closed-loop-during-evolution <suite.jsonl>` | off | Path to a closed-loop JSONL task suite (same shape consumed by the standalone `closed_loop` CLI). Constructs a `ClosedLoopFeedbackCache` and threads it into the metric. Requires `--closed-loop-hermes-repo`. |
| `--closed-loop-hermes-repo <path>` | required when the suite path is set | Path to the hermes-agent checkout the validator should mutate in place during evolution. |
| `--closed-loop-mode {feedback,trainset,both}` | `feedback` | How the cached verdict participates in GEPA. `feedback`: append a `[CLOSED_LOOP]` block to the reflection LM's input — proposal-prompt signal only, no acceptance change. `trainset`: add behavioral `dspy.Example`s to the trainset whose score (binary pass/fail) contributes to GEPA's `sum(minibatch_scores)` acceptance — lets behavioral wins break judge ties on saturated baselines. `both`: trainset + the `[CLOSED_LOOP]` feedback block on non-behavioral examples (most expensive). |
| `--closed-loop-in-valset / --no-closed-loop-in-valset` | off | When `--closed-loop-mode` is `trainset` or `both`, also include behavioral examples in the valset (adds them to the Pareto frontier + holdout scoring). Each accepted candidate triggers another full-eval pass over the behavioral examples. |
| `--closed-loop-saturation-threshold FLOAT` | `0.95` | Min judge score over the recent window for the saturation gate to open. Only consumed in `feedback` mode (`trainset` / `both` use `gate_mode="always"`). |
| `--closed-loop-min-iters INT` | `3` | Periodic-fire floor: fire closed-loop at least every N reflective iterations even when the judge isn't saturating. `feedback` mode only. |
| `--closed-loop-window-size INT` | `8` | Number of recent judge scores the saturation gate inspects. `feedback` mode only. |
| `--no-saturation-check` | off | Skip the saturation pre-flight (`evolution/core/saturation_check.py`). By default, the framework scores the baseline on the holdout (and the closed-loop suite, if configured) BEFORE GEPA starts; non-`healthy` bands prompt for confirmation (interactive) or default-deny (non-interactive) with a `--force-saturation-check` override. Pass `--no-saturation-check` to skip the probe entirely. |
| `--force-saturation-check` | off | Run the saturation pre-flight, render the panel, but proceed regardless of band. Required to override a non-`healthy` verdict in non-interactive contexts (no TTY on stdin). |

`main()` rejects `--closed-loop-during-evolution` without `--closed-loop-hermes-repo`, and rejects `--closed-loop-mode != feedback` without `--closed-loop-during-evolution`. Local imports keep the validation stack out of cold-path runs.

`--apply`, `--patch`, and `--create-pr` are all no-ops on a reject decision and emit a one-line stderr notice in that case.

### Exit conditions
- `sys.exit(1)` if `--eval-source sessiondb` produces zero usable examples — the run.log includes a per-reason drop breakdown (importer + judge stages); the suggestion is to switch to `--eval-source synthetic`.
- `sys.exit(1)` if the holdout split has fewer than `min_holdout_size` (default 10) examples.
- Returns normally (rejection path) if static or growth-quality gate fails — `evolved_FAILED.json` + `gate_decision.json` are written.

## CLI: `python -m evolution.prompts.evolve_prompt_section`

Evolves one named section of an agent's system prompt — a top-level string constant in Hermes Agent's `agent/prompt_builder.py` (e.g. `MEMORY_GUIDANCE`). Unlike the skill and tool paths, evaluation is **purely behavioral**: there is no synthetic LLM-judge signal. Every candidate is spliced into the live `prompt_builder.py` and scored by running the real agent (`hermes -z`) against the task suite, so the deploy gate is a `ClosedLoopValidator` run (pass-rate + win/loss), not a paired-bootstrap CI over judge scores.

The verdict is **compound**: Layer 1 is the same `expected_tools` / `forbidden_tools` membership rule as the closed-loop tool path; Layer 2 is an LLM judge that scores each `memory(action=add|replace)` call's content against the task's `expected_save_content` rubric (only tasks that declare a rubric are Layer-2 judged). A task may also assert an **action-level** verdict — set `expected_action`, `target_skill`, and `stale_token` (with a `skills_src` skill seeded into the sandbox) and the task passes only when the agent calls `skill_manage(action=patch|edit)` on the target skill and the patch actually touches the stale token. This is what makes discipline sections like `SKILLS_GUIDANCE` (proactively patch a stale skill) scorable. The candidate is spliced in for the duration of the run and the file is restored byte-for-byte afterward, reusing the tool-path backup + flock + checksum-drift machinery.

Behavioral triggers are often stochastic, so both the GEPA fitness and the deploy gate can run each task multiple times and score a **pass rate** rather than a single coin-flip (`--fitness-reps` / `--gate-reps`); when a candidate save/patch judge produces a rationale it is surfaced to GEPA's reflection LM as outcome-grounded feedback rather than the bare budget note. See the signal-strength note below the flag tables.

### Required flags
| Flag | Purpose |
|---|---|
| `--section <name>` | The `prompt_builder.py` top-level string constant to evolve (e.g. `MEMORY_GUIDANCE`). Dict-typed constants (e.g. `PLATFORM_HINTS`) are not supported. |
| `--hermes-repo <path>` | Path to your hermes-agent checkout. `agent/prompt_builder.py` inside it is the splice/restore target. |
| `--tasks <path>` | JSONL eval suite (e.g. `evolution/validation/suites/memory_guidance.jsonl`). Same task shape as the closed-loop tool suite, plus an optional `expected_save_content` rubric per task for Layer 2. Must contain ≥2 tasks (so the split yields a non-empty trainset and holdout). |

### Optional flags
| Flag | Default | Notes |
|---|---|---|
| `--iterations <int>` | `10` | GEPA `max_full_evals`. |
| `--holdout-ratio <float>` | `0.5` | Fraction of tasks held out for the deploy gate. Clamped to keep both the trainset and holdout non-empty. |
| `--seed <int>` | `42` | RNG seed for the train/holdout split and GEPA. |
| `--max-growth <float>` | `0.2` | Section length budget as a fraction over the baseline; framed to the `PromptSectionProposer` so candidates stay near the baseline length (set higher when evolving from a short baseline that needs to grow). |
| `--optimizer-model` / `--reflection-model` / `--eval-model <name>` | config default | Per-role LiteLLM model overrides; resolved like the other CLIs. `--eval-model` is the Layer 2 content judge. |
| `--agent-model <name>` | config default | The model the `hermes -z` agent itself runs as. A deliberately weaker agent exposes more behavioral signal (a strong agent saturates the suite regardless of the prompt). LiteLLM provider prefixes are stripped before `hermes -m`. |
| `--layer2-threshold <float>` | `0.7` | Minimum mean content-judge score for a save task to pass Layer 2. |
| `--fitness-reps <int>` | `3` | Reps per task in the GEPA fitness eval; the score is the mean pass rate (abstentions excluded from the denominator). `1` reproduces single-run scoring. Raise it when the behavior is stochastic — a single rep makes GEPA optimize a coin-flip and overfit to lucky runs. |
| `--gate-reps <int>` | `5` | Reps per task in the deploy-gate eval; the per-task verdict becomes a pass rate and a "win" requires `evolved_rate > baseline_rate`. `1` reproduces the legacy binary gate. The gate ships the decision, so it gets the cleaner (higher-rep) signal. |
| `--task-timeout-seconds <int>` | `120` | Per-task wall-clock cap for `hermes -z`. Timeouts abstain (don't tip the decision). |
| `--max-cost-usd <float>` | `150.0` | Abort cleanly when cumulative **in-process** LM cost (judge + reflection + the passthrough predictor) exceeds this. The agent's own LM spend happens inside the `hermes` child process and is not captured by this ceiling. |
| `--gepa-minibatch-size <int>` | `3` | GEPA reflective minibatch size; same meaning as the other paths. |
| `--gepa-acceptance {improvement-or-equal,strict-improvement}` | `improvement-or-equal` | Same meaning as the other paths. |
| `--apply` | off | On a deploy decision, write the evolved section into `prompt_builder.py` in place (byte-precise AST splice, `ast.parse`-guarded, atomic). |
| `--create-pr` | off | **Deferred for prompt sections** — accepted and recorded as a `skipped` PR block in `gate_decision.json`, but no PR is opened (copying a full evolved `prompt_builder.py` over `origin/<base>` would carry unrelated local changes into the diff). Use `--apply` + a manual PR. |
| `--baseline-override-file <path>` | off | Start evolution from this text instead of the live section. The live section is still the splice/restore target (backed up + restored); `--apply` still writes the evolved text. Use it to create headroom on an already-tuned section (e.g. a deliberately-weakened baseline) or for regression-injection ablations. |
| `--skip-saturation-check` | off | Skip the saturation pre-flight entirely. |
| `--force-saturation-check` | off | Run the pre-flight, render the panel, but proceed regardless of band — required to override a non-`healthy` verdict non-interactively. |
| `--dry-run` | off | Resolve the baseline + build the modules, then stop — exercises wiring with no LM/agent calls. Writes a `decision="dry_run"` `gate_decision.json`. |
| `--output-dir <path>` | `output/prompts/<section>/<timestamp>/` | Where `gate_decision.json` and the baseline/evolved section text files land. |

### Exit conditions
- `0` on a `deploy` decision (or a `--dry-run`).
- `1` on `reject` (the holdout deploy gate found a regression), `denied` (saturated baseline default-denied non-interactively), or `aborted` (cost ceiling).
- `ValueError` at startup if the suite has fewer than 2 tasks.

### Signal strength (read before evolving a stochastic behavioral section)
Evolving a section whose target behavior fires stochastically (e.g. proactive skill-patching) has a real precondition: **the holdout must contain at least one high-base-rate task** — a task the *intended* behavior passes reliably — so GEPA's val-argmax can tell a good candidate from the baseline. With only weak/low-base-rate holdout tasks, selection is noise and reverts to the baseline regardless of `--iterations` or budget; the run emits a `val_signal_warning` (recorded in `gate_decision.json`) when the holdout baseline rates are degenerate (all ≈0 or all ≈1). Put high-signal tasks in **both** train (so the reflection LM learns from them) and holdout (so selection has a signal).

Given a high-base-rate holdout task, the rate-based deploy gate ships real improvements even when they are sub-majority: a per-task "win" requires only `evolved_rate > baseline_rate`, so a candidate that lifts a stochastic trigger from (say) 0/10 toward 4/10 is credited and deployed (`n_wins`), where the legacy binary gate would have scored both as a fail and missed it. Evolved prompts tend to be correct but more concise than a hand-tuned section, so their trigger rate is often lower than a forceful production prompt's — raising `--fitness-reps`/`--gate-reps` stabilizes the measurement of that gain rather than inflating it. Steering the proposer toward more forceful, imperative phrasing (so evolved prompts hit higher trigger rates) is an available enhancement, not a prerequisite for deployment.

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

## CLI: `python -m evolution.validation.closed_loop`

Closed-loop validation: drive a real Hermes Agent through a task suite with the baseline and evolved artifacts, score real tool-selection behavior, compare. Exit 0 on pass, 1 on regression — drop-in compatible with `--benchmark-cmd`.

### Required flags

| Flag | Notes |
|---|---|
| `--tool <name>` | Hermes tool whose description is being validated (e.g. `patch`). Identifies which tool file's description gets spliced. |
| `--hermes-repo <path>` | Path to your hermes-agent checkout. The tool file inside its `tools/` directory is mutated and restored. |
| `--tasks <path>` | JSONL task-suite file. Each task has `task_id`, `user_message` (with optional `{fixture_dir}` placeholder), `expected_tools`, `forbidden_tools`, `fixture_setup`. |
| `--baseline <path>` | Path to the baseline tool-module file (typically `hermes-agent/tools/<file>.py` unmutated). |
| `--evolved <path>` | Path to the evolved tool-module file (an `evolve_tool --apply` output, or a hand-crafted candidate). |

### Optional flags

| Flag | Default | Notes |
|---|---|---|
| `--output-dir <path>` | `output/validation/<tool>/<timestamp>/` | Where the `validation_report.json` lands. |
| `--task-timeout-seconds <int>` | `600` | Per-task wall-clock cap for `hermes -z`. Timeouts count as **abstentions** in the report, not failures — they don't tip the decision either way. |

### Exit conditions

- `0` if the two-condition decision rule passes (`evolved_pass_rate >= baseline_pass_rate` AND (`n_losses == 0` OR `n_wins >= 2 * n_losses`)).
- `1` if either condition fails (regression).
- `StaleBackupError` (non-zero exit, clear message) if a `.cl_backup` file exists from a prior crashed run — the user must `mv` it back before re-running.
- `ConcurrentRunError` if another harness or interactive `hermes` session holds the `fcntl.flock` on the target tool file's parent directory.

### Crash safety

The harness mutates the user's hermes-agent install in place. Defenses:

- `.cl_backup` written atomically before any splice, validated with `ast.parse` before being trusted for restore.
- `fcntl.flock` sentinel in the parent dir prevents concurrent harness runs from corrupting each other's restore.
- `sha256` verification after every task — a YOLO-mode agent that rewrites the spliced file mid-suite (`terminal(echo > file_tools.py)`) is caught before later tasks silently run a corrupt baseline.

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
- `gate_decision.json` schema fields — `tests/skills/test_evolve_skill_validation_flow.py:TestGrowthGateDecisionSchema` and `TestStaticValidationShortCircuitsBeforeHoldout` lock `schema_version="5"` plus the full key list. See [data_models.md](data_models.md).
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
