# Usage guide

Worked examples for every evolution target, plus tuning, shipping, and validation. For the exhaustive per-flag reference see [docs/interfaces.md](interfaces.md); run `python -m evolution.<module> --help` on any command for the same flags inline.

All commands install the same way:

```bash
git clone https://github.com/jramos/agent-self-evolution.git
cd agent-self-evolution
uv sync
```

## Providers and models

The framework resolves the optimizer, reflection, eval, and judge LMs from your environment. See [docs/model_resolution.md](model_resolution.md) for the full provider mapping, local-server (vLLM/Ollama/LM Studio) examples, and per-role override patterns.

**With Hermes Agent (no env vars).** If `~/.hermes/config.yaml` exists, runs pick up your provider, model, and credentials automatically — whatever model Hermes uses (Anthropic, OpenRouter, Nous Portal, OpenAI, AWS Bedrock, a local server, etc.) becomes the default for all four LM roles. On Hermes setups with a single model, all roles collapse onto it. OAuth setups (e.g. Nous Portal) refresh via `hermes model`; API-key setups read `~/.hermes/config.yaml`'s inline `api_key` or `~/.hermes/auth.json`'s pool. On startup a ~$0.0001 credential probe runs; if anything is stale you get a Rich error panel with the exact recovery command (e.g. `hermes auth add anthropic`) instead of a traceback.

Override per role for multi-model providers:

```bash
uv run python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --optimizer-model anthropic/claude-opus-4-5 \
    --reflection-model anthropic/claude-opus-4-5 \
    --eval-model anthropic/claude-haiku-4-5
```

**Without Hermes Agent.** Set any standard provider env var; the framework auto-detects in priority order (`ANTHROPIC_API_KEY` → `OPENROUTER_API_KEY` → `OPENAI_API_KEY` → others). With neither Hermes nor an env var configured, it exits with an actionable message listing what it tried.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run python -m evolution.skills.evolve_skill --skill writing-skills --iterations 10
```

## Evolve a skill

Skills are resolved by walking `SkillSource` adapters in priority order (roots that don't exist on disk are skipped):

1. **`--skill-source-dir PATH`** (repeatable) — generic `<dir>/<name>/SKILL.md`. Use for Codex, openclaw, or any custom framework.
2. **Hermes Agent** — set `SKILL_SOURCES_HERMES_REPO=/path/to/hermes-agent` (or have `~/.hermes/hermes-agent` exist). Layout: `<root>/skills/<category>/<name>/SKILL.md`.
3. **Claude Code** — auto-discovered if `~/.claude/plugins/cache/` exists. Layout: `<vendor>/<plugin>/<version>/skills/<name>/SKILL.md`.

```bash
# Hermes
export SKILL_SOURCES_HERMES_REPO=~/.hermes/hermes-agent
uv run python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10 --eval-source synthetic

# Claude Code (no env var needed if Claude Code is installed)
uv run python -m evolution.skills.evolve_skill --skill writing-skills --iterations 10 --eval-source synthetic

# Any custom layout
uv run python -m evolution.skills.evolve_skill --skill my-skill --skill-source-dir ~/path/to/my-skills --iterations 10
```

## Evolve a tool description

For agents using MCP, Anthropic tool-use, OpenAI function calling, or any registry exportable to MCP's `list_tools()` shape:

```bash
uv run python -m evolution.tools.evolve_tool \
    --tool search_files --manifest /path/to/your/mcp-tools.json --iterations 5
```

Reads the static MCP-shape manifest, evolves one tool's top-level `description` via GEPA, writes to `output/tools/<tool>/<timestamp>/`. `--apply` rewrites the source manifest in place (every non-target tool's description, `inputSchema`, and any `_evolution_metadata` are preserved verbatim); `--patch` emits a unified diff. At evaluation time the agent sees the full rendered manifest, so cross-tool regressions (an evolved description "stealing" a confusable neighbor's selections) surface through the deploy gate.

**Hermes Agent tools** (Python `*_SCHEMA` dicts): point `--manifest` at the tools directory.

```bash
uv run python -m evolution.tools.evolve_tool \
    --tool read_file --manifest /path/to/hermes-agent/tools --fitness-profile balanced --iterations 5
```

The framework parses every `*_SCHEMA = {...}` / `*_SCHEMAS = [...]` via AST, handles literal-string descriptions and one-hop Name references, and refuses f-string-built descriptions (rewrite to a literal first). Unparseable tools appear in `gate_decision.json.dataset.dropped_tools`. With `--apply`, the evolved description is spliced into the source bytes at the original position — comments, formatting, and unrelated tools untouched.

## Evolve a system prompt section

For Hermes Agent, evolve a named top-level string constant in `agent/prompt_builder.py` (e.g. `MEMORY_GUIDANCE`):

```bash
uv run python -m evolution.prompts.evolve_prompt_section \
    --section MEMORY_GUIDANCE --hermes-repo /path/to/hermes-agent \
    --tasks evolution/validation/suites/memory_guidance.jsonl --iterations 10
```

Unlike skill/tool evolution, a prompt section is evaluated **purely behaviorally**: each candidate is spliced into the live `prompt_builder.py` and scored by running the real agent (`hermes -z`) against the suite. The verdict is compound — Layer 1 checks whether the expected tool was invoked, Layer 2 runs an LLM judge over the result against each task's rubric. The file is restored byte-for-byte after each run (atomic backup + flock + checksum-drift detection). `--apply` writes the evolved section in place; results land in `output/prompts/<section>/<timestamp>/`. To demonstrate the loop on an already-tuned section (which the saturation pre-flight will otherwise default-deny), `--baseline-override-file` starts evolution from arbitrary text (e.g. a deliberately weakened baseline).

**Claude Code `CLAUDE.md` convention** (`--target claude`): the evolvable section is a sentinel-delimited region in a `CLAUDE.md` (`<!-- evolve:NAME start -->` … `<!-- evolve:NAME end -->`); the agent is driven with `claude -p`.

```bash
uv run python -m evolution.prompts.evolve_prompt_section \
    --target claude --section REPO_CONVENTIONS --claude-md ./CLAUDE.md \
    --tasks evolution/validation/suites/claude_conventions.jsonl --agent-model sonnet --apply
```

Headless runs authenticate with `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`). The defensible headroom is **project-specific conventions** the base prompt cannot know (e.g. "run tests with `./bin/check`, never `pytest`") — not generic disciplines it already enforces. The verdict is convention adherence (scored from the agent's `Bash` calls, no LLM judge). During evolution the candidate is injected via `--append-system-prompt` inside an OS sandbox (filesystem confined to the task fixture), so your real `CLAUDE.md` is touched only by `--apply`.

## Mine real session history for evals

```bash
# Skills: pulls real usage from Claude Code (~/.claude/history.jsonl), Copilot, and Hermes session logs
uv run python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10 --eval-source sessiondb

# Tool descriptions: mines Hermes session JSON for (user_task, invoked_tool) pairs, re-judges against the manifest
uv run python -m evolution.tools.evolve_tool --tool search_files --manifest /path/to/mcp-tools.json --eval-source sessiondb
```

For tools, misselections (judge picks a different tool than the agent did, with high confidence) become flipped-label training examples that exercise exactly the failure mode evolution targets. Add `--dry-run` to confirm session discovery before spending. Only Hermes is mined for tool data — Claude Code and Copilot logs don't carry `tool_use` blocks. The eval is biased toward your session distribution, so it may underrepresent confusable-neighbor cases the synthetic eval targets directly; run synthetic first for that coverage.

## Tune the search

`--fitness-profile {balanced,compression,growth}` weights the three judge dimensions (correctness/procedure/conciseness) and selects the proposer template; a few `--gepa-*` knobs control exploration and candidate selection. The chosen profile is recorded in `gate_decision.json`. Full tables (weights, GEPA acceptance/minibatch, knee-point strategy) are in [docs/interfaces.md](interfaces.md); reach for them on calibration runs or when the saturation pre-flight flags a degenerate signal.

## Ship the evolved artifact

By default the artifact lands in `output/<artifact>/<timestamp>/` and stops there. Three opt-in, independent flags automate delivery (all no-ops on a reject decision):

```bash
# Copy the evolved artifact over the source in place on a deploy decision (no git operations)
uv run python -m evolution.skills.evolve_skill --skill X --apply

# Emit a unified diff instead — review by hand
uv run python -m evolution.skills.evolve_skill --skill X --patch | git apply

# Open a draft PR against the source repo (branch from origin/main, atomic commit, push, gh pr create)
uv run python -m evolution.skills.evolve_skill --skill X --create-pr --pr-draft
```

`--apply` skips (with a warning) when the source is under Claude Code's read-only plugin cache. `--create-pr` skips cleanly when the source isn't git-backed. **Do not pair `--create-pr` with campaign loops** — every accepted run opens its own PR. The `--pr-*` family (base branch, branch prefix, draft, allow-dirty) is documented in [docs/interfaces.md](interfaces.md).

## Safety

```bash
# Abort cleanly when cumulative LM cost exceeds a ceiling (worst-case overshoot: one call)
uv run python -m evolution.skills.evolve_skill --skill X --max-total-cost-usd 5.00

# Run your own command as a deploy gate after the framework's gate passes (nonzero exit -> reject)
uv run python -m evolution.tools.evolve_tool --tool X --manifest Y \
    --benchmark-cmd 'pytest -k smoke && custom_check.sh "$EVOLVED_PATH"'
```

On cost abort, `gate_decision.json` carries `decision="aborted"`, `reason="cost_ceiling_exceeded"`, and the full `cost_summary`. The `--benchmark-cmd` hook receives `EVOLVED_PATH`, `BASELINE_PATH`, `RUN_DIR`, `TARGET_NAME`, `ARTIFACT_TYPE` and runs under `/bin/sh -c` (invoke binaries by full name; the command string is yours — don't pass strings you didn't write).

## Saturation pre-flight

Every `evolve_skill` / `evolve_tool` run first scores the baseline on the holdout (and the closed-loop suite, if configured) and classifies into `healthy` / `no_headroom` / `weak_signal` / `uniform_failure`, refusing to spend GEPA budget on an already-saturated baseline:

```
Saturation check: holdout=0.987 (50 ex), closed-loop=1.000 (7 tasks)
╭─── No measurable headroom ───────────╮
│ Band: no_headroom                    │
│ • Baseline already saturates the eval│
│ • Try a harder closed-loop suite     │
╰──────────────────────────────────────╯
Non-interactive context; refusing to proceed.
```

Interactive contexts prompt for confirmation on non-`healthy` bands; non-interactive contexts default-deny with an override hint. Net cost is ~zero (the probe's holdout scores are reused at post-GEPA evaluation). `--no-saturation-check` skips the probe; `--force-saturation-check` runs it but proceeds regardless of band.

## Closed-loop validation (real agent on real tasks)

The synthetic deploy gate is itself a closed loop (an LM scoring another LM's output on tasks a third LM invented). To break it, point a real agent at a small task suite with the baseline and evolved artifacts and see whether real behavior shifted:

```bash
uv run python -m evolution.validation.closed_loop \
    --tool patch --hermes-repo ~/.hermes/hermes-agent \
    --tasks evolution/validation/suites/patch.jsonl \
    --baseline ~/.hermes/hermes-agent/tools/file_tools.py --evolved /tmp/evolved/file_tools.py
```

For each task the harness installs baseline then evolved into your hermes-agent (atomically, with a `.cl_backup` and `fcntl.flock`), invokes `hermes -z`, parses the session JSON, and scores against `expected_tools` / `forbidden_tools`. Decision rule: pass iff `evolved_pass_rate >= baseline_pass_rate` AND (no per-task loss OR wins offset losses 2:1). Exit 0 on pass, 1 on regression — drop-in for `--benchmark-cmd`. Cost: each task is one `hermes -z` run (~$0.05–$0.50).

During *evolution*, the same path runs inline with `--closed-loop-during-evolution <suite.jsonl>` (feeding behavioral scores back into GEPA) and also against **Claude Code** with `--closed-loop-agent-backend claude` (each candidate skill delivered as a plugin to a sandboxed `claude -p`; pick the model with `--closed-loop-agent-model`). When the behavioral suite *is* the ground truth for the change, `--closed-loop-gate-primary` makes it the deploy gate directly.

> Note: `evolution.code.evolve_code` and `evolution.monitor` are documented via `--help` (they do not yet have sections in [docs/interfaces.md](interfaces.md)).
