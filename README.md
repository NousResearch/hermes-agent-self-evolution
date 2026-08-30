# 🧬 Hermes Agent Self-Evolution

**Evolutionary self-improvement for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Hermes Agent Self-Evolution uses DSPy + GEPA (Genetic-Pareto Prompt Evolution) to automatically evolve and optimize Hermes Agent's skills, tool descriptions, system prompts, and code — producing measurably better versions through reflective evolutionary search.

**No GPU training required.** Everything operates via API calls — mutating text, evaluating results, and selecting the best variants. ~$2-10 per optimization run.

## How It Works

```
Read current skill/prompt/tool ──► Generate eval dataset
                                        │
                                        ▼
                                   GEPA Optimizer ◄── Execution traces
                                        │                    ▲
                                        ▼                    │
                                   Candidate variants ──► Evaluate
                                        │
                                   Constraint gates (tests, size limits, benchmarks)
                                        │
                                        ▼
                                   Best variant ──► PR against hermes-agent
```

GEPA reads execution traces to understand *why* things fail (not just that they failed), then proposes targeted improvements. ICLR 2026 Oral, MIT licensed.

## Quick Start

```bash
# Install
git clone https://github.com/NousResearch/hermes-agent-self-evolution.git
cd hermes-agent-self-evolution
pip install -e ".[dev]"

# Point at your hermes-agent repo
export HERMES_AGENT_REPO=~/.hermes/hermes-agent

# See what this installation can find, optimize, and gate on
hermes-evolve status

# Evolve a skill (synthetic eval data)
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source synthetic

# Or use real session history from Claude Code, Copilot, and Hermes
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source sessiondb
```

Run `hermes-evolve status` first. It reports whether a hermes-agent checkout is
reachable, how many targets each phase finds in it, and which validation gates can
actually run, which is otherwise something you learn by watching a run fail.

## Command Reference

Every phase has a `hermes-evolve` subcommand and an equivalent `python -m` form.

```bash
hermes-evolve status                      # what is reachable and gateable
hermes-evolve skill   --skill NAME        # Phase 1: skill files
hermes-evolve tools                       # Phase 2: tool descriptions
hermes-evolve prompt  --all-sections      # Phase 3: system prompt sections
hermes-evolve code    --tool file_tools   # Phase 4: tool implementation code
hermes-evolve monitor --once              # Phase 5: triage and propose
```

Phases 2 and 3 default to `--no-write`: they measure and report without touching your
hermes-agent checkout. Pass `--write` when you want the evolved text applied to a branch.

Phase 4 is different, because evolving code means running the tests against it. It works
on a temporary `evolve/code/...` branch in your checkout, commits each candidate there so
the lineage is inspectable, and restores the branch you were on when it exits, including
after an error. It refuses to start on a dirty worktree unless you pass `--allow-dirty`,
and it never merges: the output is a branch, a diff, and a PR body for you to review.

Phase 4 is also the only phase that executes code it did not ship with: the external
evolver binary, and every candidate that binary produces (the test suite, the
reproduction script and any benchmark all run with the candidate applied). None of that
executes in your checkout or with your environment. It runs in a disposable clone of the
repo with no remotes and no credential helper, under an environment built from an
explicit allowlist (`HOME` and `TMPDIR` point into the sandbox; pass anything extra by
name with `--sandbox-env NAME`), behind an OS-level sandbox - bubblewrap on Linux,
sandbox-exec on macOS - that denies the network, hides your home directory, and makes
everything outside the workspace read-only. Network stays off unless you pass
`--sandbox-allow-network`, because an evolver that can read its workspace and reach the
network can exfiltrate what it was shown. Candidate `path` values are honoured only
inside the run's own directories, in every mode. On a machine with no sandbox backend
the run refuses to start (exit code 5) rather than degrading silently;
`--unsandboxed` is the explicit waiver and is the only way to get the old behaviour.
The monitor's dispatched phases get an allowlisted environment for the same reason -
what a phase needs, never a copy of your shell.

No phase pushes a branch or opens a pull request unless you ask for it with `--push` or
`--open-pr`.

```bash
# Phase 2: evolve one toolset, keeping the repo untouched
hermes-evolve tools --toolset file --iterations 10

# Phase 3: one section, measured and reported, nothing written
hermes-evolve prompt --section MEMORY_GUIDANCE

# Phase 2: apply the result and build a branch with a PR body, still local
hermes-evolve tools --toolset file --write

# Phase 5: print an installable weekly schedule (installs nothing)
hermes-evolve monitor --emit-cron
```

`--strict-gates` is available on Phases 2, 3 and 4 and turns a gate that could not run
into a hard failure. Be aware of what that means today: hermes-agent ships none of the
benchmarks PLAN.md names, so on a stock checkout the flag blocks every candidate. It is
there for a release process that must prove every gate really ran, not for everyday use.

## What It Optimizes

| Phase | Target | Engine | Status |
|-------|--------|--------|--------|
| **Phase 1** | Skill files (SKILL.md) | DSPy + GEPA | ✅ Implemented |
| **Phase 2** | Tool descriptions | DSPy + GEPA | ✅ Implemented |
| **Phase 3** | System prompt sections | DSPy + GEPA | ✅ Implemented |
| **Phase 4** | Tool implementation code | Darwinian Evolver | ✅ Implemented |
| **Phase 5** | Continuous improvement loop | Automated pipeline | ✅ Implemented |

## Engines

| Engine | What It Does | License |
|--------|-------------|---------|
| **[DSPy](https://github.com/stanfordnlp/dspy) + [GEPA](https://github.com/gepa-ai/gepa)** | Reflective prompt evolution — reads execution traces, proposes targeted mutations | MIT |
| **[Darwinian Evolver](https://github.com/imbue-ai/darwinian_evolver)** | Code evolution with Git-based organisms | AGPL v3 (external CLI only) |

## Guardrails

Every evolved variant must pass:
1. **Full test suite** — `pytest tests/ -q` must pass 100%
2. **Size limits** — Skills ≤15KB, tool descriptions ≤500 chars
3. **Caching compatibility** — No mid-conversation changes
4. **Semantic preservation** — Must not drift from original purpose
5. **PR review** — All changes go through human review, never direct commit

Guardrail 1 is not uniform across phases, and it is worth knowing which one you are
getting:

| Phase | What the test gate actually runs |
|---|---|
| 1 | Nothing. `--run-tests` sets a config field no code reads. |
| 2 | The suite, but only with `--run-tests`; off by default. |
| 3 | A subset, `pytest tests/ -k prompt`. |
| 4 | The full suite, as a hard gate. Any failure rejects the candidate outright. |

Three guardrails are enforced mechanically rather than by convention:

**Structure is frozen.** Tool schemas and prompt constants are rewritten by replacing the
exact source span of a single string literal, never by regenerating the file. For a tool
schema the rewritten source is then re-parsed and its skeleton diffed against the
original, so a candidate that moves a parameter name, type, enum, default, or required
list is rejected before it reaches disk. For a prompt constant the equivalent proof is
`verify_only_sections_changed`, which compares the value of every module-level string
constant and fails if anything but the targeted section moved. Only text can change.

**Deleting is not drifting.** Guardrail 4 is a measurement, not a convention: every
candidate is scored against its baseline on shared content vocabulary and rejected below
`min_semantic_similarity`, which defaults to 0.4 and can be set to 0 to turn the check
off. The measure is deliberately blind to two things, because a frequency-weighted one
rejects the changes these phases exist to make. Repetition is not subject matter, so the
comparison is on the set of terms rather than their counts: a description that repeats
one clause six times is about the same thing as one that says it once. And a candidate
that introduces no vocabulary the baseline did not already have cannot have changed
subject, however much shorter it is. Twelve descriptions in a stock checkout are already
over budget, so a similarity floor must not be the thing that forbids cutting them down.
Whether a shorter description still describes its tool well enough is a real question,
and the cross-tool selection-accuracy gate is what answers it.

**A missing gate is not a passing gate.** A benchmark that is not installed reports
`unavailable`, never `passed`. Runs are permissive by default so the pipeline is usable
today; `--strict-gates` turns any unavailable gate into a hard failure for a release
process that needs every gate to have actually run.

## Full Plan

See [PLAN.md](PLAN.md) for the complete architecture, evaluation data strategy, constraints, benchmarks integration, and phased timeline.

## License

MIT — © 2026 Nous Research
