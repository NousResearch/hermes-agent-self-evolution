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

# Optional: point at a specific hermes-agent repo. The default is
# ~/.hermes/hermes-agent (the standard CLI install location), and
# ../hermes-agent (sibling checkout) is searched as a final fallback.
export HERMES_AGENT_REPO=~/.hermes/hermes-agent

# Evolve a skill against synthetic test cases (no transcripts read).
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source synthetic

# Use LLM-as-judge scoring (more accurate, more expensive).
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --use-llm-judge

# Use real session history from Claude Code, Copilot, and Hermes.
# Requires explicit consent because transcripts are sent to the eval/judge LLM.
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --eval-source sessiondb \
    --consent-external-ingest

# Run with the hermes-agent pytest suite as a regression gate, and write
# a proposal bundle (baseline / evolved / diff / decision) for review.
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --run-tests \
    --create-pr
```

### How it integrates with hermes-agent

This package operates *on* a separate hermes-agent install, not inside it.
The pipeline:

1. Discovers your hermes-agent checkout via `HERMES_AGENT_REPO`,
   `~/.hermes/hermes-agent`, or a sibling `../hermes-agent` directory.
2. Reads the target `SKILL.md` from `<hermes-agent>/skills/<skill>/SKILL.md`
   and uses it as the optimization baseline.
3. Generates or loads an evaluation dataset (synthetic / sessiondb / golden).
4. Runs DSPy + GEPA reflective evolution to mutate the skill body.
5. Validates the evolved skill against constraint gates (size, growth,
   structure, optional `pytest` against the hermes-agent test suite, optional
   benchmark regression check).
6. Writes results to `output/<skill>/<timestamp>/`. With `--create-pr`,
   also writes a proposal bundle to `output/proposals/<skill>/<timestamp>/`
   containing `baseline_skill.md`, `evolved_skill.md`, `diff.patch`, and
   `decision.json` for human review.

The package never overwrites a skill in your hermes-agent tree on its own.
Deployment is always: **inspect the proposal → copy `evolved_skill.md` into
the hermes-agent skills directory by hand (or open a PR).** This is by design
— the optimizer can produce regressions that pass synthetic gates, so a human
review step is non-optional. (Closes upstream issue #18.)

### Using MiniMax

Set your API key and pass `--use-minimax` (or specify the model directly):

```bash
export MINIMAX_API_KEY=your_key_here

# Shorthand — uses MiniMax-M2.7 for both optimizer and eval
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --use-minimax

# Explicit model selection
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --optimizer-model minimax/MiniMax-M2.7 \
    --eval-model minimax/MiniMax-M2.7-highspeed
```

Supported MiniMax models:

| Model ID | Description |
|----------|-------------|
| `MiniMax-M2.7` | Peak performance — default choice |
| `MiniMax-M2.7-highspeed` | Same performance, lower latency |

MiniMax uses the OpenAI-compatible endpoint at `https://api.minimax.io/v1`. The
`MINIMAX_API_KEY` environment variable is read automatically; no other
configuration is needed.

## What It Optimizes

| Phase | Target | Engine | Status |
|-------|--------|--------|--------|
| **Phase 1** | Skill files (SKILL.md) | DSPy + GEPA | ✅ Implemented |
| **Phase 2** | Tool descriptions | DSPy + GEPA | 🔲 Stub only — `evolution/tools/` is empty |
| **Phase 3** | System prompt sections | DSPy + GEPA | 🔲 Stub only — `evolution/prompts/` is empty |
| **Phase 4** | Tool implementation code | Darwinian Evolver | 🔲 Stub only — `evolution/code/` is empty |
| **Phase 5** | Continuous improvement loop | Automated pipeline | 🔲 Not started |

Phase 1 (skills) is the only working pipeline today. The remaining phases are
package skeletons reserving the API surface; do not assume they ship.

## Privacy & Security

Evolution sends content to third-party LLMs by design. Defaults are
deliberately conservative; opt in explicitly when stronger trade-offs are
acceptable.

- **Synthetic eval source (default):** the LLM only sees the SKILL.md file
  itself (which is your own checked-in content). No transcripts are read.
- **Sessiondb eval source:** reads chat transcripts from `~/.claude/projects/`,
  `~/.copilot/`, and `~/.hermes/sessions/`, runs them through a secret-pattern
  filter, and forwards relevance-scored snippets to the configured eval/judge
  LLM. This is **gated behind `--consent-external-ingest`** — the run aborts
  with an error otherwise. The default scoring model is `gpt-4.1-mini` /
  `gpt-4.1`; for an offline path, point `--eval-model` and `--judge-model` at
  a local OpenAI-compatible endpoint (vLLM, Ollama).
- **`--use-minimax`** sets MiniMax (a Chinese-jurisdiction provider) as the
  *default* model when none is specified. User-supplied `--optimizer-model` /
  `--eval-model` / `--judge-model` always win — `--use-minimax` will not
  silently re-route a model you explicitly chose.
- **Secret detection** is heuristic and pattern-based (OpenAI / Anthropic /
  GitHub / GitLab / Slack / AWS / Stripe / Twilio / SendGrid / JWT / private
  keys / generic env-var assignments). It is **defence-in-depth, not
  authoritative** — pair with `detect-secrets` or `gitleaks` for any output
  that ships externally. Evolved skills are scrubbed against the same patterns
  before being persisted to disk.
- **`output/`** is `.gitignore`d. Evolved skills can carry secret-shaped
  paraphrases of session content; never `git add output/` blindly.
- **Skill-text injection.** Skill bodies are wrapped with an
  "untrusted-data" preamble so smuggled "ignore previous instructions"-style
  content in the body is treated as data, not commands. The optimizer's
  signature uses HTML-comment sentinels (`<!-- HERMES_SKILL_BODY_START -->`
  / `END`) so the body can be cleanly recovered even when it contains
  markdown horizontal rules.
- **`--run-tests`** invokes `python -m pytest` inside the hermes-agent
  checkout. The runner refuses to execute if the path does not look like a
  hermes-agent repo (missing `pyproject.toml` referencing `hermes-agent` or
  missing `tests/` directory). Pointing `--hermes-repo` at an untrusted tree
  is still equivalent to running its `conftest.py`; do not.

## Engines

| Engine | What It Does | License |
|--------|-------------|---------|
| **[DSPy](https://github.com/stanfordnlp/dspy) + [GEPA](https://github.com/gepa-ai/gepa)** | Reflective prompt evolution — reads execution traces, proposes targeted mutations | MIT |
| **[Darwinian Evolver](https://github.com/imbue-ai/darwinian_evolver)** | Code evolution with Git-based organisms | AGPL v3 (external CLI only) |

## Guardrails

Every evolved variant must pass:
1. **Structural integrity** — full SKILL.md (frontmatter + body) parses cleanly,
   has `name:` and `description:` fields, and a body with at least two of
   {headings, procedural language, substantial content}.
2. **Size limits** — Skills ≤15KB body by default (configurable), tool
   descriptions ≤500 chars.
3. **Growth limits** — body cannot grow more than +20% over baseline.
4. **No-op gate** — runs whose evolved text equals baseline are flagged
   no-op, not "successful improvement". Extraction failures are surfaced
   distinctly from genuine no-ops.
5. **Optional pytest gate** (`--run-tests`) — runs the hermes-agent test
   suite; rejects the variant on any failure. Refuses to run when the
   target tree does not look like hermes-agent.
6. **Optional benchmark gate** (`--run-tblite`, off by default) — TBLite
   regression check. The runner is currently a stub returning `skipped=True`;
   wire hermes-agent `batch_runner` into `evolution/core/benchmark_gate.py`
   to enforce it for real.
7. **Human review** — `--create-pr` writes a proposal bundle to
   `output/proposals/<skill>/<ts>/` (baseline, evolved, diff, decision).
   The package never auto-merges into hermes-agent; deployment is a manual
   copy or PR step.

## Full Plan

See [PLAN.md](PLAN.md) for the complete architecture, evaluation data strategy, constraints, benchmarks integration, and phased timeline.

## License

MIT — © 2026 Nous Research
