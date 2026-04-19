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
```

### Recommended safe path: codex-batched

Use the bounded Codex CLI path when you want explicit model invocations and
conservative runtime guardrails.

```bash
# Conservative cached-dataset run with hard limits
# Dataset directory must already contain: train.jsonl, val.jsonl, holdout.jsonl

HERMES_EVOLUTION_MAX_CODEX_CALLS=2 \
HERMES_EVOLUTION_PHASE_TIMEOUT_SECONDS=90 \
HERMES_EVOLUTION_MAX_RUN_SECONDS=180 \
HERMES_EVOLUTION_MAX_EXAMPLES=4 \
python -m evolution.skills.evolve_skill_codex \
    --skill plan \
    --eval-source cached \
    --dataset-path ./datasets/plan-smoke \
    --iterations 1 \
    --hermes-repo "$HERMES_AGENT_REPO"
```

This path is currently the safer option because it:
- uses explicit `codex exec` subprocess phases instead of hidden DSPy call trees
- enforces explicit call/time budgets before each phase starts
- prefers cached datasets over live generation
- currently supports cached datasets and `iterations=1` only
- writes structured metrics including per-example results and recommendation metadata

### Legacy DSPy path

The original `evolve_skill` entrypoint is still present for compatibility and
experiments, but it is not the recommended path when running against a local
OpenAI-compatible provider. In that configuration, it is intentionally blocked
because the DSPy-based flow can fan out into too many hidden calls.

```bash
# Legacy / experimental path
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source sessiondb
```

## What It Optimizes

| Phase | Target | Engine | Status |
|-------|--------|--------|--------|
| **Phase 1** | Skill files (SKILL.md) | DSPy + GEPA | ✅ Implemented |
| **Phase 2** | Tool descriptions | DSPy + GEPA | 🔲 Planned |
| **Phase 3** | System prompt sections | DSPy + GEPA | 🔲 Planned |
| **Phase 4** | Tool implementation code | Darwinian Evolver | 🔲 Planned |
| **Phase 5** | Continuous improvement loop | Automated pipeline | 🔲 Planned |

## Engines

| Engine | What It Does | License |
|--------|-------------|---------|
| **[DSPy](https://github.com/stanfordnlp/dspy) + [GEPA](https://github.com/gepa-ai/gepa)** | Reflective prompt evolution — reads execution traces, proposes targeted mutations | MIT |
| **[Darwinian Evolver](https://github.com/imbue-ai/darwinian_evolver)** | Code evolution with Git-based organisms | AGPL v3 (external CLI only) |

## Guardrails

Current codex-batched runs enforce:
1. **Explicit runtime/call budgets** before each Codex phase starts
2. **Cached-dataset evaluation only**
3. **Size/growth/structure constraints** on the candidate skill
4. **Evaluation acceptance gating** — candidates are rejected unless evaluation prefers them with positive improvement
5. **Human review** — changes are still intended to go through PR review rather than unattended promotion

Recommended validation before merging any evolved change:
- run the relevant project tests explicitly
- inspect the structured evaluation output and generated diff

## Full Plan

See [PLAN.md](PLAN.md) for the complete architecture, evaluation data strategy, constraints, benchmarks integration, and phased timeline.

## License

MIT — © 2026 Nous Research
