# Knowledge Base Index

This directory is a structured documentation set for **`agent-self-evolution`** — a Python framework that uses DSPy + GEPA to evolve agent SKILL.md files through reflective prompt optimization, with a paired-bootstrap deploy gate as the final shipping bar.

## How to use this knowledge base (for AI assistants)

**Start here every time.** This file is the entry point — it describes which documents to consult for which kinds of question. Load it into context first; the other docs are loaded on demand.

The codebase is small (~3.5K LOC of source + ~12 test files) but architecturally dense — most of the substance is in *why* things are shaped a certain way, not *what* they are. The docs prioritize that "why."

### Question routing table

| If the user asks about... | Read these (in order) |
|---|---|
| **What this project is** | `codebase_info.md` → `architecture.md` → repo-root `README.md` |
| **How a single run works end-to-end** | `workflows.md` (Workflow 1) → `architecture.md` (top-level flow) |
| **What flag does X / how to run the CLI** | `interfaces.md` (CLI section) |
| **Why the deploy gate rejected a run** | `data_models.md` (gate_decision.json) → `components.md` (`constraints.py`) |
| **What's in `gate_decision.json` / `metrics.json`** | `data_models.md` (full schema with examples) |
| **Where is X implemented** | `components.md` (component-by-component map) |
| **How to add a new SkillSource for framework Y** | `interfaces.md` (SkillSource Protocol) → `components.md` (skill_sources.py) |
| **Why is the synthetic dataset gen LM call configured this way** | `components.md` (dataset_builder.py) → `dependencies.md` (DSPy LM kwargs) |
| **Why is GEPA + MIPROv2 fallback wired this way** | `architecture.md` (decision 7) → `workflows.md` (Workflow 4) |
| **What does `BudgetAwareProposer` do and why custom** | `components.md` (budget_aware_proposer.py) → `architecture.md` (pattern 3) |
| **What's the knee-point selection doing** | `components.md` (knee_point.py) → `architecture.md` (pattern 5) |
| **What's tested vs. not** | `interfaces.md` (test surfaces locked by tests) → `workflows.md` (Workflow 8) |
| **What dependencies are pinned and why** | `dependencies.md` |
| **What's planned but not built** | `codebase_info.md` (implementation status table) → `PLAN.md` |
| **Why use this over raw DSPy + GEPA** | `framework_advantages.md` |
| **What changed recently / project history** | `git log --oneline` |
| **Style / convention questions** | `AGENTS.md` (repo root) |

### When to read source vs. docs

- **Read docs first** for architectural understanding, why-questions, and locating where something lives.
- **Read source for** exact behavior, edge cases, current parameter defaults, recent changes. The docs are accurate as of the listed date but the source is authoritative.
- **Always check `git log`** before recommending changes — the project moves fast and a memory-recalled implementation detail may have been changed in a recent PR.

## Documents in this knowledge base

| File | Purpose | Length |
|---|---|---|
| [`codebase_info.md`](codebase_info.md) | Identity, layout, package structure, LOC, dependencies snapshot, runtime state locations | ~140 lines |
| [`architecture.md`](architecture.md) | One-line model, top-level flow, module dep graph, design patterns, statistical substrate, architectural decisions | ~170 lines |
| [`components.md`](components.md) | Per-module reference: what each owns, public surface, load-bearing implementation notes | ~210 lines |
| [`interfaces.md`](interfaces.md) | CLIs, Python API, SkillSource Protocol, output artifacts, DSPy + litellm integration points, test surfaces, env vars | ~180 lines |
| [`data_models.md`](data_models.md) | All dataclasses, on-disk formats, full `gate_decision.json` schema with worked examples | ~220 lines |
| [`workflows.md`](workflows.md) | 8 step-by-step workflows with mermaid sequence diagrams: deploy path (split into 4 phases), two reject paths, GEPA→MIPROv2 fallback, sessiondb mining, etc. | ~280 lines |
| [`dependencies.md`](dependencies.md) | Each external package — what it's used for, why it's pinned, what we don't depend on | ~180 lines |
| [`framework_advantages.md`](framework_advantages.md) | User-facing explainer of how this framework's selection layer, deploy gate, proposer, and composite fitness differ from raw DSPy + GEPA — and when raw GEPA is the right choice | ~110 lines |
| [`review_notes.md`](review_notes.md) | Consistency + completeness gaps found during this docs pass | ~50 lines |

## Documents elsewhere worth knowing about

| File | Purpose |
|---|---|
| [`../README.md`](../README.md) | User-facing quick start. Skill discovery, evolve-a-skill command, CLI examples. |
| [`../AGENTS.md`](../AGENTS.md) | AI-assistant-focused condensed reference. Project context, dirs, conventions, test/PR guidance. **Read this first when picking up the codebase.** |
| [`../PLAN.md`](../PLAN.md) | Full project roadmap. Tiers 1 and 2 (skills, tool descriptions) are implemented; Tiers 3-5 are planned. Each implemented phase carries a "Deviations from plan" subsection — load-bearing decisions documented in line. |

## Cross-cutting topics with multiple home documents

- **The deploy gate decision** spans `architecture.md` (statistical substrate), `components.md` (`constraints.py`), `data_models.md` (`gate_decision.json` schema), and `workflows.md` (Workflow 1 Phase D, Workflow 2). Read together when debugging a deploy decision.
- **LM observability** lives in `components.md` (`lm_timing_callback.py`), `interfaces.md` (litellm integration), and `dependencies.md` (litellm pinning rationale).
- **Skill discovery** is in `components.md` (`skill_sources.py`), `interfaces.md` (SkillSource Protocol), and `codebase_info.md` (priority order).

## Maintenance notes

The fast-moving parts to verify against source when consulting these docs:

- `EvolutionConfig` defaults (especially `eval_dataset_size`, `growth_*`, `bootstrap_*`)
- `gate_decision.json` schema_version (currently `"4"`)
- LM model defaults in `evolve_skill.py` CLI options
- Test count (currently 282)
- LM `request_timeout` / `num_retries` — may be tuned further

When updating: edit the relevant file, then check whether the "Question routing table" above still points to the right place. The index file is loaded into AI-assistant context every conversation, so small accuracy improvements here pay off broadly.
