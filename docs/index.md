# Knowledge Base Index

This directory is a structured documentation set for **`agent-self-evolution`** — a Python framework that uses DSPy + GEPA to evolve agent SKILL.md files and tool descriptions through reflective prompt optimization, with a paired-bootstrap deploy gate as the final shipping bar and an orthogonal closed-loop validation surface that runs the real agent against a JSONL task suite.

## How to use this knowledge base (for AI assistants)

**Start here every time.** This file is the entry point — it describes which documents to consult for which kinds of question. Load it into context first; the other docs are loaded on demand.

The codebase is mid-sized (~9K LOC of source + 61 test files / ~1166 tests) and architecturally dense — most of the substance is in *why* things are shaped a certain way, not *what* they are. The docs prioritize that "why."

### Question routing table

| If the user asks about... | Read these (in order) |
|---|---|
| **What this project is** | `codebase_info.md` → `architecture.md` → repo-root `README.md` |
| **How a skill run works end-to-end** | `workflows.md` (Workflow 1) → `architecture.md` (top-level flow) |
| **How a tool-description run works end-to-end** | `workflows.md` (Workflow 9) → `components.md` (`evolve_tool.py`) |
| **How a prompt-section run works end-to-end** | `workflows.md` (Workflow 12) → `components.md` (`evolve_prompt_section.py`) |
| **How a Claude CLAUDE.md convention run works** | `workflows.md` (Workflow 13) → `components.md` (Claude Code backend) |
| **What flag does X / how to run the CLI** | `interfaces.md` (CLI section) |
| **Why the deploy gate rejected a run** | `data_models.md` (gate_decision.json) → `components.md` (`constraints.py`) |
| **What's in `gate_decision.json` / `metrics.json`** | `data_models.md` (full schema with examples) |
| **What's in `validation_report.json` / how scoring works** | `data_models.md` (ValidationReport) → `components.md` (`validation/report.py`) |
| **Where is X implemented** | `components.md` (component-by-component map) |
| **How to add a new SkillSource for framework Y** | `interfaces.md` (SkillSource Protocol) → `components.md` (skill_sources.py) |
| **How to add a new ToolSource adapter** | `components.md` (tool_source.py, hermes_source.py) → `interfaces.md` (ToolSource Protocol) |
| **Why is the synthetic dataset gen LM call configured this way** | `components.md` (dataset_builder.py) → `dependencies.md` (DSPy LM kwargs) |
| **Why is GEPA + MIPROv2 fallback wired this way** | `architecture.md` (decision 7) → `workflows.md` (Workflow 4) |
| **What does `BudgetAwareProposer` do and why custom** | `components.md` (budget_aware_proposer.py) → `architecture.md` (pattern 3) |
| **What's the knee-point selection doing** | `components.md` (knee_point.py) → `architecture.md` (pattern 5) |
| **How does closed-loop validation work / what defenses** | `components.md` (`evolution/validation/`) → `workflows.md` (Workflow 10) |
| **How does closed-loop signal reach GEPA during evolution** | `components.md` (closed_loop_feedback.py, behavioral_example.py) → `architecture.md` (closed-loop feedback patterns) → `workflows.md` (Workflow 11) |
| **What does `--max-total-cost-usd` actually do on abort** | `data_models.md` (cost-ceiling-abort variant of gate_decision.json) → `components.md` (lm_timing_callback.py) |
| **What does `--benchmark-cmd` do** | `interfaces.md` (CLI: benchmark-cmd) → `data_models.md` (benchmark block) |
| **Why did the run abort before GEPA started / what's the saturation panel** | `components.md` (saturation_check.py) → `architecture.md` (pattern 10) → `workflows.md` (Workflow 1 Phase B.5) → `data_models.md` (SaturationReport) |
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

| File | Purpose |
|---|---|
| [`codebase_info.md`](codebase_info.md) | Identity, layout, package structure, LOC, dependencies snapshot, runtime state locations |
| [`architecture.md`](architecture.md) | One-line model, top-level flow, module dep graph, design patterns, statistical substrate, architectural decisions |
| [`components.md`](components.md) | Per-module reference: what each owns, public surface, load-bearing implementation notes |
| [`interfaces.md`](interfaces.md) | CLIs (skill, tool, closed-loop, sessiondb importer), Python API, SkillSource + ToolSource Protocols, output artifacts, DSPy + litellm integration, test surfaces, env vars |
| [`data_models.md`](data_models.md) | All dataclasses, on-disk formats, full `gate_decision.json` schema with worked examples, `ValidationReport` schema |
| [`workflows.md`](workflows.md) | Step-by-step workflows with mermaid sequence diagrams: skill deploy path, reject paths, GEPA→MIPROv2 fallback, sessiondb mining, tool evolution, closed-loop validation, closed-loop signal during evolution, prompt-section evolution |
| [`dependencies.md`](dependencies.md) | Each external package — what it's used for, why it's pinned, what we don't depend on |
| [`framework_advantages.md`](framework_advantages.md) | User-facing explainer of how this framework's selection layer, deploy gate, proposer, and composite fitness differ from raw DSPy + GEPA — and when raw GEPA is the right choice |

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
- **Saturation pre-flight** is in `components.md` (`saturation_check.py`), `architecture.md` (decision 10), `workflows.md` (Workflow 1 Phase B.5), `data_models.md` (`SaturationReport`), and `interfaces.md` (CLI flags `--no-saturation-check` / `--force-saturation-check`). Read together when debugging a "why did the run abort before GEPA" or "why was the panel suggested" question.

## Maintenance notes

The fast-moving parts to verify against source when consulting these docs:

- `EvolutionConfig` defaults (especially `eval_dataset_size`, `growth_*`, `bootstrap_*`)
- `gate_decision.json` schema_version (currently `"5"`)
- LM model defaults in `evolve_skill.py` / `evolve_tool.py` CLI options
- Test count (currently ~1166)
- LM `request_timeout` / `num_retries` — may be tuned further
- Closed-loop CLI flags on `evolve_tool` (`--closed-loop-during-evolution`, `--closed-loop-mode`, …)
- Saturation pre-flight default thresholds (`evolution/core/saturation_check.py:DEFAULT_THRESHOLDS`) — likely to be calibrated as more real-world bands are observed

When updating: edit the relevant file, then check whether the "Question routing table" above still points to the right place. The index file is loaded into AI-assistant context every conversation, so small accuracy improvements here pay off broadly.
