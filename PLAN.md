# Agent Self-Evolution — Evolutionary Self-Improvement for Agent Skills

## Vision

A standalone pipeline that decides, rigorously, whether a proposed change to an agent — a skill, prompt, tool description, or piece of tool code — actually makes it better, and ships only the ones that do. A **noise-aware deploy gate** plus a **real-agent validation loop** are the durable product; reflective evolutionary search (DSPy + GEPA) is one optional source of candidate changes feeding the gate. Lives in its own repo (`jramos/agent-self-evolution`), operates ON a target agent's repo — not part of it. Originally built for Hermes Agent; now works for any agent framework that emits `<skill>/SKILL.md` files (Claude Code, Hermes, custom local layouts).

> **Where this landed (current — the founding plan below is partly superseded; see [the findings report](reports/asymmetry_findings.md) and Forward Roadmap items 11–12).** The campaign behind this plan found that on a *capable* agent, evolving artifact *text* (skills, tool descriptions, prompt sections) does **not** measurably change behavior in either direction — so for those surfaces the pipeline is **regression-catching, not improvement-finding**, and the deploy gate (not the evolver) is the asset that holds up. The one surface with a real improvement gradient is **tool-code repair from failing-test feedback** (~0.60–0.74 on real bugs, and not only easy ones — ~0.69 on large >20-LOC fixes), honestly scoped as *test-feedback repair*, not autonomous re-derivation. Read the original tiers/phases below as the founding intent; the gate is the headline.

Three engines were planned; what actually carried the value is noted inline:

| Engine | What It Optimizes | License | Integration |
|--------|------------------|---------|-------------|
| **DSPy + GEPA** | Skills, prompts, instructions, tool descriptions | MIT | Native Python, candidate generator (decoupled from capable-agent behavior — see "Where this landed") |
| **Iterative test-feedback repair** | Tool implementation code | MIT | Whole-file repair in an isolated worktree, gated by the deploy gate — *this is what shipped for code*. A head-to-head (item 12) tested the planned Darwinian Evolver's population search vs plain best-of-N resampling on the loop's failures: population search was **strictly dominated by best-of-N** (4/23 vs 6/23) — not justified on this corpus, tested not assumed |
| **DSPy MIPROv2** | Few-shot examples, instruction text | MIT | Native Python, fallback optimizer |

GEPA is a strong candidate *generator* — integrated into DSPy, it reads execution traces to understand WHY things fail (not just that they fail) and works with as few as 3 examples. But the campaign's durable result is that the **deploy gate** that adjudicates candidates, not the evolver, is the asset that holds up: on a capable agent, evolving artifact text doesn't move behavior (see "Where this landed").

**Important: No GPU training required.** Everything in this plan operates via API calls only. DSPy+GEPA and MIPROv2 optimize the *text* of prompts, instructions, and few-shot examples — they mutate and evaluate strings, not model weights. The code path repairs tool files (also text). The only DSPy component that trains weights (`BootstrapFinetune`) is explicitly excluded from this plan. All evaluation runs through batch_runner making standard LLM API calls.

---

## What Can Be Improved

### Tier 1: Skill Files (Highest Value, Lowest Risk)
- **What:** SKILL.md files — procedural instructions the agent follows
- **How:** Wrap skill text as a DSPy module, evaluate on test tasks via batch_runner, evolve with GEPA
- **Why it works:** Skills are pure text, easily mutated, and directly measurable (did the agent complete the task correctly when following this skill?)
- **Example:** Evolve the `github-code-review` skill to produce better reviews by testing against a dataset of known-good code reviews

### Tier 2: Tool Descriptions (Medium Value, Low Risk)
- **What:** The `description` field in tool schemas (what the agent sees when deciding which tool to use)
- **How:** GEPA evolves descriptions, evaluates whether the agent picks the right tool for given tasks
- **Why it works:** Tool selection is a classification problem — perfect for DSPy optimization
- **Example:** Evolve the `search_files` description so the agent picks it over `terminal(grep)` more reliably

### Tier 3: System Prompt Components (High Value, Higher Risk)
- **What:** Sections of the system prompt (persona, policies, formatting instructions)
- **How:** Parameterize prompt_builder.py sections as DSPy Signatures, optimize with GEPA
- **Why it works:** System prompt quality directly determines agent behavior quality
- **Risk:** Must be careful not to break prompt caching — only optimize offline, deploy as new versions
- **Example:** Evolve the "tool usage guidelines" section to reduce unnecessary tool calls

### Tier 4: Code Repair (the one improvement gradient — shipped)
- **What:** Tool implementation code, helper functions
- **How (shipped):** iterative **test-feedback repair** — an LLM proposes a whole-file fix from the failing test's output, in a throwaway git worktree with an isolated venv, accepted only when it passes the gate (surface freeze, file scope, held-out split, baseline-diff regression floor). This is a single-proposer iterative loop, *not* the planned Darwinian-evolver population search — which a head-to-head (item 12) showed is **strictly dominated by plain best-of-N resampling** on the loop's failures, so it is not justified here.
- **Why it works:** a deterministic test is a concrete, executable oracle with no agent between the artifact and the verdict — the one place artifact quality is *not* decoupled from behavior.
- **Result:** deploy-reachable ~0.60–0.74 on real harvested bugs (see Forward Roadmap item 12), honestly scoped as test-feedback repair, not autonomous re-derivation.
- **Risk:** code changes can break things — the gate's anti-gaming checks (held-out split, freeze) are the guardrail.
- **Example:** repair `file_tools.py` traversal handling to match the upstream fix.

---

## Architecture

### The Optimization Loop

```
┌─────────────────────────────────────────────┐
│  1. SELECT TARGET                           │
│     - Pick a skill, prompt section, or tool  │
│     - Load current version as baseline       │
│                                             │
│  2. BUILD EVALUATION DATASET                │
│     - Mine session_db for real usage examples │
│     - Or use hand-crafted test cases         │
│     - Split: train / validation / test       │
│                                             │
│  3. WRAP AS DSPy MODULE                     │
│     - Skill text → dspy.Signature            │
│     - Agent workflow → dspy.ReAct             │
│     - Tool selection → dspy.Predict           │
│                                             │
│  4. RUN OPTIMIZER                           │
│     - Primary: dspy.GEPA (reflective evolution)│
│     - Fallback: dspy.MIPROv2 (bayesian opt)  │
│     - Code: test-feedback repair (gated)     │
│                                             │
│  5. EVALUATE & COMPARE                      │
│     - Run optimized version on held-out test  │
│     - Compare: accuracy, cost, latency        │
│     - Statistical significance check          │
│                                             │
│  6. DEPLOY (with approval)                  │
│     - Git commit the improved version         │
│     - A/B test in production (optional)       │
│     - Rollback mechanism via git revert       │
└─────────────────────────────────────────────┘
```

### Integration Points with Existing Hermes Infrastructure

| Hermes Component | Role in Self-Improvement |
|-----------------|------------------------|
| `batch_runner.py` | Evaluation harness — run agent on test tasks in parallel |
| `agent/trajectory.py` | Collect execution traces for GEPA's reflective analysis |
| `hermes_state.py` (SessionDB) | Mine real usage data for evaluation datasets |
| `skills/` directory | The primary optimization targets |
| `tools/registry.py` | Tool descriptions to optimize |
| `agent/prompt_builder.py` | System prompt components to optimize |
| `tests/` | Guardrails — evolved code must pass all tests |
| Git history | Track all evolution lineage, enable rollback |

### Data Flow

```
SessionDB (real conversations)
    │
    ▼
Evaluation Dataset Builder
    │
    ├──► DSPy Module Wrapper (wraps skill/prompt/tool as optimizable module)
    │        │
    │        ▼
    │    GEPA Optimizer ◄── Execution Traces (from batch_runner)
    │        │                    ▲
    │        │                    │
    │        ▼                    │
    │    Candidate Variants ──► batch_runner (parallel evaluation)
    │        │
    │        ├──► Constraint Validation (tests, char limits, caching compat)
    │        │
    │        ▼
    │    Best Valid Variant
    │        │
    ▼        ▼
Git Branch + PR (with diff, metrics, before/after comparison)
    │
    ▼
Human Review & Merge
```

---

## Implementation Structure

### Where It Lives

Hermes Agent Self-Evolution lives in its own repo (`NousResearch/hermes-agent-self-evolution`), separate from hermes-agent. It pip-installs or clones hermes-agent to access its infrastructure, and outputs PRs against the hermes-agent repo.

```
hermes-agent-self-evolution/             # Standalone repo
├── PLAN.md                             # This file
├── README.md                           # Setup, usage, examples
├── pyproject.toml                      # Package config + dependencies (dspy, gepa)
│
├── evolution/                          # Main package
│   ├── core/                           # Shared infrastructure
│   │   ├── __init__.py
│   │   ├── dataset_builder.py          # Eval dataset generation (synthetic, SessionDB mining)
│   │   ├── fitness.py                  # Fitness functions (LLM-as-judge, rubrics, length penalties)
│   │   ├── constraints.py              # Constraint validators (char limits, caching compat, test suite)
│   │   ├── benchmark_gate.py           # Benchmark gating (run TBLite/YC-Bench, check regression)
│   │   └── pr_builder.py              # Auto-generate PR with metrics, diffs, comparison
│   │
│   ├── skills/                         # Phase 1: Skill evolution
│   │   ├── __init__.py
│   │   ├── evolve_skill.py            # Main entry: python -m evolution.skills.evolve_skill --skill <name>
│   │   └── skill_module.py            # Wraps SKILL.md as DSPy module
│   │
│   ├── tools/                          # Phase 2: Tool description evolution
│   ├── prompts/                        # Phase 3: System prompt evolution
│   ├── code/                           # Phase 4: Code repair (test-feedback, gated)
│   └── monitor/                        # Phase 5: Propose-only triage sentinel
│
├── datasets/                           # Generated eval datasets (gitignored, local)
│   ├── skills/
│   └── tools/
│
└── tests/                              # Test suite
```

### How It's Invoked

```bash
# Clone and install
git clone https://github.com/jramos/agent-self-evolution.git
cd agent-self-evolution
uv sync

# Point at hermes-agent repo (auto-detected from ~/.hermes/hermes-agent or env var)
export HERMES_AGENT_REPO=~/.hermes/hermes-agent

# Phase 1: Evolve a skill
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source synthetic         # or: sessiondb, golden, auto

# Phase 2: Evolve tool descriptions
python -m evolution.tools.evolve_tool_descriptions \
    --iterations 5 \
    --benchmark-gate tblite-fast

# Phase 3: Evolve a system prompt section
python -m evolution.prompts.evolve_prompt_section \
    --section MEMORY_GUIDANCE \
    --iterations 5

# Phase 4: Repair a broken tool from its failing test, gated (shipped)
python -m evolution.code.evolve_code \
    --repo ~/.hermes/hermes-agent \
    --tool tools/file_tools.py \
    --visible-test tests/tools/test_file_tools_a.py \
    --holdout-test tests/tools/test_file_tools_b.py

# All commands output a PR branch + summary against hermes-agent. Human merges.
```

### Relationship to hermes-agent

**hermes-agent-self-evolution operates ON hermes-agent, not inside it.** Zero changes to the agent repo are needed. It reads from the hermes-agent codebase and writes evolved versions to git branches, creating PRs for human review.

| hermes-agent Component | How Self-Evolution Uses It |
|------------------------|-------------------|
| `batch_runner.py` | Run agent on eval tasks in parallel |
| `environments/benchmarks/tblite/` | Benchmark gating |
| `environments/benchmarks/yc_bench/` | Coherence checks |
| `hermes_state.py` (SessionDB) | Mine real usage for eval data |
| `agent/prompt_builder.py` | Read current prompt sections (read-only) |
| `tools/registry.py` | Read current tool descriptions (read-only) |
| `skills/` directory | Read current skills, write evolved versions to branch |

---

## Execution Plan

### How Phases Work

Phases are sequential — each one builds on infrastructure from the previous one and must prove itself before we move on. The flow is:

```
Phase 1 ──► Validation Gate ──► Phase 2 ──► Validation Gate ──► Phase 3 ──► ...
  Build       "Did it actually       Build       "Did it work         Build
  & test       make things            & test       without breaking     & test
               better?"                            anything?"
```

**Between every phase:**
1. Run full benchmark suite (TBLite + YC-Bench fast_test) to establish a new baseline
2. Review all evolved artifacts — are the changes sensible to a human?
3. Merge the proven improvements via PR
4. Retrospective: what worked, what didn't, adjust approach for next phase

**Each phase has three stages:**
- **Build** (~1-2 weeks): Write the optimization infrastructure for that tier
- **Run** (~1 week): Execute optimization on real targets, iterate on eval datasets
- **Validate** (~1 week): Benchmark, review, merge. Decide if results justify moving to next phase

If a phase doesn't produce meaningful improvements (evolved variants aren't better than baseline), we stop and reassess before moving on. No point optimizing tool descriptions if we can't even improve skills.

### Timeline Overview

| Phase | What | Duration | Depends On | Gate to Next |
|-------|------|----------|-----------|-------------|
| **Phase 1** | Skill evolution | 3-4 weeks | Nothing — starts here | ≥1 skill measurably improved, no benchmark regression |
| **Phase 2** | Tool descriptions | 2-3 weeks | Phase 1 infra (GEPA runner, eval framework) | Tool selection accuracy improved, no benchmark regression |
| **Phase 3** | System prompt | 2-3 weeks | Phase 1-2 infra + validated benchmark gating | Behavioral tests pass, benchmarks hold or improve |
| **Phase 4** | Code repair (test-feedback, gated) | shipped | the deploy gate + worktree isolation | Oracle-matching fix on real bugs (~0.60–0.74) |
| **Phase 5** | Continuous loop (propose-only sentinel) | shipped | Phase 4 + the sentinel | Ranked real-bug queue; human triggers any repair |

**Total: ~13-17 weeks if all phases prove valuable.** But we may stop at Phase 1 or 2 if the returns diminish — no obligation to do all five.

### Detailed Phase Breakdown

---

### Phase 1: Skill Evolution via DSPy+GEPA (Core Capability)

**Status:** ✅ Complete. See "Deviations from plan" at the end of this section.

**Goal:** The agent can optimize any SKILL.md file by running it through GEPA.

**Week 1-2 (Build):**
- Install DSPy + GEPA, verify they work in Hermes' .venv
- Build the skill-as-DSPy-module wrapper (takes a SKILL.md → DSPy module)
- Build the eval dataset generator (strong model reads skill → generates test cases)
- Build the GEPA optimization runner (wraps dspy.GEPA with Hermes config)
- Unit tests for all components

**Week 2-3 (Run):**
- Pick 2-3 target skills: `github-code-review`, `systematic-debugging`, `arxiv`
- Generate eval datasets for each (15-30 examples per skill)
- Run GEPA optimization (5-10 iterations per skill)
- Compare baseline vs evolved on holdout set
- Iterate on eval dataset quality if results are noisy

**Week 3-4 (Validate):**
- Run TBLite + YC-Bench fast_test with evolved skills vs baseline
- Human review of all evolved skill diffs — do the changes make sense?
- Create PRs for improvements that pass all gates
- Document what worked and what didn't

**Done when:**
- ≥1 skill shows measurable improvement on its eval dataset (≥10% score increase)
- No benchmark regression (TBLite score holds within 2%)
- The evolved skill diff reads sensibly to a human reviewer
- The optimization pipeline is reusable (can point it at any skill and run)

**What to build:**
1. **Skill-as-DSPy-Module wrapper** — Takes a SKILL.md, creates a DSPy module that:
   - Injects the skill text as the system prompt
   - Runs the agent on a test task
   - Returns the result for scoring

2. **Evaluation dataset builder** — Creates train/val/holdout splits from multiple sources:

   **Source A: Synthetic generation (primary, bootstrapping)**
   Use a strong model (e.g., Claude Opus) to generate test cases for a skill:
   - Read the skill file → understand what it does
   - Generate 15-30 realistic (task_input, expected_behavior) pairs
   - Expected_behavior is a rubric, not exact text — e.g., "should identify the SQL injection on line 42" not "output this exact string"
   - Split: 10 train / 5 val / 5-10 holdout
   - GEPA works with as few as 3 examples, so this is sufficient to start

   **Source B: SessionDB mining (real usage, LLM-as-judge scored)**
   - Query SessionDB for sessions where the skill was loaded (search for skill name in messages)
   - Extract the task the user gave and the agent's full response
   - Use LLM-as-judge to score each (task, response) pair on a rubric
   - High-scoring pairs become "good" examples; low-scoring pairs become failure cases for GEPA's reflective analysis
   - This improves over time as more real usage accumulates

   **Source C: Hand-curated golden sets (optional, high-value skills)**
   - Manually written test cases with expected outputs
   - Stored as JSONL in `~/.hermes/evolution/datasets/<skill-name>/golden.jsonl`
   - Highest quality signal but requires manual effort — reserve for critical skills

   **Source D: Skill-specific auto-evaluation (where applicable)**
   - `systematic-debugging`: Plant a bug, run the skill, check if tests pass after
   - `arxiv`: Search for known papers, check if they're found
   - `github-code-review`: Create a PR with planted issues, check if they're caught
   - Not all skills have natural auto-eval — this is a bonus, not a requirement

   **Scoring: LLM-as-judge with rubrics**
   For most skills, there's no binary right/wrong — quality is subjective. The fitness function uses an LLM judge that scores on a rubric:
   - Did the agent follow the skill's procedure? (0-1)
   - Was the output correct/useful? (0-1)
   - Was it concise (within token budget)? (0-1)
   - Rubrics are skill-specific and stored alongside the eval dataset

3. **GEPA optimization runner** — Wraps `dspy.GEPA` with Hermes-specific config:
   - Uses batch_runner for parallel evaluation
   - Captures execution traces (trajectories) for GEPA's reflective analysis
   - Saves snapshots for pause/resume

4. **Comparison & deployment** — Side-by-side evaluation:
   - Runs baseline vs optimized on held-out test set
   - Shows diff of what changed
   - Commits improved version with evolution metadata

**CLI interface:**
```bash
# Evolve a skill with auto-generated eval data from session history
hermes evolve skill github-code-review --iterations 10

# Evolve with a custom evaluation dataset
hermes evolve skill arxiv --dataset eval_tasks.jsonl --iterations 5

# Compare baseline vs evolved
hermes evolve compare github-code-review --version latest

# Deploy evolved version
hermes evolve deploy github-code-review --version 3
```

**Or as agent tool calls:**
```
The agent can self-invoke optimization:
"I notice this skill could be improved. Let me run GEPA optimization on it."
→ Uses execute_code to run DSPy+GEPA
→ Evaluates results
→ Proposes the improved skill for human approval
```

**Deviations from plan (Phase 1):**

1. **Benchmark gating (TBLite/YC-Bench) was not wired as a built-in.** The plan made "TBLite within 2%" the next-phase gate. The framework went framework-agnostic (any agent that emits `SKILL.md`), so the Hermes-specific benchmark suites no longer apply uniformly. The built-in deploy gate is a paired-bootstrap CI on a held-out split of the skill's own eval set (`evolution/core/stats.py` + `_check_growth_with_quality_gate`). For users who want a benchmark-style gate, `--benchmark-cmd "<shell command>"` runs an arbitrary user-provided command after the built-in gate passes; nonzero exit flips the decision to reject with `reason="benchmark_failed"` and a `benchmark` block in `gate_decision.json`. This restores the plan author's "benchmarks as gates" vision in a framework-agnostic way: TBLite users wire `--benchmark-cmd "tblite-runner --threshold 0.02"`, no built-in coupling.
2. **Eval datasets are larger than the planned 15–30 examples.** `eval_dataset_size=150` is the current default, sized so the bootstrap CI half-width can detect ±2% effects. Rationale and the supporting study live in `reports/calibration_findings.md`.
3. **Source D (skill-specific auto-evaluation) is built for tool descriptions and for `systematic-debugging`; remaining skill-side suites still deferred.** `evolution/validation/closed_loop` drives a real Hermes Agent via `hermes -z` against a JSONL task suite, scores actual tool-selection behavior with both expected and forbidden tools, compares baseline vs. evolved with a two-condition decision rule (no aggregate regression + no per-task loss unless wins offset 2:1). v1 ships a 5-task suite for `patch` (tool-side). Skill-side now ships `systematic-debugging` via a new `SkillFileInstaller` plus a `test_command` verdict mode (exit code of a planted test replaces tool-call membership scoring); same harness, different artifact + scoring shape. Remaining skill-side equivalents (known-paper recall for `arxiv`, planted-issue for `github-code-review`) remain deferred — same harness, different verdict mechanisms (rubric-based for arxiv, planted-issue identification for code-review).
4. **The selection and gating layer is much thicker than the plan anticipated.** Knee-point Pareto selection, a budget-aware reflection-prompt proposer with growth/balanced/compression modes, paired-bootstrap CI, non-inferiority gate option, and the `gate_decision.json` v4 audit schema were added on top of raw GEPA. Rationale: raw "ship the best valset candidate" overfits at small N. See `docs/framework_advantages.md`.
5. **CLI shipped as `python -m evolution.skills.evolve_skill`, not a `hermes evolve skill` subcommand.** The framework rebranded out of Hermes-only branding so it could target any agent framework whose skills are `SKILL.md` files.
6. **`--run-tests` (target-repo pytest as a hard gate) was specified but never wired and has been removed.** Skill text doesn't touch target-repo Python, so coupling the deploy decision to the target repo's test suite would gate on unrelated signal. The CLI flag set a config field that no code read. The unused `EvolutionConfig.run_pytest`, `run_tblite`, `tblite_regression_threshold` fields and `ConstraintValidator.run_test_suite()` method were removed alongside the flag.

### Phase 2: Tool Description Optimization

**Goal:** Optimize the natural language descriptions in tool schemas so the agent picks the right tools more reliably and uses them correctly.

**Status:** ✅ Complete. See "Deviations from plan" at the end of this section.

**Prerequisite:** Phase 1 gate passed — GEPA optimization loop proven to work on skills.

**Week 1 (Build):** Adapt Phase 1's GEPA runner for tool descriptions. Build tool selection evaluator and synthetic dataset generator. The hard part is cross-tool evaluation — ensuring one tool's improvement doesn't steal from another.

**Week 2 (Run):** Generate tool selection dataset (~200-400 triples). Run GEPA on all tool descriptions simultaneously. Mine SessionDB for misselection patterns.

**Week 3 (Validate):** Benchmark gate. Human review of evolved descriptions — do they still accurately describe the tools? PR.

**Done when:**
- Tool selection accuracy improves on holdout set (≥5% improvement)
- No individual tool's selection rate regresses
- Benchmarks hold (TBLite within 2%)
- Evolved descriptions are factually accurate and ≤500 chars

**What gets evolved:**
Tool descriptions are hardcoded string constants in `tools/*.py` files, registered via `registry.register()`. Each tool has:
- A top-level `description` field (what the tool does, when to use it, behavioral guidance)
- Per-parameter `description` fields (what each parameter means, valid values)
- Some tools have separate description constants (e.g., `TERMINAL_TOOL_DESCRIPTION`)

These descriptions are sent with every API call as part of the tool schema — every extra character multiplies across the entire conversation.

**What to build:**

1. **Tool selection evaluator** — Given a task description, does the agent pick the right tool?
   - Build a dataset of (task_description, correct_tool, correct_params) triples
   - Example: "find all Python files containing 'import os'" → `search_files` (not `terminal(grep)`)
   - Example: "read lines 50-100 of config.py" → `read_file` (not `terminal(cat)`)
   - Score: tool_selection_accuracy + parameter_correctness

2. **Description optimizer** — GEPA evolves description text to improve selection accuracy
   - Wrap each tool description as a DSPy Signature parameter
   - Mutate descriptions, evaluate on tool selection dataset
   - GEPA reads traces of WRONG tool selections to understand why the agent was confused

3. **Cross-tool evaluation** — Ensure improving one description doesn't hurt others
   - Always evaluate ALL tool descriptions together (not in isolation)
   - Fitness function penalizes regressions on any tool's selection rate
   - This prevents a `search_files` description from "stealing" selections from `read_file`

**Evaluation data sources:**

   **Source A: Synthetic tool selection dataset**
   Generate (task, correct_tool, correct_params) triples using a strong model:
   - For each tool, generate 10-20 tasks where that tool is clearly the right choice
   - Include 10-20 "confusing" tasks where two tools could work but one is better
   - Include 10 tasks where the agent should use NO tool (just respond directly)
   - Total: ~200-400 triples, split 60/20/20 train/val/holdout

   **Source B: SessionDB mining — tool selection patterns**
   - Find conversations where the agent used a tool
   - Identify cases where the agent used `terminal(grep)` when `search_files` was better (or similar mismatches)
   - LLM-as-judge scores whether the tool choice was optimal
   - Misselections become high-value training examples

   **Source C: Benchmark-derived tool selection**
   - Run TBLite with baseline descriptions, log every tool call
   - Identify tasks where wrong tool selection caused failures
   - These become hard examples in the eval dataset

**Constraints specific to tool descriptions:**
- Max 500 chars per tool description (sent every API call)
- Max 200 chars per parameter description
- Must remain factually accurate (can't claim a tool does something it doesn't)
- Schema structure (parameter names, types, required fields) is FROZEN — only text evolves

**Deviations from plan (Phase 2):**

1. **Per-parameter descriptions are not evolved.** Only the top-level `description` text is mutated. Per-param descriptions and parameter-correctness scoring are out of scope for now — top-level descriptions dominate both selection accuracy and per-API-call token spend.
2. **One target tool per run, not joint optimization across all tools.** The plan called for evolving every tool description simultaneously with a manifest-wide selection-rate matrix. The framework addresses the same "stealing selections" risk with a three-bucket synthetic eval (50% target-correct, 30% confusable-neighbor against an explicit declared peer, 20% regression-detection where another tool is correct), and the agent sees the full rendered manifest at eval time so cross-tool effects surface naturally.
3. **SessionDB mining is Hermes-only; benchmark-derived hard cases are deferred.** The plan listed both as additional sources beyond synthetic. SessionDB mining is now wired (`--eval-source sessiondb`) but Hermes is the only source — Claude Code and Copilot session logs don't record tool calls in a usable form. The miner re-judges each `(task, invoked_tool)` pair against the *current* manifest with a confidence-banded LLM judge; misselections at confidence ≥0.85 become flipped-label training examples, the 0.6–0.85 band is dropped as too noisy. Benchmark-derived hard cases remain deferred.
4. **Benchmark gating again not built in.** Same as Phase 1 deviation #1 — the built-in deploy gate is paired-bootstrap CI on the holdout. For benchmark-style gating users wire `--benchmark-cmd "<shell command>"`; the hook also runs on the tool path against the rendered manifest (`EVOLVED_PATH` points at the JSON).
5. **Eval triples are `(task, correct_tool)`, not `(task, correct_tool, correct_params)`.** Schema structure is frozen by design; parameter correctness is a separate concern from description quality.
6. **Input surface is broader than planned.** The plan framed Phase 2 around Hermes' `tools/registry.py`. The build supports any agent that can emit an MCP `list_tools()`-shape JSON manifest *and* Hermes-style Python `*_SCHEMA` declarations (via `HermesToolSource`'s AST walker, with one-hop `Name`-reference resolution for cases like `TERMINAL_TOOL_DESCRIPTION`). This is a scope expansion, not a contraction.
7. **N=2 saturated baselines.** Weak-target hunt ran `evolve_tool` against `write_file` (98.8–99.2% holdout, 3 seeds, 1×/3× iter) and `search_files` (98.6% holdout). Both runs produced evolved descriptions byte-identical to the baseline — the knee-point picker correctly reverts to the unchanged baseline when GEPA's variants tie. The framework's tool-description pipeline is regression-catching, not improvement-finding, on these hand-tuned descriptions.
8. **Closed-loop signal can flow into reflection but doesn't change selection on saturated baselines.** The `--closed-loop-during-evolution` flag plumbs `ValidationReport`s into the GEPA reflection LM's feedback channel via the existing 5-arg metric protocol, opt-in, saturation-gated. Verified end-to-end on `write_file`: closed-loop fired (file mutated + restored), the reflection LM saw the verdict, GEPA still selected the baseline byte-for-byte. The bottleneck sits upstream of reflection — GEPA's `sum(judge_scores)` acceptance rule ties when every candidate hits 1.0 on a saturated minibatch. Extending the Pareto frontier into behavioral space (closed-loop tasks as additional training-set instances with their own per-instance scores so a candidate can stay on the frontier by winning behavioral tasks) is the structural direction that would address this; the cache + renderer added here are the natural building blocks for that work.

   **Follow-up — Path F (saturation pre-flight) addresses the user-visible symptom, not the underlying mechanism.** A separate investigation (`reports/pareto_frontier_feasibility.md`, two spike runs) confirmed the deviation's diagnosis and reframed it: the bottleneck isn't frontier shape, it's GEPA's stochastic small-minibatch `sum()` acceptance gate discarding per-instance signal before it can move selection. Path F (`evolution/core/saturation_check.py`) ships the user-visible fix — detect the saturated case before GEPA starts, render a panel explaining why no improvement is possible, default-deny in non-interactive contexts. This prevents the wasted-budget UX without solving the mechanism gap. The mechanism-side fix (Pareto-dominance acceptance, larger minibatch, or stratified sampling) is tracked as "Path D/E/C" in the feasibility report and remains future work.

### Phase 3: System Prompt Evolution

**Goal:** Optimize the sections of the system prompt that guide agent behavior.

**Status:** ✅ Complete (MEMORY_GUIDANCE proof point). See "Deviations from plan" at the end of this section.

**Prerequisite:** Phase 2 gate passed — benchmark gating validated, GEPA producing sensible text mutations.

**Week 1 (Build):** Build section-as-DSPy-parameter wrapper for the 5 evolvable prompt sections. Build behavioral test suite generator. This is the riskiest tier so far — system prompt changes affect everything.

**Week 2 (Run):** Generate behavioral test scenarios (~60-80 total across all sections). Run GEPA on each section independently first, then jointly. Run benchmarks after each optimization round.

**Week 2-3 (Validate):** Full benchmark suite (TBLite + YC-Bench). Extra scrutiny here — system prompt changes have the widest blast radius. Multiple human reviewers if possible.

**Done when:**
- Behavioral test scores improve (≥10% on targeted sections)
- Benchmarks hold or improve (zero tolerance for regression here)
- The agent's personality/tone hasn't drifted noticeably
- Prompt stays within caching boundaries

**What gets evolved:**
The system prompt is assembled in `run_agent.py` / `agent/prompt_builder.py` from 8 distinct sections:

| Section | Location | What It Does | Evolvable? |
|---------|----------|-------------|-----------|
| `DEFAULT_AGENT_IDENTITY` | prompt_builder.py | Core persona, behavioral traits | ✅ Yes — tone, priorities, approach |
| `MEMORY_GUIDANCE` | prompt_builder.py | How to use persistent memory | ✅ Yes — when to save, what to save |
| `SESSION_SEARCH_GUIDANCE` | prompt_builder.py | When to search past sessions | ✅ Yes — trigger conditions |
| `SKILLS_GUIDANCE` | prompt_builder.py | When to save/load skills | ✅ Yes — trigger conditions |
| `PLATFORM_HINTS` | prompt_builder.py | Per-platform formatting guidance | ✅ Yes — per platform |
| Memory block | memory_store.py | User's actual memories | ❌ No — user data |
| Skills index | prompt_builder.py | Auto-generated skill list | ❌ No — auto-generated |
| Context files | prompt_builder.py | AGENTS.md, .cursorrules | ❌ No — project-specific |

**What to build:**

1. **Section-as-DSPy-parameter wrapper** — Each evolvable section becomes a DSPy Signature field
   - The optimizer can mutate each section independently
   - Sections are evaluated together (the full system prompt matters, not individual sections)

2. **Behavioral evaluator** — Does the agent behave correctly with this system prompt?
   - Measure: tool usage patterns, response quality, memory usage, skill loading
   - Use batch_runner to run the agent on diverse tasks with the evolved prompt

3. **Benchmark-gated validation** — Evolved prompts must not regress on benchmarks
   - Run TBLite (fast, ~1-2 hours) as a regression check
   - If TBLite score drops, reject the variant regardless of other metrics

**Evaluation data sources:**

   **Source A: Behavioral test suite (synthetic)**
   Generate scenarios that test specific prompt sections:
   - Memory guidance: "Does the agent save important user preferences?" (10 scenarios)
   - Session search: "Does the agent search history when the user says 'like last time'?" (10 scenarios)
   - Skills guidance: "Does the agent load relevant skills before starting?" (10 scenarios)
   - Identity: "Is the response helpful, direct, and not overly verbose?" (20 scenarios)
   - Platform hints: "Does CLI output avoid markdown? Does Telegram use formatting?" (10 per platform)

   **Source B: Benchmark scores as fitness signal**
   - TBLite (100 tasks, ~1-2 hours, binary pass/fail) — primary regression check
   - YC-Bench fast_test preset (~50 turns, composite score) — tests long-term coherence
   - These don't measure specific prompt sections, but they catch broad regressions
   - A prompt variant that scores higher on behavioral tests but lower on TBLite is rejected

   **Source C: SessionDB — behavioral pattern mining**
   - Find sessions where the agent failed to search memory when it should have
   - Find sessions where the agent was too verbose or used wrong formatting
   - These become targeted test cases for the relevant prompt section

**Constraints specific to system prompt sections:**
- Each section must not exceed its current size by >20% (prevents prompt bloat)
- Total system prompt must stay under the model's prompt caching boundary
- Identity section must retain core traits (helpful, direct, admits uncertainty)
- Platform hints must remain platform-accurate (don't tell Telegram to use ANSI codes)

**Deviations from plan (Phase 3):**

1. **Integration is in-place splice-and-restore, not an env-var hook or a plugin.** The design's primary path routed candidate overrides through an upstream `HERMES_PROMPT_OVERRIDES_JSON` env var; that hook was not accepted upstream, so depending on it would make the framework a local-only patch that silently no-ops on any hermes pull. A plugin alternative was ruled out as non-viable: consumers bind the constants at import time (`from agent.prompt_builder import MEMORY_GUIDANCE` in `run_agent.py` and `agent/system_prompt.py`), so a plugin's `register()` runs too late to reach them. Phase 3 instead splices the candidate directly into `agent/prompt_builder.py` (byte-precise AST replacement via `repr()`, parse-checked) and restores from an atomic backup, reusing Phase 2's `ClosedLoopValidator` flock + sha-drift + stale-backup machinery. No upstream dependency; runs against stock Hermes. This also **collapsed the planned parallel `PromptSectionValidator`** into a small `HermesPromptSectionInstaller` (an `ArtifactInstaller`) plus a one-method Layer-2 hook on the shared validator — less code than the design called for.

2. **One target section per run; MEMORY_GUIDANCE is the only proof point.** Joint multi-section optimization and identity/persona evolution are deferred — joint runs carry Phase 2's "stealing selections" risk, and `DEFAULT_AGENT_IDENTITY` has no tool-call anchor for the verdict. The `PromptSource` abstraction supports the other string sections (`SKILLS_GUIDANCE`, `SESSION_SEARCH_GUIDANCE`, etc.) with no refactor; dict-typed sections like `PLATFORM_HINTS` are out of scope for v1 (string constants only).

3. **Verdict is compound (tool-membership + LLM content judge), threaded per-task.** Layer 1 is the Phase 2 expected/forbidden rule on whether `memory` was invoked; Layer 2 is an LLM judge scoring the saved content against each task's `expected_save_content` rubric. The validator builds the judge per-task (a factory) so the content judge sees the task's rubric and message — a fixed global judge couldn't. Note the real Hermes `memory` tool actions are **`add`/`replace`** (content-bearing), not `save`; the full set is add/replace/remove/read.

4. **Eval suite ships as a curated 12-task golden set, not 50 synthetic + 10 golden.** A hand-authored `memory_guidance.jsonl` spans the five categories (save-preference, save-correction, dont-save-task-progress, dont-save-completed-work-log, declarative-vs-imperative). The synthetic generator (`build_memory_guidance_dataset`) is built and unit-tested, but full synthetic expansion via a funded generation run is deferred — curation gives a higher-signal first suite and avoids upfront generation spend.

5. **PR automation is deferred for prompt sections.** `create_pr` atomically copies a full evolved artifact over `origin/<base>`'s file; deriving an evolved `prompt_builder.py` from the local checkout would carry the unmerged override-hook commit into the PR diff. `--create-pr` is accepted but records a skipped block; the deploy path is `--apply` (writes the evolved section into the live file) plus a manual PR. A section-scoped PR path (splice into `origin/<base>`'s file, not the local one) is future work.

6. **The shared closed-loop runner had to be rebuilt for current Hermes.** Surfaced by the Phase 3 end-to-end smoke: `hermes -z` one-shot mode is now ephemeral — it prints only the final response and no longer writes `session_*.json`; sessions persist to a SQLite `state.db` in `HERMES_HOME`. `HermesAgentRunner` globbed for the obsolete JSON files, so **every** closed-loop run had been silently abstaining ("no session JSON") across Phases 1–3. The runner now reads the most-recent session's messages from `state.db` (the `tool_calls` column carries the same OpenAI-nested shape the extractors already parse). This is a shared-infrastructure fix that unblocks all closed-loop validation, not just prompts.

7. **Behavioral eval is serialized, and agent-subprocess cost is invisible to the budget cap.** Because every candidate is spliced into one shared `prompt_builder.py`, the GEPA inner-loop scorer serializes splice+run under a lock (DSPy's evaluator is multi-threaded; the shared file is not). Per-section closed-loop is therefore effectively serial — an accepted v1 cost of the splice-and-restore model. The agent's own LM spend happens inside the `hermes` child process, invisible to the in-process cost ledger, so `--max-cost-usd` bounds only judge + reflection + passthrough spend; `sessions.actual_cost_usd` in `state.db` could close that gap later.

8. **Saturation default-deny confirmed on a capable agent; a demonstrated improvement required an adversarial baseline.** With `gpt-5.4-mini`, both the live `MEMORY_GUIDANCE` and a *passively*-weakened baseline scored 1.0 / 6 holdout — `no_headroom`, correctly default-denied. This matches Phase 2's "regression-catching, not improvement-finding on tuned artifacts" finding and the binary model-tier effect (a capable agent saves correctly regardless of vague guidance). A real mutation was demonstrated only by *actively misdirecting* the baseline: an adversarial "never proactively save" section scored 0.67, GEPA's reflective proposer inverted it to restore proactive saving (and made it shorter), and the deploy gate measured **0.67 → 1.00, 2 wins / 0 losses → deploy**. The `--baseline-override-file` flag enables this ablation (and regression-injection testing generally) without mutating the live section.

9. **Benchmark gating again not built in (same as Phases 1–2).** The built-in deploy gate is paired-bootstrap CI plus the dual-condition rule on the holdout; `--benchmark-cmd` remains the external-benchmark hook. TBLite / YC-Bench wiring is left to the user's `--benchmark-cmd`.

### Phase 4: Code Repair (iterative test-feedback, gated)

> **Status — a single-proposer test-feedback loop shipped; the Darwinian Evolver was
> never run.** The week-by-week Darwinian-Evolver plan in this subsection is
> **unbuilt** — what actually shipped (`evolution/code/`) is iterative test-feedback
> repair: a capable proposer rewrites the whole file from the failing test's output, in
> a throwaway worktree, accepted only through the deploy gate (surface freeze, file
> scope, held-out split, regression floor). That loop works (deploy-reachable
> ~0.60–0.74). Now settled by experiment: whether the population-based evolutionary search
> the authors intended would beat this loop on the ~30% of bugs it fails — that
> head-to-head has now been **run** (item 12): population search was strictly dominated by plain best-of-N resampling (4/23 vs 6/23), so "the evolver
> is overkill here" is a tested result, not an assumption. For the authoritative current status (the leakage
> reframe, the difficulty curve) see **Forward Roadmap item 12** and
> [the findings report](reports/asymmetry_findings.md). The original plan is kept below as founding intent.

**Goal:** Evolve tool implementation code for better performance and fewer bugs.

**Prerequisite:** Phases 1-3 complete — strong evaluation pipeline, validated benchmark gating, confidence in the optimization loop.

**Week 1-2 (Build):** Set up Darwinian Evolver as external CLI. Build code-as-organism wrapper mapping tool files to GitBasedOrganism. Build composite fitness function (pytest + benchmarks + bug reproduction). This phase uses a different engine (Darwinian Evolver instead of DSPy+GEPA) so there's new infrastructure to build.

**Week 2-3 (Run):** Start with known bugs from GitHub issues — create reproduction scripts, run evolution to find fixes. Then try edge case hardening on 1-2 tools (e.g., `file_tools.py`, `search_files`).

**Week 3-4 (Validate):** Full test suite + full benchmark suite (including TerminalBench2 for thorough validation). Strictest human review — every line of evolved code reviewed before merge.

**Done when:**
- ≥1 known bug fixed by evolution (validated by reproduction script)
- Full test suite passes (2550+ tests)
- Full benchmark suite holds (TBLite + TerminalBench2 + YC-Bench)
- No function signatures or registry calls changed
- Human reviewer approves all code changes

**What gets evolved:**
Actual Python source code in `tools/*.py` files. This is the highest-risk tier — code changes can break everything.

**What to build:**

1. **Code-as-organism wrapper** — Maps tool source files to Darwinian Evolver's `GitBasedOrganism`
   - Each tool file is a separate organism
   - Mutations are proposed by the LLM based on specific failure cases
   - All mutations are committed to a git branch for traceability

2. **Test-driven fitness function** — Composite score from multiple signals:
   - pytest results (hard gate — must pass 100%)
   - Benchmark scores (TBLite pass rate)
   - Specific failure case resolution (did the mutation fix the bug it targeted?)
   - Code quality heuristics (no regressions in error handling, no removed safety checks)

3. **Safety guardrails** — Strictest of all tiers:
   - Full test suite must pass
   - No changes to function signatures (would break callers)
   - No changes to registry.register() calls (would break tool discovery)
   - No removal of error handling or safety checks
   - Human review required on every PR

**Evaluation data sources:**

   **Source A: pytest suite (2550+ tests, primary gate)**
   - Every code mutation must pass the full test suite
   - Test failures = immediate rejection, no exceptions
   - This is the hard floor — it prevents regressions

   **Source B: Benchmark scores (broad capability check)**
   - Run TBLite with evolved code to verify tool implementations still work end-to-end
   - TerminalBench2 for thorough validation (89 tasks, but expensive — use selectively)
   - YC-Bench for long-horizon coherence (does the agent still handle 200-turn sessions?)

   **Source C: Known bug reproduction datasets**
   - Collect GitHub issues that report tool bugs
   - Create reproduction scripts that trigger the bug
   - Fitness: does the evolved code fix the bug while passing all tests?
   - This is the most targeted and efficient use of code evolution

   **Source D: Edge case generation**
   - Use a strong model to generate adversarial inputs for each tool
   - Example: `read_file` with symlinks, binary files, huge files, missing files, permission errors
   - Example: `search_files` with regex edge cases, unicode, very large repos
   - Score: does the tool handle all edge cases gracefully?

**Constraints specific to code evolution:**
- Full test suite (2550+ tests) must pass — zero tolerance
- Function signatures frozen (no breaking API changes)
- registry.register() calls frozen (no tool discovery changes)
- Error handling coverage must not decrease
- Darwinian Evolver runs as external CLI only (AGPL v3)
- All PRs require human review — no auto-merge for code changes

### Phase 5: Continuous Self-Improvement Loop

**Goal:** The agent automatically identifies its weakest areas and improves them over time.

**Prerequisite:** Phases 1-4 proven — manual optimization works reliably for skills, tools, prompts, and code. Now we automate it.

**Week 1 (Build):** Build performance monitor (tracks skill success rates, tool selection accuracy, benchmark scores over time). Build auto-triage logic (ranks optimization targets by impact × frequency). Wire up to Hermes cron scheduler.

**Week 2 (Deploy & Monitor):** Set up weekly benchmark runs via cron. Set up threshold-triggered optimization (when a skill's failure rate exceeds X%, auto-trigger GEPA). All automated PRs still require human merge.

**Done when:**
- Weekly benchmark runs execute unattended and report scores
- Auto-triage correctly identifies underperforming skills
- At least one optimization cycle runs end-to-end (detect problem → optimize → PR) without manual intervention
- Human still reviews and merges every PR — this phase automates detection and optimization, not deployment

**What to build:**

1. **Performance monitor** — Track metrics from real usage:
   - Per-skill success rates (from SessionDB — was the skill loaded? did the task succeed?)
   - Tool selection accuracy (from trajectories — did the agent pick the right tools?)
   - Benchmark scores over time (periodic TBLite + YC-Bench runs)
   - User corrections (when the user says "no, use X instead" — that's a signal)

2. **Auto-triage** — Identify what to optimize next:
   - Skills with declining success rates or high failure rates
   - Tools that are frequently misselected
   - Benchmark categories with low pass rates
   - Rank by (potential improvement × usage frequency)

3. **Scheduled optimization** — Cron job pipeline:
   - Weekly: Run TBLite + YC-Bench fast_test, log scores
   - When scores drop or skill failure rate exceeds threshold: trigger GEPA optimization
   - Generate PR with evolved improvements
   - Notify for human review

4. **Feedback loop** — Real usage improves evaluation datasets:
   - User corrections are logged and added to eval datasets
   - High-quality sessions become positive examples
   - Failed sessions become failure cases for GEPA's reflective analysis
   - Evaluation datasets grow organically over time

---

## Benchmarks as Fitness Signals

The three existing benchmarks serve different roles in the optimization pipeline:

| Benchmark | What It Tests | Speed | Cost | Role in Self-Improvement |
|-----------|-------------|-------|------|------------------------|
| **TBLite** | Coding/sysadmin (100 tasks, calibrated difficulty) | ~1-2 hours | ~$20-50 | **Primary regression gate** — fast enough to run on every candidate |
| **TerminalBench2** | Coding/sysadmin (89 harder tasks, Docker sandboxes) | ~2-4 hours | ~$50-200 | **Thorough validation** — run on final candidates before PR |
| **YC-Bench** | Long-horizon strategic coherence (100-500 turns) | ~3-6 hours | ~$50-200 | **Coherence check** — ensures evolved prompts don't break multi-turn behavior |

**How benchmarks fit into the optimization loop:**

```
Candidate Variant
    │
    ├──► pytest (must pass 100%) ────────── GATE 1: functional correctness
    │
    ├──► TBLite fast subset (20 tasks) ──── GATE 2: quick capability check (~20 min)
    │
    ├──► Task-specific eval dataset ──────── FITNESS: skill/tool/prompt quality score
    │
    ▼
Top Candidates Only (top 3)
    │
    ├──► Full TBLite (100 tasks) ─────────── GATE 3: thorough regression check
    │
    ├──► YC-Bench fast_test ──────────────── GATE 4: coherence check
    │
    ▼
Best Candidate → PR with full metrics
```

**Key principle:** Benchmarks are GATES, not fitness functions. The fitness function is task-specific (did the skill/tool/prompt do its job better?). Benchmarks ensure the improvement didn't break something else. A variant that improves skill quality by 20% but drops TBLite by 5% is REJECTED.

---

## Constraints & Guardrails

Every candidate variant must pass ALL of these before it can be considered valid. Variants that fail any constraint are discarded — GEPA/MIPROv2 never see them as successful.

### 1. Full Test Suite
```
python -m pytest tests/ -q  # Must pass 100% — zero tolerance
```
Every evolved variant (skill text, tool description, code) triggers the full test suite. If any test fails, the variant is rejected. This is the hard floor — nothing ships that breaks existing functionality.

### 2. Character/Token Limits
Evolved text must stay within strict size budgets:

| Target | Max Size | Why |
|--------|----------|-----|
| Skill files (SKILL.md) | Configurable per skill, default 15KB | Skills are injected as user messages — bloated skills waste context window |
| Tool descriptions | 500 chars | Tool schemas are sent every turn — every extra char multiplies across the entire conversation |
| System prompt sections | Must not exceed current section size by >20% | Prevents prompt bloat that degrades model attention and increases cost |

The optimizer's fitness function applies a **length penalty** — variants that approach the limit get scored lower even if they're otherwise better. This prevents evolutionary drift toward verbose solutions.

### 3. Prompt Caching Compatibility
Hermes relies on prompt caching to keep costs manageable. Evolved content must not break this:

- **Skills**: Injected as user messages at conversation start. Evolved skills are deployed as new versions — they take effect on NEW sessions only, never mid-conversation.
- **Tool descriptions**: Part of the tool schema sent with every API call. Changes take effect on next session start. Schema structure (parameter names, types) must NOT change — only the description text.
- **System prompt sections**: Rebuilt once at session start. Evolved sections deploy as config updates, applied on next session. No mid-session prompt rebuilds.

**Rule: No evolved content is ever hot-swapped into an active conversation.** All changes take effect on the next fresh session.

### 4. Semantic Preservation
The optimizer must preserve the core behavior/intent of what it's evolving:

- A skill for "GitHub code review" must still perform code reviews, not drift into something else
- Tool descriptions must still accurately describe what the tool does
- System prompt sections must maintain their functional role

This is enforced by including **semantic similarity checks** in the fitness function — the evolved text is compared against the original to ensure it hasn't drifted too far in meaning, only improved in effectiveness.

### 5. Deployment via PR (Never Direct Commit)
All evolved changes go through a pull request:

```bash
git checkout -b evolve/<target>-<timestamp>
# Apply evolved changes
git add <files>
git commit -m "evolve: <target> — score improved X% → Y%

Optimizer: GEPA (N iterations, M candidates evaluated)
Eval dataset: <dataset name> (K examples)
Before: <baseline score>
After: <evolved score>
Holdout: <holdout score>"
git push -u origin evolve/<target>-<timestamp>
gh pr create --title "evolve: <target>" --body "<metrics, diff, comparison>"
```

The PR body includes:
- Before/after scores on train, validation, AND holdout sets
- The full diff of what changed
- Cost of the optimization run
- Any constraint violations that were caught and rejected during evolution

---

## Practical Considerations

### Cost
- GEPA optimization: ~$2-10 per run (depending on eval dataset size)
- Darwinian Evolver: ~$2-9 per task
- Batch evaluation: depends on number of test cases and model cost
- Recommendation: Start with small eval sets (10-20 examples), scale up for important skills

### Safety
See the **Constraints & Guardrails** section above for the full enforcement list. Summary:
- **Human approval required** — all changes deploy via PR, never direct commit
- **Full test suite gate** — zero tolerance, every variant must pass 100%
- **Character/token budgets** — prevents evolutionary bloat
- **Caching compatibility** — no mid-conversation changes, ever
- **Semantic preservation** — evolved text must not drift from its original purpose
- **Git-tracked lineage** — every evolution step is a commit, rollback is trivial
- **Holdout test sets** — separate from training data to catch overfitting

### Licensing
- DSPy: MIT ✓ (can import and integrate freely)
- GEPA: MIT ✓ (integrated into DSPy, also standalone `uv pip install gepa`)
- ~~Darwinian Evolver: AGPL v3 ⚠️~~ — **not used.** Code evolution shipped as
  test-feedback repair (no AGPL dependency), so this risk never materialized; the
  whole repo stays MIT.
- All code: MIT ✓

---

## Relationship to Existing Issues

| Issue | Status | Relationship |
|-------|--------|-------------|
| #336 (Darwinian Evolver Skill) | Open | Subsumed by Phase 3 of this plan |
| #337 (Evolutionary Self-Improvement) | Open | This plan IS the implementation of #337 |
| #339 (PR: Darwinian Evolver skill) | Open | Close — replaced by this unified approach |

---

## Open Questions

1. Should the optimization skill live in the repo (bundled) or Skills Hub (optional install)?
   - Recommendation: Core orchestration in repo, optimization engines as optional dependencies
   
2. How do we build evaluation datasets for skills that don't have much usage history?
   - Option A: LLM-generated synthetic test cases
   - Option B: Manual curation by skill authors
   - Option C: Community-contributed eval sets

3. Should evolved skills be versioned separately from the main repo?
   - Recommendation: Git branches per evolution run, merge winning variants to main

4. What's the minimum viable first target?
   - Recommendation: Pick 2-3 well-used skills with clear success metrics (e.g., arxiv paper search, github-code-review, systematic-debugging)

---

## Forward Roadmap — Signal-First Sequencing (2026-06)

Phases 1–3 shipped the optimizer, the selection/gating layer, and three
artifact tiers. The accumulated findings (saturation defaults, A/A noise
floors, the "suite states the win" result, the tier-binarity null) converge on
a single diagnosis: **the binding constraint is no longer selection or gating
machinery — it is discriminating evaluation signal.** On hand-tuned artifacts
the pipeline is regression-catching, not improvement-finding, because the
suites and judges saturate. The frontier literature agrees on the lever:
gskill's published headroom came from ~300 planted-fault, test-verified tasks
per repo, not from a better optimizer.

So this roadmap is sequenced **measurement → signal generation → honest
mechanism → autonomy**, not by artifact tier. Each item carries the evidence
it rests on and the cheapest experiment that would falsify it. Items are
funded only after their prerequisite signal is shown to exist; nothing in the
mechanism wave spends optimizer budget before its signal wave lands.

This section reframes the original Phase 4 (code evolution) and Phase 5
(continuous loop) as the probe-gated final wave — guides, not commitments. A
loop that selects bad targets or evolves against saturated suites compounds
waste; target selection and signal generation are the prerequisites the
original plan under-weighted.

### Campaign conclusion — capable-agent artifact decoupling (hardened)

The signal-first sequencing produced one decisive, multi-direction result. **On a
capable agent (gpt-5-mini class), artifact quality/correctness is decoupled from
behavior in BOTH directions:**
- *Improving* an artifact doesn't help — skills (item 5), tool descriptions
  (item 7), and turn-level sequences (item 13) are all null, including under a
  **judge-free `test_command` verdict** (so judge saturation is not the cause).
- *Degrading* an artifact doesn't hurt — a strong misdirecting write_file
  description and even an **actively-false** one ("write_file merges and preserves
  content, never deletes") each produced **0/7 caught-losses against a 0/7 A/A
  false-alarm floor**; the agent routes from the tool name + task + its own prior
  over the artifact text. The regression-detection rig itself works (a delivery
  canary separates a real behavior difference, 1.00 vs 0.00) — there is simply
  nothing to catch.

So the evolution/validation pipeline is **low-value for artifact quality on
capable agents**. Its value requires an agent whose behavior the artifact actually
drives: a **weaker/cheaper tier** (where gskill's published headroom lived; the
weak tier here abstains on the closed-loop path — a diagnosed, fixable delivery
bug) or a **novel MCP tool** whose behavior the agent can't infer from its name
(no MCP-bearing agent backend exists — a build, not a probe). The remaining items
(6/8/9/11) presuppose a discriminating artifact-quality signal this result shows
is absent on capable agents; pursue them only on a tier/surface where the coupling
exists. Durable wins from the campaign: a real skill-delivery bug fixed (the
candidate skill never reached the agent), and the decoupling thesis turned from
suspicion into a measured, both-directions certificate.

### Wave 1 — Measurement & free wins

1. **Mine the gate-decision archive for item statistics.** 336 of 350
   archived `gate_decision.json` carry paired per-example vectors (201 deploy /
   148 reject). Calibrate the synthetic saturation threshold (currently the
   uncalibrated `0.99` in `evolution/core/saturation_check.py`), emit
   v5-gate-relative error rates, and add a zero-cost overfitting-trajectory
   retrospective (val score vs discovery order, from `lineage.json`'s
   `val_subscores`) to address the known many-iteration overfitting caveat.
   Honestly scoped: only ~12 archived runs carry closed-loop data, so the
   closed-loop thresholds (`0.95` / `0.15`) are calibrated later from forward
   telemetry (item 2), not from this archive. *Pure compute, ~$1. Kill: n/a —
   even a null result is an empirical certificate for the current magic
   numbers.*

   **Status — shipped, reframed to instrumentation-first.** The archive proved
   *unable* to calibrate the thresholds, which is itself the finding: the gated
   region (baseline holdout ≥ 0.95) holds only 15 of 338 paired-vector runs,
   none with a bootstrap lower bound > 0, so the false-abort rate is 0% with a
   Wilson 95% upper bound of 20–49% — absence of evidence, not a green light.
   The circular `decision`-based definition would have inflated that to 75–87%,
   which is why the primary definition is `lower_bound > 0`.
   `scripts/analysis/calibrate_saturation.py` and
   `reports/saturation_calibration_findings.md` ship the honest certificate and
   become the live calibrator once forward data accrues. The
   overfitting-trajectory retrospective is **forward-only** — zero historical
   `lineage.json` (the feature postdates the archive), and per-candidate
   *holdout* scores are never stored, so the forward signal is "val plateaus
   before the search budget is spent," not "val climbs while holdout flattens";
   it ships as a gracefully-empty section. Closed-loop thresholds
   (`0.95` / `0.15`) remain uncalibratable until the ledger (item 2) fills.
   To actually settle the thresholds, the corpus must be *generated*, not
   waited for: the pre-flight stops gated-region runs before they reach an
   outcome, so the calibration fuel (gated-region runs that proceeded to a
   `gate_decision.json`) only comes from deliberate overrides. The recipe is a
   batch run with `--force-saturation-check` on a handful of high-baseline
   targets, then re-run `scripts/analysis/calibrate_saturation.py`.

2. **Promote "deploy the compiled floor" to a first-class gate outcome.** The
   zero-LM suite-constraint compiler captures ~79–85% of measured headroom but
   currently only prints a suggestion. When a compiled floor clears the
   noise-aware closed-loop gate, make it a deployable outcome; record
   `floor_score` and preflight closed-loop pass-rates on every run to build the
   corpus item 1 lacks. Plausibly the highest value-per-dollar item here.
   *Effort S. Kill: if the compiled floor never clears the gate on any shipped
   suite, the promotion is inert and reverts to a suggestion.*

   **Status — shipped (skills + prompts); tools deferred.**
   *Telemetry half:* `evolution/core/saturation_telemetry.py` records the
   pre-flight band plus holdout / closed-loop / floor scores on every run —
   including the abort paths the gate archive never captured — joined to each
   run by `run_id` in `output/saturation_ledger.jsonl`.
   *Gate half:* a cleared compiled floor is now a first-class **fallback** deploy
   outcome — when the GEPA candidate fails the closed-loop gate but
   `baseline + floor_text` clears the same noise-aware gate, the framework
   deploys the floor instead of rejecting (`resolve_floor_fallback` in
   `quality_gate.py`; opt-in behind `--compile-floor`; recorded as
   `deployed_artifact="compiled_floor"` + a `floor_fallback` block). The skill
   path computes the floor on a held-out split of the CL suite (the floor is
   compiled from train, scored on holdout, with GEPA's behavioral examples
   restricted to train so it never trains on the floor's holdout); the prompt
   path reuses its existing floor probe (after fixing a noise-aware
   inconsistency so the probe's verdict matches the gate). **Tools deferred:**
   the floor's patch/convention clauses are skill/prompt disciplines and don't
   map onto a tool-selection description (a tool suite yields at most the thin
   over-eagerness clause); not a char-cap reason — there is no MCP-protocol cap,
   `max_tool_desc_size=500` is a tunable framework guardrail. Default-off paths
   are byte-identical; the `resolve_floor_fallback` decision rule is unit-tested
   and the full floor-deploy path was exercised by a manual `claude -p` smoke
   (floor deploy fired: baseline 0.00 → floor 0.50, deployed `compiled_floor`).
   Note an honest asymmetry: the skill CL-primary gate requires a strict gain,
   so a deployed floor there beat baseline; the prompt no-regression gate does
   not, so "deployed floor" on the prompt path means the floor cleared the gate
   over a non-improving evolved, not necessarily a measured win over baseline.

3. **Memoize the baseline phase during search only.** `ClosedLoopValidator`
   re-runs the baseline phase on every `validate()` call; in trainset mode
   that is near-pure duplication (≈2× closed-loop savings). The memo must be
   exposed as a freshness-aware API that **never** serves a gate decision
   (see item 9) — search-phase reuse only, staleness-bounded against
   intra-run provider drift. Moved early because items 6–7's reps pay this
   cost today. *Effort S. Kill: concordance check — memoized vs fresh
   baseline phase must agree within the A/A flip floor.*

### Wave 2 — Signal generation (the bottleneck)

4. **Mine Claude Code project transcripts.** The in-code claim that Claude
   Code logs carry no tool-call data (`evolution/tools/session_mining.py`) is
   false for `~/.claude/projects/*/*.jsonl`, which carry thousands of
   `tool_use` blocks (true only for the older `history.jsonl`). Add a
   `ClaudeCodeSessionImporter` beside the Hermes one, feeding the existing
   confidence-banded misselection judge. These are judge-scored dataset
   examples, not verifiable behavioral tasks — the "use X not Y" tractability
   limit still applies. Correct the false docstring regardless of yield.
   *Effort S. Kill: $1 parse-only spike — if usable misselection/correction
   density is near zero, the importer ships only as a correction-miner.*

   **Status — investigated and DEFERRED (kill experiment fired).** A measured
   spike killed the single-call miner. (1) The only framework-evolvable tools
   are MCP descriptions, and the privacy-safe scope (a single project's own
   transcripts — never a glob across all local projects) has **~0 MCP calls**.
   (2) Claude turns are multi-step (≈14 tool calls/turn), so the judge's
   `(task, invoked_tool, manifest)` shape — judging one call with no turn
   context — produces a misselection label that **flips ~20%** of the time vs.
   judging with context: a context artifact that would poison GEPA, not a
   description defect. The roadmap's correction-mining fallback is also dead
   (~0 corrections observed). The docstring premise was itself off — it's
   accurate about `history.jsonl`; the roadmap conflated that with
   `projects/*.jsonl`. The docstring now records the negative. The salvageable
   idea is item 13 below.

5. **Claude Code skill closed-loop + planted-fault task synthesis.** Two
   halves. (a) Adapter: `claude_runner.py` never consumes the protocol's
   `skills_src` field; `hermes_runner.py` shows the staging pattern. A $1
   canary first resolves headless skill-discovery semantics. (b) Signal
   engine: a gskill-style spike generating N planted-bug, `test_command`-
   verified tasks against one target repo, discrimination-labeled. This is the
   artifact class with the strongest published headroom evidence; the
   synthetic-generation drop-rate null does not bind here (planted faults +
   real tests are a different mechanism than LM-judged dataset cases).
   *Effort M, $1 + $10 staged. Kill: <30% of generated tasks land
   discriminative against the live skill → the synthesis approach is shelved
   for that repo class.*

   **Status — a real bug found + fixed; the signal probe is a NULL.**
   *(a) Bug fixed (the durable win):* the skill closed loop never delivered the
   candidate `SKILL.md` to the agent for suites whose tasks omit `skills_src`
   (systematic_debugging) — the validator only threaded `task.skills_src`, so the
   `SkillFileInstaller`'s staged candidate never reached the sandbox. The prior
   "advanced suite saturates 5/5=5/5" result was therefore a **no-op** (no skill
   delivered), not evidence. The validator now falls back to the installer's
   staged skill; a delivery canary confirmed it (sentinel skill 1.00 vs weak skill
   0.00). *(b) Signal probe → NULL.* With the skill genuinely delivered, a gated
   difficulty probe on a capable agent (gpt-5-mini): **Gate A** — five
   genuinely-tricky hand-authored planted bugs (merge-touching-intervals,
   wrong-end LRU eviction, memo-missing-key, sliding-window off-by-one,
   float-equality) all solved **1.00**; **Gate C** — even an *actively-misdirecting*
   skill ("don't read the code, try random edits") did **not** degrade the agent:
   **1.00 vs 1.00 on every bug, zero discrimination** (Fisher p=1.0). Rig
   sensitivity was confirmed by the canary, so the null is trustworthy. A capable
   agent root-cause-debugs planted faults regardless of skill guidance — **null #1
   holds for skills via a judge-free verdict**, the third artifact class (after
   tool descriptions and turn-level sequences) where a capable agent saturates the
   quality signal. The synthesis engine (b) is **not justified**: the
   discriminative band is empty for a capable agent, so generating more bugs can't
   help. The Claude `skills_src` adapter (a) is likewise moot for *signal* (though
   it remains a real backend gap). Forward: the untested surface is **MCP tool
   descriptions on an MCP-bearing agent** (the description is the sole routing
   signal there); or a weaker validator tier (but the weak tier abstains on the
   closed-loop path); or accept the pipeline's regression-catching nature.

6. **Consume the discrimination labels: one-shot suite hardening.** The
   per-task labeler ships labels; nothing acts on them. LLM-mutate
   `too_easy`-labeled tasks while preserving the verifiable
   `fixture_setup` / `test_command` scaffold; relabel with reps **and a
   ceiling artifact** to catch variants that silently became unfillable;
   output a human-reviewed suite diff plus a regenerated noise sidecar. Fund a
   recurring co-evolution loop only if ≥30% of variants land discriminative.
   *Effort M, $10–60.*

7. **Goldilocks tier probe — the experiment is the deliverable.** Sweep the
   baseline at reps=3 across 3–4 agent tiers × 2 suites, logging graded
   channels (tool-call sequence, step count, duration) beside the binary
   verdict. This distinguishes "no mid-band tier exists" from "no mid-band
   *under binary verdicts*," and pre-feeds item 12's channel-stability
   question. *Effort S–M, $10–25. Kill: every tier reproduces the
   all-pass/all-fail cliff → record the null, and item 9's weak-tier
   improvement framing collapses to a deploy-tier regression guard.*

   **Status — investigated; no graded signal (kill experiment fired).** A staged
   probe (~$12, under budget) ran the experiment as designed. (1) Cheap
   kill-switch: **no transition rung is reachable** in the [0.3, 0.7] band — even
   strong active misdirection only drags a capable agent (gpt-5-mini) from 0.86
   to 0.71–0.79, and the weak tier (gpt-5-nano) does not run on the closed-loop
   path, so the ladder cannot sample a middle. (2) Thin-rung salvage on the
   closest rung (0.79, the one tier with a real pass/fail mix): within the only
   task that flips across reps, the continuous channels (step count,
   distinct-tool count) do **not** separate passed from failed reps (AUC ≈ 0.50),
   while the duration negative control tripped *harder* (0.72) at that sample
   size — no trustworthy graded signal. The failure signature is a tool **swap at
   identical sequence length** (`read_file→patch` on pass vs `read_file→write_file`
   on fail, both ~2 steps), so a graded channel only re-encodes the binary
   membership verdict — no graded gain (the item-13 lesson at the channel level);
   the pooled cross-task "separation" is a between-task length confound, not a
   fitness signal. **Verdict: the binary tier-flip (finding #2) is fundamental,
   not a measurement artifact a continuous channel can rescue.** Consequences:
   item 9's weak-tier-improvement framing collapses to a deploy-tier regression
   guard; item 13's deferred partial-credit verdict (Stage 2) is **not** revived.
   The forward signal path is item 5 (planted-fault, test-verified tasks), which
   sidesteps this saturation regime.

### Wave 3 — Honest mechanism

8. **Stratified behavioral-aware minibatch sampling (the acceptance fix).**
   GEPA's `BatchSampler` is a pluggable protocol and `dspy.GEPA`'s
   `gepa_kwargs` splat straight into `gepa.optimize` (the repo already rides
   this for `acceptance_criterion`), so no fork is required;
   `reflection_minibatch_size` must be `None` when a custom sampler is passed.
   A sampler guaranteeing ≥1 behavioral example per reflective minibatch
   (weighted toward parent-failing examples) closes the acceptance bottleneck
   identified in `reports/pareto_frontier_feasibility.md` — per-instance
   behavioral signal that stochastic small-minibatch sum-acceptance discards
   ≈62% of the time at 7/46-failing. *Effort M, ~$10. Kill: paired-seed runs
   (with/without sampler, same seed) show no increase in accepted candidates
   that win ≥1 behavioral task, compared via lineage acceptance traces.*

9. **Gate on a fresh paired draw; conditional transfer guard.** The gate can
   currently consume a cached search-phase report (`force_run` returns cache
   hits), i.e. a stale, unpaired draw with selection bias the A/A floor does
   not cover. Require one fresh paired baseline-vs-winner run at gate time
   (evolution tier), with the noise-aware required-gain calibrated for the
   fresh draw. Separately budgeted — **not** shareable across tiers — an
   opt-in deploy-tier non-inferiority check, sold as a regression guard, not
   an overfitting detector (capable tiers saturate). Its weak-tier-improvement
   framing is funded **iff item 7 finds a mid-band tier.** *Effort M.*

10. **Closed-loop parallelism behind an A/A fidelity gate.** Splice happens
    once per phase and each rep is tempdir-isolated, so the real risks are
    cost-ledger races and ceiling overshoot, not splice corruption.
    Parallelize reps within a task first; cross-task-within-phase is a stretch
    behind the same serial-vs-N-worker flip-rate check. Makes reps=3–5
    affordable, which tightens every noise floor. *Effort M. Kill: 4-worker
    flip rate exceeds the serial A/A floor → keep it serial.*

### Wave 4 — The original creator items, probe-gated

11. **Continuous loop as a propose-only, self-hosting triage queue**
    (`evolution/monitor/`, currently an empty stub). A ranked queue of
    (artifact, suite, headroom band, last-evolved age, estimated cost,
    ready-to-run command), fed by a detection-only sentinel (artifact
    manifest + scheduled noise-aware live-vs-snapshot validation + a runs
    registry). No auto-evolve, no auto-PR — matching the original Phase 5
    human-merge mandate. The manifest **enrolls this repo's own artifacts**
    (its CLAUDE.md, the orchestration prompts, its skills) as first-class
    entries, so the loop becomes self-hosting rather than terminating at
    "improve other agents." This also accumulates the cross-run lineage
    substrate a future clade-aggregated parent-selection experiment would
    need. *Effort M–L. Kill: triage ranks a known-saturated artifact above the
    known-headroom class on ground truth → the ranking signal is broken,
    shelve until fixed.*

    **Status — SHIPPED, reframed to a CODE-target sentinel + validated.** Reframed
    by the post-validation panel: rather than enrolling the decoupled artifact
    surfaces (skills/CLAUDE.md — null on capable agents), the sentinel enrolls the
    one surface with a proven gradient — **code**, with the validated repair loop as
    its consumer. `evolution/monitor/` now scans a target repo's recent git stream
    for repair candidates, classifies them (`dependency_regression` first — its
    pre-bump behavior is a correct reference — then `bug_fix`), ranks them, and emits
    a propose-only `triage_queue.json` + report. `--attempt-top K` runs the validated
    worktree→repair→oracle-gate flow on the top K and annotates each row; never opens
    a PR. A free recon first overturned the "no live feed" premise (~674 tests/tools
    commits/quarter; ~6 dependency-regressions/quarter). A gated `--attempt-top` demo
    validated the path end-to-end on real Hermes material: skips correctly classified
    (`not_valid`/`too_large`/`source_missing`, free), a real deploy-reachable success
    (`tool_backend_helpers`, a dep-regression, 3/3 seeds), cost tracked ($0.895,
    ceiling enforced). **Finding:** the `dependency_regression` label is noisy (it
    flags any pyproject/uv.lock touch, ≠ a dep-version bump); genuine dep-regressions
    came back `not_valid` (broad migrations). The general novel-bug product (held-out
    split, no oracle) stays deferred: metamorphic verification was killed (0/8 — real
    bugs are input-specific, don't violate generic invariants).

    **Cross-version-dep-env build — KILLED on a $0 supply check (don't build).** The
    proposed next build (repair dependency-regressions, verified via a cross-version
    differential: pre-bump tool @ old dep vs repaired @ new dep) presupposed a supply
    of clean tool-local dep-version regressions. There is none: on Hermes/365d, of
    ~161-184 dep-manifest commits, **0 in-place pyproject pin bumps and 0 clean
    tool-local dep-version regressions**; the ~3-6 version-bumps touching a tool+test
    are all broad migrations (28-97 files) that can't reproduce as a tool-local repair,
    and the one small candidate was a feature-add. Triple-confirmed (strict + loosened
    git counts + the live demo's `not_valid` results + an independent red-team re-run).
    Weeks of two-dep-env machinery for ~0 instances — a clean signal-before-mechanism
    kill. Salvage angles closed: a looser scope dissolves the tool-local verification
    model; the cross-version differential is parasitic on absent supply; a higher-churn
    repo is a pivot off the validated Hermes substrate, not a save.

    **Frontiers mapped → redirect (panel-ranked); options (1) and (2) SHIPPED.** With
    artifact-quality decoupled, general no-oracle repair dead/narrow, and the
    dep-regression supply absent, the durable wins are test-feedback code repair
    (~0.60–0.74 on real bugs) + the shipped sentinel.
    - (1) **Bank the body of work — SHIPPED.** The *asymmetry* is banked as a findings
      report + a paper-style PDF rendered through a sibling renderer that **fails
      closed** if a prose number drifts from its source JSON, with committed source
      snapshots so it reproduces from version control (`reports/asymmetry_*`). The
      sentinel is operationalized (runbook + `$0` scan wrapper writing a stable
      `latest` pointer + an opt-in scheduled scan; `--attempt-top` now *requires*
      `--max-cost-usd`). A 4-expert panel + two spend-backed self-checks hardened it.
    - (2) **Push N + difficulty boundary — SHIPPED.** A 2nd campaign (fresh harvest,
      same protocol, `$12`-capped at N=46) gives deploy-reachable **0.74 [0.60, 0.84]**,
      corroborating N=20's 0.60 (honest range **~0.60–0.74**; ICC swung 0.33→0.63). The
      difficulty curve answers "only easy bugs?" → **no**: by fix-LOC {≤5: 1.00, 6–20:
      0.83, >20: **0.69**}, median fix 45 LOC, 35/46 large fixes — graceful degradation,
      not a triviality cliff. No 2nd convention-repo exists locally, so true external
      validity is unbuilt.
    - **Positioning reframe (shipped).** The README now leads with the **deploy gate**
      as the product (the evolver is one candidate source), so the decoupling null
      reads as the gate working, not a failure; a "use the gate without evolving
      anything" section surfaces the standalone CLIs.
    - **Remaining open bets (need explicit go-aheads):** (3) the **$10 MCP-description
      probe** (last untested artifact surface), and **true external validity** (a 2nd
      corpus — real setup effort). Item-10 CL-parallelism (serves the decoupled path)
      and clade selection (no substrate) are skipped.

12. **Code-evolution pilot, gated on a measured signal** (`evolution/code/`,
    empty stub). Before any Darwinian loop: an A/A coefficient-of-variation
    probe on per-task step-count / cost / duration channels for tool-code
    fitness (duration is provider-latency-dominated and expected to fail;
    step-count and cost are the plausible channels). Build the minimal loop
    (target-repo tests as a hard floor, behavioral suite as fitness, an
    AGPL-licensed evolver integrated as an external CLI only) **only if a
    channel proves stable.** *Effort: $10 probe, then L if funded. Kill: no
    channel's A/A CoV is tight enough to detect a plausible code improvement
    → the signal is too noisy and the pilot does not proceed.*

    **Status — multi-round evolver NOT justified on single-site bugs; one-shot
    test-feedback repair works; Tier-A (real bugs) pending.** The item-12 kill as
    originally specified was mis-specified (A/A CoV on agent-mediated
    step-count/cost/duration — the channels item 7 found noisy). Reframed to a
    **proposer-gradient probe**: an LLM proposer fixes broken code under a
    deterministic test; measure whether iterative test-feedback fixing has a
    reachable gradient vs one-shot/flat. On **23 non-circular organisms**
    (mechanical single-site mutations — operator-flips / off-by-one /
    dropped-returns — of real Hermes tool functions with self-contained tests):
    **19/23 flat (one-shot-trivial), 3/23 band, 1 near-cliff** → band fraction
    0.13, Wilson [0.05, 0.32], lower bound 0.05 ≤ 0.10 → **KILL** on the
    reachable-gradient threshold. A no-feedback ablation decomposed it: the
    deterministic test's **failure output is the essential signal and it enables
    ONE-SHOT repair, not multi-round evolution** — feedback survival climbs only
    0.83→0.94 over five rounds while the blind ablation is dead-flat at 0.75;
    within-organism, a subtle off-by-one goes 0/5 blind → 5/5 with the test output
    (one-shot). So a capable proposer + deterministic test feedback = effective
    one-shot code repair (the clean-fitness premise holds), but the **multi-round
    Darwinian-evolver machinery is overkill** for single-site bugs — the right
    shape is a simple one-shot test-feedback repair loop under the existing
    deterministic-floor + noise-aware gate (the commodity SWE-agent pattern; the
    project's distinctive asset is the gate, not the evolver).

    **Tier-A update — the single-site KILL was a difficulty artifact; real bugs
    show a real gradient (GREEN).** Re-ran the proposer-gradient probe on **8 real
    git-history bugs** (parent-commit tool source + the fix commit's test, on the
    self-contained tools) plus 14 harder multi-site mutations. On the **real**
    stratum: **0/8 one-shot (none trivial), 6/8 climb to a fix with iterative test
    feedback, 2/8 cliff** → band fraction **0.38, Wilson [0.14, 0.69]** (lower
    bound 0.14 > 0.10) → **GREEN**. So on authentic bugs the proposer never
    one-shots and iterative test-feedback fixing has a genuine gradient — the
    multi-round loop IS justified, **reversing the single-site interim read**
    (single-site mechanical mutations, and trivial-module multi-site like
    `ansi_strip`, are one-shot-trivial — too easy to populate the band; difficulty
    tracks code complexity, not mutation count). A real bug backlog exists (8 from
    5 self-contained tools; far more across the repo). Caveats: N=8 (wide CI); the
    multi-site supplement was contaminated by `ansi_strip`'s simplicity, so the
    real stratum is authoritative. **Net: code evolution — iterative test-feedback
    repair of real tool bugs, tests as a hard floor + the noise-aware deploy gate,
    self-hosting on the agent's own tools — is the campaign's first viable
    improvement-finding direction, distinct from the capable-agent artifact-quality
    decoupling.**

    **Head-to-head RUN — evolutionary search is not justified (tested, not inferred).**
    Earlier framing conflated two things: "the multi-round loop IS justified" means
    *iterating a single proposer with test feedback* beats one-shot (tested, true); it
    is **not** evidence about the **Darwinian Evolver's population-based search**
    (population + fitness-guided selection on partial credit + recombination), which had
    never been run. So we ran it — an evolutionary proxy vs. plain **best-of-N**
    resampling at equal candidate budget, on the **23 organisms the single-proposer loop
    fails** (the regime where exploration should help), same oracle gate
    (`reports/asymmetry_evolution_proxy.json`). Result: **evolutionary 4/23 [0.07, 0.37]
    vs best-of-N 6/23 [0.13, 0.47]** — best-of-N **strictly dominated** (recovered every
    bug evolution did *plus 2 more*; evolution recovered none best-of-N missed). On
    `skill_usage` the evolutionary selection converged to a fix passing all bug-tests but
    failing the oracle gate — a local optimum best-of-N's diversity avoided. So the
    population/selection machinery adds **no advantage** over simple resampling and is
    not justified on this corpus — now **proven by head-to-head, not asserted**. (N=23,
    CIs overlap, so "best-of-N *beats* evolution" isn't significant; "evolution shows no
    advantage" is.) Genuine cheap lever surfaced: **more independent samples** (best-of-N
    = 9 vs the loop's 3 seeds) recover ~26% of the loop's failures — a loop tweak (more
    seeds), not the evolver. *Method note:* the first proxy run was **voided** by dspy
    response-caching silently collapsing each population/best-of-N into one repeated
    sample (caught via an implausibly low $0.98 cost + a direct probe); re-run with
    `cache=False` + unique per-draw tags + a draws-vs-calls guard.

    **Self-improvement intent — double-walled (verification + supply, supply
    confirmed at scale).** The original Phase-4 intent (an agent improves its own
    tools/usage) is blocked on two independent walls. *Verification:* a self-directed
    loop where the agent authors both the test and the fix has no external oracle —
    the gate's anti-gaming core (held-out split) is operator-supplied and the gate
    rejects `holdout == visible` as a tautology, so self-authored-test + self-authored
    fix removes the very checks that make a green test trustworthy (the agent grades
    its own homework; the leakage result shows the test's expected values do ~2/3 of
    the work). *Supply:* the reproducible-tool-bug feed a self-hardening loop needs is
    structurally ~0. First mis-judged on a sparse Hermes `state.db` (6 sessions → 0;
    an unrepresentative-sample over-claim, retracted), then measured on a large local
    agent-session corpus (**836 sessions, 41,376 tool calls**): of ~3,000
    tracebacks-in-tool-results, **71% were a file-read returning text that merely
    contains an error string** (logs/test-output/transcripts), **24% were shell runs
    of code/tests crashing** (the normal dev loop, fixed in-session), and **~0% were a
    tool itself throwing a bug**; human corrections were abundant (**14.8%** of turns)
    but target the agent's work/plan — the artifact-quality axis already shown
    decoupled on capable agents. So scale yields volume but not the right fuel:
    agents fix their own crashes in-loop and are corrected on their work, not on
    reproducible tool internals. **Conclusion:** self-improvement is not reachable on
    this substrate; the durable asset is the deploy gate (an adversarial verifier for
    machine-authored patches), and the mapped boundary — *self-improvement is gated by
    independent verification and a supply of verifiable failures, both structurally
    scarce for a capable agent* — is the honest finding. Revisit only with a new
    substrate (fleet telemetry; a weaker tier where artifacts couple; a novel-contract
    MCP surface).

    **Loop shipped, then de-risked, then reframed to a measurement campaign.** The
    iterative test-feedback repair loop shipped (`evolution/code/`: `evolve_code`,
    isolated git-worktree+venv, a gate with surface-freeze, file-scope, a held-out
    test split, and a baseline-diff regression floor). Three free de-risking checks
    then corrected the estimate and the next step:
    - The pooled "0.60 per-attempt" was **pseudo-replicated** (24/40 over 5 seeds × 8
      organisms, ICC 0.57, design-effect 3.28). **The honest unit is the organism:
      band 0.38 [Wilson 0.14, 0.69], deploy-reachable 0.75 [0.41, 0.93]** — one
      organism above the 0.10 kill line. Demote the 0.60 everywhere.
    - The `tests/tools` floor is **deterministic** (no flaky node-ids over 5 runs).
    - The held-out split's anti-gaming power is **real but limited** (only ~3/8 real
      bugs have input-diverse tests), and there is **no live bug feed** (the repo
      keeps its tests green). Since every harvested historical bug carries its
      upstream fix as a ground-truth **oracle**, the next build is a **measurement
      campaign** that verifies repairs by behavioral match to the oracle (sidestepping
      the held-out problem) and confirms-or-retracts the GREEN at staged N=8→20→50.

    **Measurement campaign — built, N=8 pilot GREEN.** `evolution/code/{harvest,gate
    (run_code_oracle_gate),campaign,campaign_report}.py`. A free oracle-validity recon
    cleared the supply risk (30 valid organisms in 44 checked, 68%, across 54 candidate
    tools → N=50 reachable). Reporting is organism-level (Wilson + cluster bootstrap +
    ICC). The campaign stratifies across tools for organism independence and
    futility-stops if deploy-reachable Wilson-lower < 0.10. **Staged result (GREEN,
    tightening):** N=8 pilot deploy-reachable 5/8 = 0.625 [0.31, 0.86]; **N=20
    deploy-reachable 12/20 = 0.600, Wilson [0.39, 0.78], cluster-bootstrap P(<0.10
    kill)=0 → GREEN** (CI materially narrowed from N=8; ICC 0.33, effective-N 36). All
    on diverse, non-trivial tools (approval, browser_camofox, checkpoint_manager,
    cronjob_tools, discord_tool, env_passthrough, file_operations, file_tools, …),
    distinct from and harder than the Tier-A self-contained set — so the gradient
    generalizes. A `max_tokens=8000` truncation artifact was caught and fixed
    (whole-file rewrites of larger tools silently truncated → false failures; raised to
    32000 + a too-large exclusion). Cost tracking was a no-op until the litellm cost
    callback was wired in (now real: the --max-cost-usd ceiling is enforced).

    **Corroborated + reframed (post-banking).** A larger run (N=46, $12-capped) gives
    deploy-reachable **0.74 [0.60, 0.84]**, tightening the lower bound to 0.60 — honest
    rate **~0.60–0.74** across both runs (ICC swung 0.33→0.63, so no single effective-N
    is trustworthy). A **difficulty curve** shows repair holds on substantial bugs
    (0.69 on >20-LOC fixes, median fix 45 LOC), not just one-liners. **The headline is
    `test-feedback repair`, NOT autonomous re-derivation:** a leakage check (re-run with
    the test's *expected values* withheld) drops 11/12 successes → 3/12, so 0.60–0.74 is
    the deployable rate (the failing test is always present in production) and ~0.25 is
    the autonomy floor — which *sharpens* the asymmetry (the test's concrete values are
    the gradient the behavioral arm lacks). A **fuzzed differential** cleared the 4
    control fixes it could meaningfully test (4/4 generalize on fresh inputs, 0 overfit)
    but is thin (4 of 20). All banked in `reports/asymmetry_*`. The novel-bug-repair
    **product** (held-out split + a live feed + solving held-out independence) remains
    **deferred**.

13. **Turn-level tool-sequence evaluation** (the salvage from item 4's
    deferral). Item 4's single-call miner died because Claude agentic turns are
    multi-step: judging one call out of a ≈14-call turn with no context flips
    the misselection label ~20% of the time. The idea worth keeping is a
    different, larger one: evaluate a whole TURN's tool *sequence* rather than
    individual calls. Shape: mine `(task → ordered tool sequence)` units; judge
    with full turn context ("was any tool clearly wrong-for-the-task, or any
    necessary tool missing"); restrict to declared confusable tool pairs
    (`ToolManifest.confusable_neighbor_for`, `tool_source.py:130`) so a
    description edit can plausibly move behavior; and add a **turn-level metric**
    `evolve_tool` lacks today (it is single-choice by construction,
    `tool_module.py:53`). Open questions: where the privacy-safe turn corpus
    comes from (a single project's own transcripts are built-in-only; MCP turns
    aren't present in the privacy-safe scope), and whether a turn-level judge
    can produce signal a capable agent doesn't already saturate. *Effort L;
    needs its own brainstorm + design pass before any build.*

    **Status — investigated; the turn-level JUDGE is not justified (kill
    experiment fired).** A measured probe reframed the substrate from mining
    transcripts to running a live agent against the committed confusable suite
    (`write_file_ambiguous.jsonl`), comparing the production `write_file`
    description to an *actively-misdirected* one on a capable agent over three
    reps. Findings: (1) Active misdirection is **not inert** — it flipped
    exactly **one of seven** tasks (the agent fell from `patch` to the
    forbidden, content-wiping `write_file` on a preserve-edit task), a real
    effect that an earlier passive-weakening probe missed. But it is **faint**:
    the other six tasks are description-insensitive (the capable agent routes
    from task semantics, not the description text), the aggregate gap is ~0.10,
    and it is not shown to exceed the suite's ~15–20% single-run flip noise.
    (2) The proposed **turn-level judge agreed with the existing zero-LM
    membership verdict on every signal-bearing task** and added no
    discrimination on the one task that moved; its only divergence was a
    false-positive on the either-tool control task. So it does not beat
    membership — "the suite states the win" again. **Do not build the
    turn-level judge for built-in tools**; the faint regression signal that
    exists is already caught by the existing membership verdict on a confusable
    suite (zero new code). The surface where a description→behavior test could
    actually pay off is **MCP tool descriptions** — the only framework-evolvable
    tools, and the one place the description is the sole routing signal rather
    than inferable from the task. That is the forward path if pursued. (One
    suite task fails regardless of description — a non-discriminating task worth
    pruning.)

### Deferred / not pursued (with reasons, so they aren't re-litigated)

- **Clade-aggregated parent selection** (HGM-style): no campaign substrate
  yet — the archive holds zero multi-generation lineage trees, and artifact
  saturation collapses descendant-outcome aggregation precisely where
  campaigns would run. Revisit after item 11 accumulates real lineage.
- **MCP description-overriding proxy evolution:** gated behind a $10
  discrimination-labeler headroom probe on a genuinely confusable server;
  most servers' tools may be as non-confusable as built-ins.
- **Task-level gate racing / sequential early stopping:** the offline replay
  corpus does not exist (no archived per-task validation reports); the
  rep-level early-stop is folded into item 10.
- **Novelty rejection-sampling of candidates:** waste is unquantified at this
  repo's scale (tens of candidates per run); gated behind the near-duplicate-
  rate measurement in item 1.
- **Elo / tournament selection** (eval-as-selection): unreviewed single-run
  preprint evidence, and the design conflicts with the gate-centric
  architecture. Revisit only if item 8's acceptance fix fails to clear the
  bottleneck.
