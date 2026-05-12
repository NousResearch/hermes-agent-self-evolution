# Agent Self-Evolution — Evolutionary Self-Improvement for Agent Skills

## Vision

A standalone optimization pipeline that systematically improves an agent's performance by evolving skills, prompts, tool descriptions, and agent configurations using automated optimization loops. Lives in its own repo (`jramos/agent-self-evolution`), operates ON a target agent's skill repo — not part of it. Originally built for Hermes Agent; now works for any agent framework that emits `<skill>/SKILL.md` files (Claude Code, Hermes, custom local layouts).

Three complementary engines, unified under one workflow:

| Engine | What It Optimizes | License | Integration |
|--------|------------------|---------|-------------|
| **DSPy + GEPA** | Skills, prompts, instructions, tool descriptions | MIT | Native Python, primary engine |
| **Darwinian Evolver** | Code files, algorithms, tool implementations | AGPL v3 | External CLI only |
| **DSPy MIPROv2** | Few-shot examples, instruction text | MIT | Native Python, fallback optimizer |

GEPA is the star — it's integrated into DSPy, reads execution traces to understand WHY things fail (not just that they fail), and works with as few as 3 examples. It outperforms both RL and previous DSPy optimizers.

**Important: No GPU training required.** Everything in this plan operates via API calls only. DSPy+GEPA and MIPROv2 optimize the *text* of prompts, instructions, and few-shot examples — they mutate and evaluate strings, not model weights. The Darwinian Evolver evolves code files (also text). The only DSPy component that trains weights (`BootstrapFinetune`) is explicitly excluded from this plan. All evaluation runs through batch_runner making standard LLM API calls.

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

### Tier 4: Code Evolution (High Value, Highest Risk)
- **What:** Tool implementation code, helper functions
- **How:** Darwinian Evolver with GitBasedOrganism, test via pytest + batch_runner
- **Why it works:** Some tool implementations have subtle bugs or inefficiencies that evolutionary search can find
- **Risk:** Code changes can break things — requires strong test suites as guardrails
- **Example:** Evolve `file_tools.py` patch matching to handle more edge cases

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
│     - Code: Darwinian Evolver (external CLI)  │
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
│   ├── code/                           # Phase 4: Code evolution (Darwinian Evolver)
│   └── monitor/                        # Phase 5: Continuous loop
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

# Phase 4: Evolve tool code (uses Darwinian Evolver CLI)
python -m evolution.code.evolve_tool_code \
    --tool file_tools \
    --bug-issue 742 \
    --iterations 10

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
| **Phase 4** | Code evolution | 3-4 weeks | Phases 1-3 + strong eval pipeline | Bugs fixed, tests pass, benchmarks hold |
| **Phase 5** | Continuous loop | 2 weeks | All above working | Automated pipeline runs unattended |

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

1. **Benchmark gating (TBLite/YC-Bench) was not wired.** The plan made "TBLite within 2%" the next-phase gate. The framework went framework-agnostic (any agent that emits `SKILL.md`), so the Hermes-specific benchmark suites no longer apply uniformly. The deploy gate is a paired-bootstrap CI on a held-out split of the skill's own eval set (`evolution/core/stats.py` + `_check_growth_with_quality_gate`), which catches per-run regression risk without coupling to one agent's benchmark infrastructure.
2. **Eval datasets are larger than the planned 15–30 examples.** `eval_dataset_size=150` is the current default, sized so the bootstrap CI half-width can detect ±2% effects. Rationale and the supporting study live in `reports/calibration_findings.md`.
3. **Source D (skill-specific auto-evaluation) was not built.** Synthetic, sessiondb-mining, and golden loaders are wired; planted-bug / known-paper / planted-issue harnesses are not. The plan itself framed Source D as a bonus rather than a requirement.
4. **The selection and gating layer is much thicker than the plan anticipated.** Knee-point Pareto selection, a budget-aware reflection-prompt proposer with growth/balanced/compression modes, paired-bootstrap CI, non-inferiority gate option, and the `gate_decision.json` v4 audit schema were added on top of raw GEPA. Rationale: raw "ship the best valset candidate" overfits at small N. See `docs/framework_advantages.md`.
5. **CLI shipped as `python -m evolution.skills.evolve_skill`, not a `hermes evolve skill` subcommand.** The framework rebranded out of Hermes-only branding so it could target any agent framework whose skills are `SKILL.md` files.
6. **`--run-tests` (target-repo pytest as a hard gate) is off by default.** Skill text doesn't touch target-repo Python, so coupling the deploy decision to the target repo's test suite would gate on unrelated signal. The flag exists for callers who want it.

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
4. **Benchmark gating again absent.** Same as Phase 1 — the deploy gate is paired-bootstrap CI on the synthetic holdout, not "TBLite within 2%."
5. **Eval triples are `(task, correct_tool)`, not `(task, correct_tool, correct_params)`.** Schema structure is frozen by design; parameter correctness is a separate concern from description quality.
6. **Input surface is broader than planned.** The plan framed Phase 2 around Hermes' `tools/registry.py`. The build supports any agent that can emit an MCP `list_tools()`-shape JSON manifest *and* Hermes-style Python `*_SCHEMA` declarations (via `HermesToolSource`'s AST walker, with one-hop `Name`-reference resolution for cases like `TERMINAL_TOOL_DESCRIPTION`). This is a scope expansion, not a contraction.

### Phase 3: System Prompt Evolution

**Goal:** Optimize the sections of the system prompt that guide agent behavior.

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

### Phase 4: Code Evolution via Darwinian Evolver

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
- Darwinian Evolver: AGPL v3 ⚠️ (external CLI only, no Python imports)
- All Hermes-native code: MIT ✓

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
