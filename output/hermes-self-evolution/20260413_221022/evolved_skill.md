---
name: hermes-self-evolution
description: "Evolutionary self-improvement system for Hermes Agent. Uses DSPy + MIPROv2 to optimize skills, prompts, and instructions. LLM-as-judge evaluation with MAD confidence scoring for statistically rigorous quality gates. Point at any skill to evolve it, or at the system itself for meta-evolution."
version: 1.0.0
author: Hermes Agent Self-Evolution
license: MIT
metadata:
  hermes:
    tags: [Evolution, DSPy, Self-Improvement, Meta, Optimization, Quality-Gates]
    related_skills: [dspy, hermes-agent, systematic-debugging]
---

# hermes-self-evolution

Evolutionary optimization pipeline that improves Hermes Agent capabilities through automated prompt/skill evolution.

## System Architecture (3,081 lines, 17 modules)

```
evolution/
├── core/                    # Shared infrastructure
│   ├── config.py           (98 lines)   EvolutionConfig — repo discovery, model selection, constraints
│   ├── dataset_builder.py  (211 lines)  SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader
│   ├── fitness.py          (182 lines)  FitnessScore, LLMJudge, skill_fitness_metric
│   ├── hermes_judge.py     (146 lines)  HermesJudge — LLM-as-judge via hermes chat subprocess
│   ├── mad_scoring.py      (387 lines)  ConfidenceScoredFitness, compute_confidence, compute_mad
│   ├── constraints.py      (174 lines)  ConstraintValidator — size, growth, structure, test gates
│   └── external_importers.py (785 lines) Session mining from Claude Code, Copilot, Hermes
├── skills/                  # Phase 1: Skill evolution
│   ├── evolve_skill.py     (479 lines)  Main entry point — full optimization loop
│   ├── MADevolve_skill.py  (485 lines)  Variant with MAD-guarded optimizer metric
│   └── skill_module.py     (123 lines)  SKILL.md as DSPy module, load/find/reassemble
├── tools/                   # Phase 2: Tool description evolution (planned)
├── prompts/                 # Phase 3: System prompt evolution (planned)
├── code/                    # Phase 4: Code evolution (planned)
└── monitor/                 # Phase 5: Continuous loop (planned)
```

## Core Interfaces

### Config & Discovery
```python
# Auto-discovers hermes-agent repo
config = EvolutionConfig(
    iterations=10,
    optimizer_model="openai/gpt-4.1",
    eval_model="openai/gpt-5.4",
    judge_model="openai/gpt-5.4",
)
path = get_hermes_agent_path()  # HERMES_AGENT_REPO → ~/.hermes/hermes-agent → sibling
```

### Skill Loading
```python
skill = load_skill(skill_path)       # Returns {path, raw, frontmatter, body, name, description}
path = find_skill("arxiv", hermes_path)  # Searches skills/**/SKILL.md
full = reassemble_skill(frontmatter, evolved_body)  # Reconstructs SKILL.md
```

### DSPy Module Wrapper
```python
module = SkillModule(skill_text)  # Wraps skill as dspy.Predict with skill_instructions field
prediction = module(task_input="...")  # Returns prediction.output
```

### Evaluation
```python
# Fast heuristic (keyword overlap) — used by optimizer
score = skill_fitness_metric(example, prediction)

# LLM-as-judge (GPT-5.4 via hermes) — used by holdout
judge = HermesJudge(model="gpt-5.4")
fitness = judge.score(task_input, expected_behavior, agent_output, skill_text)
# Returns FitnessScore(composite, correctness, procedure_following, conciseness, feedback)

# MAD confidence scoring — wraps any judge
mad_judge = ConfidenceScoredFitness(judge, n_trials=3)
fitness, confidence = mad_judge.score_with_confidence(...)
# Returns (FitnessScore, ConfidenceResult(label, confidence, delta, decision))
```

### MAD Confidence Math
```python
compute_mad(values)  # Median Absolute Deviation — robust noise measure
compute_confidence(scores)  # confidence = |best - baseline| / MAD
# Labels: "likely real" (≥2.0x), "marginal" (≥1.0x), "within noise" (<1.0x)
# Decision: "keep" if improvement AND confidence ≥ 2.0x
```

### Constraints
```python
validator = ConstraintValidator(config)
results = validator.validate_all(artifact_text, "skill", baseline_text=original)
# Checks: size_limit, growth_limit, non_empty, skill_structure (frontmatter)
# Optional: run_test_suite(hermes_repo) — pytest gate
```

## Data Flow

```
SKILL.md → load_skill() → {frontmatter, body}
    ↓
SkillModule(body) → DSPy module with skill_instructions parameter
    ↓
SyntheticDatasetBuilder.generate(body) → 20 eval examples → train/val/holdout split
    ↓
MIPROv2 optimizer:
  - Proposes 3 instruction candidates from few-shot examples
  - Evaluates on valset using skill_fitness_metric (keyword overlap)
  - Bayesian optimization selects best instruction+demo combination
    ↓
Extract evolved_body from optimized_module.skill_text
    ↓
ConstraintValidator.validate_all(evolved_full, "skill")
    ↓ (if passes)
HermesJudge.score() on each holdout example (3 trials per example)
    ↓
ConfidenceScoredFitness aggregates trials → per-example ConfidenceResult
    ↓
Holdout-level MAD: compute_confidence([baseline_mean] + evolved_scores)
    ↓
Decision: "keep" if improvement AND confidence ≥ 2.0x
    ↓
Save: evolved_skill.md + metrics.json + proof.json (instructions, demos, diff)
```

## CLI Usage

```bash
# Evolve a hermes-agent skill
cd right/absorb/hermes-agent-self-evolution
HERMES_AGENT_REPO=/path/to/hermes-agent .venv/bin/python -m evolution.skills.evolve_skill \
    --skill arxiv \
    --iterations 5 \
    --eval-source synthetic \
    --mad-trials 3 \
    --judge-model "gpt-5.4"

# Evolve the system itself (meta-evolution)
.venv/bin/python -m evolution.skills.evolve_skill \
    --skill self-evolution-architecture \
    --iterations 5 \
    --eval-source synthetic \
    --mad-trials 3 \
    --judge-model "gpt-5.4"
```

## Proven Results

### Self-Evolution (2026-04-13)
```
Baseline:  0.772
Evolved:   0.875 (+13.4%)
Confidence: marginal (1.03x)
Time:       683.1s
Method:     MIPROv2 instruction optimization (content unchanged)
```

### Arxiv Skill Holdout (regression caught)
```
Baseline:  0.739
Evolved:   0.656 (-11.2%)
Confidence: likely real (3.32x)
Decision:   discard (correctly rejected worse variant)
```

## What Would Improve This Architecture

### Immediate Impact
1. **Direct API for judge** — skip hermes subprocess overhead (60-120s → 2-5s per call)
2. **Parallel holdout eval** — run examples concurrently, not sequentially
3. **LLM-as-judge for optimizer metric** — not just holdout (keyword overlap is weak proxy)

### Medium-term
4. **Ensemble judging** — multiple models score each example, aggregate
5. **Adaptive trial count** — more trials when MAD is high, fewer when stable
6. **Cached judge results** — don't re-score identical inputs across evolution runs
7. **Save optimization proof** — export optimized_module.signature.instructions + demos

### Long-term (Phase 2-5)
8. **Tool description evolution** — optimize tool schemas for selection accuracy
9. **System prompt evolution** — parameterize prompt_builder.py sections
10. **Code evolution** — Darwinian Evolver with test gates
11. **Continuous loop** — automated pipeline running unattended

## Evaluation Criteria for Architecture Changes

An improvement is valid if:
1. Total evaluation time decreases by ≥30%
2. Holdout confidence labels remain meaningful (not all "within noise")
3. Constraint validation still passes (no false negatives)
4. metrics.json captures all confidence data
5. proof.json captures optimized instructions + demos
6. The evolved artifact passes constraints with valid YAML frontmatter
