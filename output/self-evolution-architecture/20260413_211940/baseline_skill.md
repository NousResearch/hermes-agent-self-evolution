---
name: self-evolution-architecture
description: Evolutionary self-improvement architecture for Hermes Agent skills, prompts, tool descriptions, and code. Uses DSPy + GEPA for prompt optimization, LLM-as-judge for evaluation, and MAD confidence scoring for statistically rigorous quality gates.
version: 1.0.0
author: Hermes Agent Self-Evolution
license: MIT
metadata:
  hermes:
    tags: [Architecture, Evolution, DSPy, GEPA, Self-Improvement, Meta]
    related_skills: [dspy, hermes-agent]
---

# Self-Evolution Architecture

Evolutionary optimization pipeline that improves Hermes Agent capabilities through automated prompt/skill evolution.

## Core Loop

```
1. SELECT TARGET — Pick a skill, prompt section, or tool description
2. BUILD EVAL DATASET — Generate test cases (synthetic, session mining, or golden)
3. WRAP AS DSPY MODULE — Skill text becomes optimizable parameter
4. RUN OPTIMIZER — MIPROv2/GEPA proposes instruction variants
5. EVALUATE — Score candidates on holdout set with LLM-as-judge
6. GATE — MAD confidence determines if improvement is real or noise
7. DEPLOY — Only variants passing confidence threshold (≥2.0x) proceed
```

## Current Architecture

### Components
- **SkillModule** (`skill_module.py`): Wraps SKILL.md as DSPy module, extracts frontmatter + body
- **SyntheticDatasetBuilder** (`dataset_builder.py`): Generates eval tasks from skill text
- **MIPROv2 Optimizer** (`evolve_skill.py`): Bayesian optimization over instruction + few-shot variants
- **HermesJudge** (`hermes_judge.py`): LLM-as-judge via `hermes chat -m MODEL -Q` subprocess
- **ConfidenceScoredFitness** (`mad_scoring.py`): Multi-trial MAD scoring wrapper
- **ConstraintValidator** (`constraints.py`): Size limits, structure checks, test gates

### Evaluation Pipeline
- Optimizer metric: `skill_fitness_metric` (keyword overlap heuristic, fast)
- Holdout metric: `HermesJudge.score()` → `ConfidenceScoredFitness` (LLM quality + MAD confidence)
- Confidence: `|mean_delta| / MAD` on per-example holdout scores
- Decision: `keep` if delta > 0 AND confidence ≥ 2.0x, else `discard`

### Known Bottlenecks
1. **Judge speed**: `hermes chat` subprocess ~60-120s per call. 30 calls for 5-example × 3-trial holdout = 30 min
2. **Optimizer metric**: keyword overlap heuristic doesn't correlate with LLM quality
3. **Single-model constraint**: Nous free tier only serves MiMo-v2-pro (reasoning model, deterministic scoring)
4. **No parallelism**: holdout eval runs sequentially, not parallelized
5. **MAD bootstrap**: bootstrap resampling in `mad_fitness_metric` measures synthetic noise, not real variance

### What Would Improve This Architecture
- Direct API calls for judge (skip hermes subprocess overhead)
- Parallel holdout evaluation (run multiple examples simultaneously)
- LLM-as-judge for optimizer metric (not just holdout)
- Ensemble judging (multiple models score each example, aggregate)
- Adaptive trial count (more trials when MAD is high, fewer when stable)
- Cached judge results (don't re-score identical inputs)
- Streaming evaluation (score examples as they complete, don't wait for all)

## Evaluation Criteria

An architecture improvement is valid if:
1. Total evaluation time decreases by ≥30%
2. Holdout confidence labels remain meaningful (not all "within noise")
3. Constraint validation still passes (no false negatives)
4. Metrics.json captures all confidence data
5. The evolved skill passes constraints and has valid YAML frontmatter

## Constraints
- Max architecture description: 15,000 chars
- Must preserve existing API (--skill, --iterations, --eval-source, --mad-trials, --judge-model)
- Must not break existing tests
- Must maintain backward compatibility with skill_module.py interface
