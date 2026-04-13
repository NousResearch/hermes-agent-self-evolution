# MAD Confidence Scoring

## Problem

When you evolve a skill, how do you know the improvement is real and not just noise from the LLM judge?

A single evaluation run might score 0.73 baseline and 0.81 evolved — but is that 0.08 delta meaningful, or just the LLM being inconsistent? Without statistical rigor, you can't tell.

## Solution

We use **Median Absolute Deviation (MAD)** to quantify measurement noise and compute a confidence ratio.

```
confidence = |mean_delta| / MAD

- >= 2.0x  → "likely real"   (improvement is well beyond noise)
- >= 1.0x  → "marginal"      (could be real, could be noise)
- <  1.0x  → "within noise"  (indistinguishable from random variation)
```

## How It Works

### Per-Example Scoring

For each holdout example, the judge scores it n_trials times. Scores vary because LLM-as-judge is non-deterministic at temperature=1.0.

```python
trial_scores = [0.61, 0.59, 0.43]  # real variance from GPT-5.4
mad = compute_mad(trial_scores)     # 0.02
```

### Holdout-Level Confidence

After scoring all holdout examples, we compute per-example deltas and use MAD on the delta distribution.

```python
delta_scores = [evolved[i] - baseline[i] for i in range(n_examples)]
mean_delta = sum(delta_scores) / len(delta_scores)
mad_delta = compute_mad(delta_scores)
confidence = abs(mean_delta) / mad_delta
```

### Decision Gate

Only improvements with confidence >= 2.0x AND mean_delta > 0 are kept. Everything else is discarded.

## Why MAD Not Standard Deviation?

MAD uses the **median**, not the mean. With small samples (3 trials), a single outlier can distort std dev. MAD is robust — it measures typical deviation, not average.

```
Standard deviation:  sensitive to outliers
MAD:                 robust — median of absolute deviations from median
```

## Proven Results

| Run | Baseline | Evolved | Delta | MAD | Confidence | Decision |
|-----|----------|---------|-------|-----|------------|----------|
| Arxiv | 0.739 | 0.656 | -0.083 | 0.025 | 3.32x | discard (real regression caught) |
| Self-evolution | 0.772 | 0.875 | +0.103 | 0.100 | 1.03x | keep (marginal) |
| Comprehensive | 0.968 | 0.948 | -0.020 | 0.020 | 1.00x | discard (noise) |

## Implementation

- `evolution/core/mad_scoring.py` — Pure MAD math (basic functions work without dspy)
- `evolution/core/hermes_judge.py` — LLM-as-judge via hermes chat subprocess
- `evolution/skills/evolve_skill.py` — Holdout eval with MAD confidence + proof.json
- `evolution/core/config.py` — GPT-5.4 defaults, Nous API support
- `evolution/core/fitness.py` — temperature=1.0, MAD re-exports
