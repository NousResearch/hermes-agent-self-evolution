# Deploy-gate calibration: findings

A calibration campaign ran 24 instrumented evolve runs across nano-pdf, apple-notes, polymarket, maps, and linear (with controls). The campaign aimed to set evidence-based defaults for `growth_free_threshold`, `growth_quality_slope`, the knee-point ε, and the `eval_dataset_size`/`holdout_ratio` pair.

**Verdict on the proposed `(free, slope) = (0.10, 0.50)` defaults: rejected.** The current defaults `(0.20, 0.30)` outperformed the proposed values on held-out skills (3.3× higher mean lift on deployed variants). The current defaults remain unchanged.

The campaign nonetheless produced four findings that are independent of the verdict and worth recording for future calibration work.

## Finding 1 — Architectural coupling: gate parameter doubles as proposer target

`evolution/skills/evolve_skill.py` constructs `BudgetAwareProposer` with `max_growth=config.growth_free_threshold`. A single field controls **both** the gate's pass-criteria curve **and** the proposer's prompt target. Lowering `growth_free_threshold` from 0.20 to 0.10 in the proposed arm tightened the proposer's prompt to the reflection LM, which produced more conservative variants — 2 of 4 proposed-arm validation runs were no-op deploys (the knee-point fell back to candidate 0, the baseline itself, because GEPA found nothing better).

**Implication for future calibration of `(free, slope)`:** decouple the proposer's `max_growth` from the gate's `growth_free_threshold`, or co-optimize both. As-is, picking a "stricter" gate also cripples the proposer, and the two effects are inseparable in the data.

## Finding 2 — Synthetic generator caps natural distinct test cases per skill

The `SyntheticDatasetBuilder` requests `num_cases=N` from the judge LM, but the LM treats `num_cases` as a soft suggestion and emits whatever count it can support given the skill's natural test-case space. Observed drop rates at `eval_dataset_size=250`:

| Skill | Baseline chars | Valid examples returned | Drop rate |
|---|---|---|---|
| nano-pdf | 1372 | 55 | 78% |
| apple-notes | 2169 | 31 | 88% |
| polymarket | 2932 | 60 | 76% |
| huggingface-hub | 3639 | ~20 | 92% |
| plan | 1981 | 16 | 94% |
| maps | 6643 | 105 | 58% |
| linear | 11185 | 123 | 51% |

Bumping `eval_dataset_size` to 500 did not help — the LM produced the same count. **Implication:** future calibration corpora should restrict to skills ≥3000 chars or use a different eval source. Sub-2500-char skills are unsuitable for this evaluation pipeline regardless of the requested `num_cases`.

## Finding 3 — Validated starting points (informative, not shipped)

The four campaign studies produced robust observations even though the (free, slope) verdict was negative:

| Knob | Observed sweet spot | Basis |
|---|---|---|
| `eval_dataset_size` (N\*) | **250** | Median bootstrap CI half-width 0.022 across 13 historical runs; smallest grid value satisfying the 0.025 target |
| `holdout_ratio` (ratio\*) | **0.65** | Same sweep; tighter CIs without dropping below the bootstrap floor |
| Knee-point ε | **0.5 / n\_val** | Mean val→holdout transfer error 0.085 (matched 1.0/n\_val baseline); knee picked == GEPA default in 7 of 9 runs, so this knob barely moves outcomes on this corpus |
| `growth_free_threshold`, `growth_quality_slope` | **no change** | Study C's perfect-classifier ties (15 of 25 grid pairs scored Youden's J=1.0 on a 5-run, 3-positive-2-negative corpus); Study D rejected the tiebreaker pick on held-out skills |

These are starting points for a future campaign with a richer corpus, not current defaults.

## Finding 4 — Non-inferiority gate at tolerance 0.05 strictly improves on `no_regression`

A post-hoc gate-rule replay (script: `scripts/analysis/option1_replay_gate_rule.py` on the campaign's archive branch) tested whether the **non-inferiority** rule (`bootstrap.lower_bound ≥ -tolerance`) better matches the campaign's compression-bias behavior than the current `no_regression_only` rule (`bootstrap.mean ≥ 0`). Sweep across the 17 instrumented runs:

| Tolerance | Flip rejected → deployed | Flip deployed → rejected | Verdict vs current |
|---|---|---|---|
| 0.01 | 0 | 2 | strictly worse |
| 0.02 (preset's previous default) | 0 | 0 | no change |
| 0.05 | **1** (polymarket: mean=−0.005, lower=−0.050) | 0 | **strictly dominates** |
| 0.07 | 2 (+ maps seed=7: mean=−0.019, lower=−0.061) | 0 | strictly dominates |
| 0.10 | 2 | 0 | strictly dominates |

The `polymarket` run that flips at tolerance=0.05 is the canonical "noise-level rejection": bootstrap CI [−0.05, +0.04] straddles zero, mean dipped −0.5pp into negative noise, and the current rule rejected it as a regression. Under non-inferiority at tolerance=0.05 it correctly deploys.

**Shipping recommendation (this PR):** bump the `non-inferiority` preset's `inferiority_tolerance` default from `0.02` to `0.05`. Users who explicitly select `--quality-gate non-inferiority` for compression-bias evaluation get the sweet-spot tolerance. The `default` preset (`gate_mode=no_regression`) is unchanged — this PR does not switch the default rule, only improves the opt-in alternative.

A future campaign could revisit promoting `non-inferiority` to the default rule with explicit held-out validation.

## Out of scope

`bootstrap_*`, `min_holdout_size`, `max_skill_size`, `max_absolute_chars`, BCa bootstrap, GEPA's own knobs, per-category defaults, knee-point `strategy` choice. Same exclusions as the original campaign plan.

The `--max-absolute-chars 12000` override applied to Stage 7 validation runs (so the size ceiling didn't preempt the (free, slope) check on baselines >5000 chars) was a localized run-time fix and is not a calibrated default change. The in-tree default remains `5000`.

## Audit trail

Full campaign artifacts — runbook, analysis scripts, `study_*_results.json`, and per-run `output/<skill>/<ts>/{gate_decision,band_holdout,metrics}.json` — live on the `archive/2026-deploy-gate-calibration` branch. No-op merge to main; the report and the one-line preset change are the only durable changes from this campaign.
