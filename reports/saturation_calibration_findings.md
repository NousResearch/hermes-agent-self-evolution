# Saturation-threshold calibration: findings

This report mines the deploy-gate archive to ask whether the saturation pre-flight's synthetic thresholds (`no_headroom_synthetic=0.99`, `weak_signal_synthetic=0.95`) would wrongly abort runs that actually improve. **The archive cannot yet settle the thresholds** — it is data-starved in exactly the gated region, and the pre-flight's abort decision was never persisted historically. The numbers below are a survivorship-bounded counterfactual on GEPA-completed runs, reported with Wilson bounds so absence of evidence is not mistaken for evidence of safety.

## Headline

At τ=0.95, **15 runs** would be aborted (4.4% of the pool); 0 produced a statistically-supported improvement → false-abort rate 0.0% (Wilson 95% upper bound **20.4%**). At τ=0.99, 4 runs, 0 improvements, upper bound **49.0%**. A point estimate of 0% on a near-empty gated region is absence of evidence; the upper bound is the honest read.

Pool: 341 paired-vector runs of 356 archived (0 unparseable files skipped; 15 closed-loop runs excluded from synthetic calibration; 340 in the homogeneous balanced+no_regression+synthetic stratum used for the primary analysis, 1 off-profile). Schema versions: {'5': 264, '4': 92}.

## Finding 1 — Binning by baseline holdout score (homogeneous stratum)

| bin | n | deploy% (Wilson) | mean gain | lb>0 frac (Wilson) | no-op deploy% | lb min/med/max | n_ex med |
|---|---|---|---|---|---|---|---|
| <0.90 | 325 | 54% [48,59] | 0.020 | 15% [12,20] | 38% | 0.000/0.000/0.441 | 10 |
| 0.90-0.95 | 0 | 0% [0,100] | — | 0% [0,100] | 0% | —/—/— | — |
| 0.95-0.99 | 11 | 91% [62,98] | -0.001 | 0% [0,26] | 91% | -0.037/0.000/0.000 | 50 |
| >=0.99 | 4 | 75% [30,95] | 0.000 | 0% [0,49] | 75% | 0.000/0.000/0.000 | 57 |

Under `no_regression` (the bulk of the archive) a 'deploy' means 'did not regress', not 'improved' — read the `lb>0 frac` and the `no-op deploy%` columns, not the deploy rate.

## Finding 2 — False-abort sweep (primary: bootstrap lower bound > 0)

| τ | would-abort n | % of pool | real improvements | false-abort rate | Wilson 95% upper |
|---|---|---|---|---|---|
| 0.95 | 15 | 4.4% | 0 | 0.0% | 20.4% |
| 0.97 | 15 | 4.4% | 0 | 0.0% | 20.4% |
| 0.99 | 4 | 1.2% | 0 | 0.0% | 49.0% |
| 1.0 | 4 | 1.2% | 0 | 0.0% | 49.0% |

## Finding 3 — Sensitivity across improvement definitions

False-abort rate (Wilson upper) at each τ, per definition:

| definition | τ=0.95 | τ=0.97 | τ=0.99 | τ=1.0 |
|---|---|---|---|---|
| lower_bound | 0% (≤20%) | 0% (≤20%) | 0% (≤49%) | 0% (≤49%) |
| decision_CIRCULAR | 87% (≤96%) | 87% (≤96%) | 75% (≤95%) | 75% (≤95%) |
| gain>=0.0 | 93% (≤99%) | 93% (≤99%) | 100% (≤100%) | 100% (≤100%) |
| gain>=0.01 | 0% (≤20%) | 0% (≤20%) | 0% (≤49%) | 0% (≤49%) |
| gain>=0.02 | 0% (≤20%) | 0% (≤20%) | 0% (≤49%) | 0% (≤49%) |
| gain>=0.05 | 0% (≤20%) | 0% (≤20%) | 0% (≤49%) | 0% (≤49%) |

`decision_CIRCULAR` is the gate's own verdict — reported only to show how badly the circular definition inflates apparent success; it is not the headline.

## Threats to validity

- **Data-starvation (fatal to the headline):** the gated region holds only 15 run(s) and none show a statistically-supported improvement. 0% false-abort is absence of evidence — see the Wilson upper bounds.

- **Survivorship:** the archive contains only GEPA-completed runs; the pre-flight's abort was almost never recorded, so this is a reconstructed counterfactual on survivors, not a measurement of the deployed policy.
- **Circularity:** `decision` is the gate's own bootstrap-driven output and under `no_regression` collapses to `mean ≥ 0`; hence the primary definition is `lower_bound > 0`, not `decision`.
- **Proxy exactness:** `avg_baseline` equals the pre-flight's holdout score verbatim when the pre-flight ran (cache reuse), and is the same estimator on the same examples when it was skipped.
- **Heterogeneity:** per-example vectors index different synthetic datasets and holdout sizes; runs are treated as exchangeable units, never pooled at the example level. `n_examples` reported per bin.
- **Small-N / multiple comparisons:** every rate carries a Wilson interval; the definition×threshold grid is descriptive, not a battery of tests. The primary definition was fixed before reading outcomes.

## Data-collection recommendation (the actual fix)

The archive can never settle these thresholds because aborted runs and the pre-flight band+score were not logged. The fix ships alongside this report: `evolution/core/saturation_telemetry.py` writes one `output/saturation_ledger.jsonl` row per pre-flight — including aborts — joined to each run by `run_id`. Ledger status: **not yet written**. Re-run this script once the ledger accrues runs in the gated region with measured outcomes; the false-abort rate then becomes a real measurement rather than a survivorship counterfactual.

## Overfitting trajectory (forward-only)

0 `lineage.json` files in the archive — the lineage feature postdates the entire archive, so per-candidate val-vs-discovery-order analysis is forward-only and populates as new runs accrue. (Per-candidate holdout scores are never stored, so even forward the signal is 'val plateaus before the search budget is spent', not 'val climbs while holdout flattens'.)

## Out of scope

Closed-loop thresholds (`no_headroom_closed_loop=0.95`, `uniform_failure_closed_loop=0.15`): too few closed-loop runs in the archive to calibrate; they wait on the forward ledger. Changing any threshold value is a separate behavior change, not part of this measurement.

## Audit trail

Regenerate: `uv run python scripts/analysis/calibrate_saturation.py`. Machine-readable companion: `reports/saturation_calibration.json`.
