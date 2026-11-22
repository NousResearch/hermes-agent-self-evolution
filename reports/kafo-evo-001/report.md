# KAFO-EVO-001 sprint report

> **SYNTHETIC RESULTS. The simulator sections (gate 4 policy comparison, simulated cost and latency) use a hypothetical simulator with invented success rates, costs and latencies; the verifier reference scores use reviews derived from the benchmark labels, not a model. The live full run (live-1), the one-time sealed final evaluation and a 4-case smoke test used live Hermes (claude-sonnet-5-5[1m]) on the hand-written, synthetic python_review benchmark. Nothing in this report is evidence that Hermes improves on any real task: no result here is evidence of real-world improvement.**

## Headline findings

- Pre-registered sealed final evaluation (run once, live Hermes, 10 hand-written synthetic test cases): candidate `github-code-review@28dcec5ab528` minus baseline `github-code-review@11ad06e6d49b`, STRICT micro-F1 +0.302 [0.157, 0.462] (primary; CI lower bound above 0, so the pre-registered rule is met), mostly precision (precision +0.446 [0.260, 0.600], recall +0.111 [0.000, 0.375]); LENIENT micro-F1 +0.332 [0.194, 0.474] (secondary).
- Live full run live-1 (2-arm comparison on live Hermes outcomes, validation split): the best fixed strategy `fixed:skill` has utility 0.607; no adaptive policy improves on it: epsilon_greedy -0.032 [-0.039, -0.024] (worse); linucb -0.034 [-0.041, -0.026] (worse).
- GEPA candidate `github-code-review@28dcec5ab528` (parent `github-code-review@11ad06e6d49b`) was admitted by gate 6 with a validation utility gain of +0.127 [+0.006, +0.272]; gate 7's post-admission check then kept it (catalogue validation utility 0.617 -> 0.809).
- Simulator: No adaptive policy shows an improvement over the best fixed strategy (`fixed:structured`) on held-out validation: epsilon_greedy -0.006 [-0.024, +0.014] (no improvement detected); linucb +0.012 [-0.009, +0.033] (no improvement detected).
- On train, epsilon_greedy +0.025 [+0.007, +0.043]; linucb +0.041 [+0.026, +0.056] against `fixed:structured`. That baseline was chosen in hindsight on the same train data, so this does not show generalisation.
- Every adaptive policy beats uniform random selection in both phases.
- A live Hermes smoke run scored 4 validation cases (4 completed, 0 violations). With so few cases this is a plumbing check, not a quality estimate.
- The simulator findings are results from a hypothetical simulator, not from Hermes. The live full run (live-1), the one-time sealed final evaluation and a 4-case smoke test used live Hermes on a hand-written, synthetic benchmark, so none of this is evidence of real-world improvement.

## Policy comparison (gate 4, synthetic simulator)

Benchmark `synthetic-python-review/1`, executor `scripted`. 20 seeds x 300 online train episodes, then one pass over 60 validation tasks with learning off. Mean over seeds with a 95% bootstrap CI (2000 resamples, unit: seed-level run mean).

### Quality and utility

| Policy | Train utility | Validation utility | Validation quality |
|---|---|---|---|
| fixed:direct | 0.503 [0.492, 0.514] | 0.517 [0.517, 0.517] | 0.583 [0.583, 0.583] |
| fixed:structured | 0.565 [0.554, 0.575] | 0.584 [0.584, 0.584] | 0.783 [0.783, 0.783] |
| fixed:tool_assisted | 0.398 [0.384, 0.411] | 0.483 [0.483, 0.483] | 0.783 [0.783, 0.783] |
| random | 0.489 [0.478, 0.500] | 0.538 [0.518, 0.560] | 0.720 [0.700, 0.742] |
| epsilon_greedy | 0.590 [0.576, 0.604] | 0.579 [0.559, 0.598] | 0.751 [0.735, 0.767] |
| linucb | 0.606 [0.592, 0.619] | 0.596 [0.576, 0.617] | 0.760 [0.741, 0.778] |

Utility = verified quality - 0.10 x min(cost ratio, 2) - 0.05 x min(latency ratio, 2); a critical violation scores -1 (`core.score_outcome`, unchanged).

### Paired utility difference

Best fixed baseline, chosen by train utility: `fixed:structured`.

| Policy | Versus | Phase | Difference [CI] | Verdict |
|---|---|---|---|---|
| epsilon_greedy | fixed:structured | train | +0.025 [+0.007, +0.043] | improvement |
| epsilon_greedy | fixed:structured | validation | -0.006 [-0.024, +0.014] | no improvement detected |
| epsilon_greedy | random | train | +0.101 [+0.089, +0.112] | improvement |
| epsilon_greedy | random | validation | +0.041 [+0.009, +0.071] | improvement |
| linucb | fixed:structured | train | +0.041 [+0.026, +0.056] | improvement |
| linucb | fixed:structured | validation | +0.012 [-0.009, +0.033] | no improvement detected |
| linucb | random | train | +0.117 [+0.101, +0.133] | improvement |
| linucb | random | validation | +0.058 [+0.029, +0.088] | improvement |

## Cost and latency (simulated)

Invented per-strategy costs (USD) and latencies (seconds) from `synthetic.py`.

| Policy | Validation cost USD | Validation latency s | Train cost USD |
|---|---|---|---|
| fixed:direct | 0.0132 [0.0132, 0.0132] | 2.62 [2.62, 2.62] | 0.0133 [0.0133, 0.0133] |
| fixed:structured | 0.0397 [0.0397, 0.0397] | 8.00 [8.00, 8.00] | 0.0400 [0.0399, 0.0400] |
| fixed:tool_assisted | 0.1059 [0.1059, 0.1059] | 19.55 [19.55, 19.55] | 0.1065 [0.1063, 0.1068] |
| random | 0.0503 [0.0483, 0.0524] | 9.59 [9.22, 9.94] | 0.0532 [0.0524, 0.0539] |
| epsilon_greedy | 0.0457 [0.0420, 0.0492] | 8.81 [8.13, 9.53] | 0.0454 [0.0431, 0.0479] |
| linucb | 0.0393 [0.0370, 0.0418] | 7.69 [7.24, 8.17] | 0.0449 [0.0417, 0.0478] |

## Verifier scores on the open benchmark splits (gates 2 and 3)

Benchmark `python_review` v1 (manifest `fada5f14b7d8`), split sizes train 28, validation 10. Verifier `kafo-verifier/1`, rule `category-equal+line-overlap/1`, source `f3310b479ff6`. Case-level bootstrap CI, seed 0, 1000 resamples.

These reviews are references, not models: `ground_truth` reports exactly the labelled defects (ceiling), `empty` reports nothing (floor), and `noisy_reference` is an invented, seeded perturbation of the labels (keeps each defect with p=0.7, shifts lines by up to 1, adds a false alarm with p=0.25).

| Review | Split | Cases | TP / FP / FN | Micro precision | Micro recall | Micro F1 | Macro F1 |
|---|---|---|---|---|---|---|---|
| ground_truth | train | 28 | 31 / 0 / 0 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| ground_truth | validation | 10 | 10 / 0 / 0 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| empty | train | 28 | 0 / 0 / 31 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.143 [0.036, 0.250] |
| empty | validation | 10 | 0 / 0 / 10 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.200 [0.000, 0.500] |
| noisy_reference | train | 28 | 18 / 4 / 13 | 0.818 [0.650, 0.958] | 0.581 [0.412, 0.758] | 0.679 [0.522, 0.818] | 0.589 [0.422, 0.762] |
| noisy_reference | validation | 10 | 7 / 0 / 3 | 1.000 [1.000, 1.000] | 0.700 [0.400, 0.929] | 0.824 [0.571, 0.963] | 0.767 [0.500, 0.967] |

The empty review's macro F1 is above 0 because clean control cases (no defects) score 1.0 when nothing is reported.

## Uncertainty

- Policy comparison: percentile bootstrap over seed-level means; paired per-seed differences for verdicts (`improvement` only if the whole CI is above 0). Validation intervals for fixed policies have zero width because those runs are deterministic.
- Verifier: percentile bootstrap over cases (micro pools TP/FP/FN, macro averages per-case scores).
- Same code and seeds reproduce this report byte for byte.

## Safety

- Critical violations (tool call outside the strategy allowlist, simulated): 474 in 43200 simulated policy decisions (all seeds and phases). Each scores -1 and disqualifies the outcome.
- Sealed test split: 0 blocked access attempts while generating this report (everything ran inside `optimisation_phase()`).

| Policy | Train violations / steps | Validation violations / steps |
|---|---|---|
| fixed:direct | 0 / 6000 | 0 / 1200 |
| fixed:structured | 0 / 6000 | 0 / 1200 |
| fixed:tool_assisted | 307 / 6000 | 0 / 1200 |
| random | 96 / 6000 | 0 / 1200 |
| epsilon_greedy | 35 / 6000 | 0 / 1200 |
| linucb | 36 / 6000 | 0 / 1200 |

### Candidate registry (gate 5)

| Source | Status | Candidates |
|---|---|---|
| baseline | admitted | 1 |
| gepa | admitted | 1 |

| Candidate | Parent | Source | Status |
|---|---|---|---|
| `github-code-review@11ad06e6d49b` | - | baseline | admitted |
| `github-code-review@28dcec5ab528` | `github-code-review@11ad06e6d49b` | gepa | admitted |

Transitions: `github-code-review@11ad06e6d49b` new -> admitted (registry:baseline); `github-code-review@28dcec5ab528` new -> proposed (registry:gepa); `github-code-review@28dcec5ab528` proposed -> admitted (admission-gate:1c3cd3e28dd5).

Rejected candidates: 0. Retired: 0.

### Admission (gate 6)

Fixed policy `1c3cd3e28dd5` on the validation split, seeds [0, 1, 2], 2000 resamples. A proposed candidate is admitted only with:

- zero critical violations for the candidate over all runs
- mean utility gain over the baseline of at least 0.05
- lower bound of the 95% paired bootstrap CI of that gain above 0

Decisions are recorded in the candidate registry (above; from the live full-run snapshot unless --registry is given); the live run's decision and its evidence are in the live full run section.

### Checkpoints and rollbacks (gate 7)

Checkpoints by status: good 2. Log events: promote 1, save 2. Rollbacks: 0.

Checkpoint log: save checkpoint 1: admission:github-code-review@28dcec5ab528:pre; save checkpoint 2: admission:github-code-review@28dcec5ab528:post; promote checkpoint 2 (from 1): post-admission validation passed.

## Live Hermes smoke run

**4 live cases; a smoke test, not a benchmark result.** Plumbing check, not evidence that Hermes improves or performs well. One run, no confidence intervals. Cost is Hermes' estimate at API rates; the provider bills a subscription, so metered spend may be zero.

Source: live Hermes smoke run, 2026-11-22, owner-approved (PR #10); transcribed from docs/sprints/KAFO-EVO-001.md "Live smoke run", values rounded as published. Model: claude-sonnet-5-5[1m] via claude-subscription-directsdk-experimental, no toolsets. Split: validation.

| Case | Completed | Quality (F1) | Est. cost USD | Latency s | Violation |
|---|---|---|---|---|---|
| pyrev-007 | True | 0.50 | 0.0041 | 4.3 | False |
| pyrev-008 | True | 0.67 | 0.0045 | 3.7 | False |
| pyrev-031 | True | 1.00 | 0.0014 | 3.7 | False |
| pyrev-043 | True | 0.80 | 0.0063 | 5.2 | False |

- Completed: 4 / 4; violations: 0.
- Mean quality 0.743, mean cost USD 0.0041, mean latency s 4.2.

## Live full run (live-1)

**Live Hermes on the hand-written, synthetic benchmark: not evidence of real-world improvement.** Model `claude-sonnet-5-5[1m]`, provider `claude-subscription-directsdk-experimental`, budget 8 iterations / 300 s per call, run `kafo-full-run/1`. Baseline skill `github-code-review@11ad06e6d49b` from `hermes-agent@caafe5389a:skills/software-development/github/references/code-review.md`. Splits used: train, validation; sealed split open attempts: 0. Snapshot `live_run_live-1.json` (source file SHA-256s recorded in it).

Stages: admission done (467 s), canary done (13 s), compare done (377 s), gepa done (647 s).

### Live policy comparison (gate 4 on live outcomes)

2 arms (direct, skill). Each (case, strategy) pair ran once on live Hermes (73 fresh calls, 3 reused from the canary); 20 seeds x 300 train episodes over 28 train cases replay those outcomes, then one pass over 10 validation cases with learning off. 95% bootstrap CI over seeds (2000 resamples).

| Policy | Validation utility | Validation quality | Validation cost USD | Validation latency s |
|---|---|---|---|---|
| epsilon_greedy | 0.575 [0.567, 0.583] | 0.616 [0.609, 0.624] | 0.0042 [0.0041, 0.0042] | 3.29 [3.18, 3.39] |
| fixed:direct | 0.581 [0.581, 0.581] | 0.624 [0.624, 0.624] | 0.0038 [0.0038, 0.0038] | 3.65 [3.65, 3.65] |
| fixed:skill | 0.607 [0.607, 0.607] | 0.647 [0.647, 0.647] | 0.0043 [0.0043, 0.0043] | 3.13 [3.13, 3.13] |
| linucb | 0.573 [0.565, 0.581] | 0.614 [0.606, 0.621] | 0.0042 [0.0041, 0.0042] | 3.28 [3.21, 3.38] |
| random | 0.600 [0.588, 0.612] | 0.641 [0.630, 0.652] | 0.0041 [0.0041, 0.0041] | 3.36 [3.28, 3.43] |

Best fixed baseline, chosen by train utility: `fixed:skill`.

| Policy | Versus | Phase | Difference [CI] | Verdict |
|---|---|---|---|---|
| epsilon_greedy | fixed:skill | train | +0.008 [+0.001, +0.015] | improvement |
| epsilon_greedy | fixed:skill | validation | -0.032 [-0.039, -0.024] | worse |
| epsilon_greedy | random | train | +0.026 [+0.017, +0.035] | improvement |
| epsilon_greedy | random | validation | -0.025 [-0.041, -0.011] | worse |
| linucb | fixed:skill | train | +0.044 [+0.039, +0.048] | improvement |
| linucb | fixed:skill | validation | -0.034 [-0.041, -0.026] | worse |
| linucb | random | train | +0.062 [+0.057, +0.066] | improvement |
| linucb | random | validation | -0.027 [-0.040, -0.013] | worse |

### GEPA (gate 5 input)

`dspy.GEPA` on python_review v1 train split only (20 GEPA-train / 8 GEPA-val cases), max 150 metric calls, seed 0, task and reflection calls through `hermes:claude-subscription-directsdk-experimental:claude-sonnet-5-5[1m]`. GEPA's own validation score 0.675 -> 0.750; skill size 13211 -> 7465 characters; constraints passed: True; 638 s.

| GEPA program | Parent | Merge | Characters | SHA-256 | Registered as |
|---|---|---|---|---|---|
| 0 (seed) | - | no | 13210 | `7885961688b5` | - |
| 1 | 0 | no | 7464 | `ca35278bcbe9` | `github-code-review@28dcec5ab528` |
| 2 | 1 | yes | 8998 | `e4cb996b16d9` | - |
| 3 | 2 | no | 10237 | `d5daec3b43e0` | - |

### Admission and checkpoint (gates 6 and 7)

Candidate `github-code-review@28dcec5ab528` (parent `github-code-review@11ad06e6d49b`): **admitted** by gate 6, policy `1c3cd3e28dd5` (validation split, 10 cases, seeds [0, 1, 2], 2000 resamples). Mean utility gain +0.127 [+0.006, +0.272] (needs at least 0.05 and CI above 0); mean quality 0.716 -> 0.833; violations 0 / 0; unparseable outputs 0.

Gate 7: catalogue validation utility 0.617 before admission -> 0.809 after (30 tasks, critical violation: False); admitted and kept (status `kept`, registry `admitted`, active skill `github-code-review@28dcec5ab528`).

### Spend by stage (live Hermes estimates)

| Stage | Calls | Cost USD | Errors | Unknown cost | Retries | Mean latency s |
|---|---|---|---|---|---|---|
| admission | 120 | 0.3527 | 0 | 0 | 0 | 2.54 |
| canary | 3 | 0.0352 | 0 | 0 | 0 | 2.79 |
| compare | 73 | 0.3420 | 0 | 0 | 0 | 3.76 |
| final_eval_preflight:baseline | 1 | 0.0015 | 0 | 0 | 0 | 2.05 |
| final_eval_preflight:candidate | 1 | 0.0012 | 0 | 0 | 0 | 1.70 |
| final_test_evaluation:baseline | 10 | 0.0489 | 0 | 0 | 0 | 3.37 |
| final_test_evaluation:candidate | 10 | 0.0289 | 0 | 0 | 0 | 2.05 |
| gepa:reflection | 11 | 0.6571 | 0 | 0 | 0 | 21.48 |
| gepa:task | 92 | 0.5991 | 0 | 0 | 0 | 2.65 |

Total USD 2.0667 (Hermes' estimates at API rates; the provider bills a subscription). The final-evaluation stages are the one-time sealed run reported below.

## Final test evaluation (one-time, recorded)

Status **completed**, paired design, started 2026-11-22T19:58:35.764887+00:00, finished 2026-11-22T19:59:57.471797+00:00, reviewer `hermes:claude-sonnet-5-5[1m]:claude-subscription-directsdk-experimental:review`, synthetic: True. Baseline `github-code-review@11ad06e6d49b`, candidate `github-code-review@28dcec5ab528`. Pre-registered plan `kafo-evo-001-final-eval-plan/1` (sha256 `345f05baf846`); lenient table `category_alternates_v1.json` (sha256 `042fbfaca801`).

- baseline: model `claude-sonnet-5-5[1m]` (provider `claude-subscription-directsdk-experimental`), strategy `review:github-code-review@11ad06e6d49b`, cost USD 0.048931 (statuses {'estimated': 10}, shared cap 2.0), calls 10, completed 10, unparseable 1, violations 0, exceptions 0, skipped 0.

- candidate: model `claude-sonnet-5-5[1m]` (provider `claude-subscription-directsdk-experimental`), strategy `review:github-code-review@28dcec5ab528`, cost USD 0.028937 (statuses {'estimated': 10}, shared cap 2.0), calls 10, completed 10, unparseable 0, violations 0, exceptions 0, skipped 0.

10 sealed test cases, the same for both skills.

| Scoring | Skill | Micro F1 | Micro precision | Micro recall | Macro F1 |
|---|---|---|---|---|---|
| STRICT (primary) | baseline | 0.522 [0.320, 0.714] | 0.429 [0.263, 0.667] | 0.667 [0.375, 1.000] | 0.617 [0.383, 0.833] |
| STRICT (primary) | candidate | 0.824 [0.556, 1.000] | 0.875 [0.600, 1.000] | 0.778 [0.500, 1.000] | 0.867 [0.633, 1.000] |
| LENIENT (secondary) | baseline | 0.609 [0.500, 0.762] | 0.500 [0.389, 0.700] | 0.778 [0.545, 1.000] | 0.683 [0.483, 0.867] |
| LENIENT (secondary) | candidate | 0.941 [0.833, 1.000] | 1.000 [1.000, 1.000] | 0.889 [0.714, 1.000] | 0.967 [0.900, 1.000] |

Paired difference, candidate minus baseline, 95% paired bootstrap CI:

| Scoring | Micro F1 | Macro F1 | Micro precision | Micro recall |
|---|---|---|---|---|
| STRICT (primary) | 0.302 [0.157, 0.462] | 0.250 [0.067, 0.467] | 0.446 [0.260, 0.600] | 0.111 [0.000, 0.375] |
| LENIENT (secondary) | 0.332 [0.194, 0.474] | 0.283 [0.117, 0.483] | 0.500 [0.300, 0.611] | 0.111 [0.000, 0.375] |

Primary result: the strict micro-F1 difference's CI lower bound is above 0: the candidate is better on the sealed test under the pre-registered rule. Synthetic hand-written cases: not evidence of a real Hermes improvement.

## Pending


## Reproduce

```bash
python -m evolution.adaptive.report --seeds 20 --train-episodes 300 --benchmark-seed 0 --resamples 2000 --verifier-seed 0 --verifier-resamples 1000
```
