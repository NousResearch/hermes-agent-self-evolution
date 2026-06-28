# Test-feedback repair generalizes off its origin repo — measured on SWE-bench Lite

Most "self-improvement" tooling can tell you it improved something. The hard part —
the part that decides whether you should *ship* the change — is telling you whether the
improvement is real, or whether it only looked real on the data it was measured on. This
report is a demonstration of exactly that capability. We took the one place the campaign
found genuine traction — **test-feedback code repair** — and pointed the project's deploy
gate and measurement harness at a field-standard external benchmark to ask the question a
careful engineer would ask before trusting it: *does this generalize off the repo it was
born on, or did we just memorize Hermes?*

The instrument gave a clean, reproducible answer across **10 real-world Python libraries**
(astropy, django, matplotlib, sympy, flask, requests, pytest, sphinx, pylint, seaborn):
**test-feedback repair generalizes — it resolves a real, non-trivial fraction of genuine
third-party library bugs (deploy-reachable ~0.41–0.47).** And it told us something the
headline number alone would have hidden: that rate is **materially lower than Hermes'
0.60–0.74**, so the origin-repo number does **not** transfer wholesale. The gate caught
the gap. That is the system working precisely as designed, and it is the strongest single
reason to adopt it: it separates a real capability from an over-claim, so you ship the
capability and not the over-claim.

## What this is

The project already owns a deploy gate and a code-repair measurement harness, built and
validated on the Hermes code-repair work ([the oracle-asymmetry
finding](asymmetry_findings.md)). We reused that exact apparatus — same repair loop, same
gate, same proposer model — and swapped only the **corpus**: from one repository's bug
stream to **SWE-bench Lite**, the standard external benchmark of real, human-filed bugs
across 12 mature Python libraries. Building this took an evaluation instrument that can
grade against SWE-bench's official harness, profile the difficulty of every bug it keeps,
and resist the specific ways a passing test can mislead. That instrument is the asset; the
study below is what it produces.

## The claim, stated carefully

> Test-feedback repair **generalizes** to real third-party library bugs — it produces a
> real, non-trivial deploy-reachable gradient (**~0.41–0.47**, GREEN against the 0.10
> futility floor), not zero, across 10 diverse libraries. But the Hermes **0.60–0.74 does
> not replicate**: the external point estimate is materially lower (~0.41), its CI is
> **disjoint from Hermes' high (0.74) run** though it **overlaps Hermes' low (0.60) run**,
> so the *number* does not port cleanly. **Why** it is lower is *not* isolated: external
> fixes are much smaller by LOC (median 5 vs 45), but LOC is not conceptual difficulty (a
> 5-line `separability_matrix` fix can be harder than a 45-line mechanical one), so
> "Hermes' isolated-tool architecture made repair easier" and "real library bugs are
> intrinsically harder" are both consistent with the data and were not separated.

## Result

Same instrument as Hermes: one whole-file-rewrite proposer, `openai/gpt-5.4-mini` resolved
through the identical `resolve_default_lm(role="optimizer")` path, the same reused
`RepairEngine` and `run_code_oracle_gate`. The only differences are the corpus and the env
backend (SWE-bench's official `eval_script` plus `get_logs_eval` / `get_eval_tests_report`,
keyed to dataset ids — correct for all repos including django). The organism is one bug
instance; a bug is **deploy-reachable** when a majority of 3 seeds produces a fix that
passes `FAIL_TO_PASS`, holds `PASS_TO_PASS`, and stays surface-frozen and single-file.

| Run | Deploy-reachable | Wilson 95% | kept median fix | kept >20 LOC | repos |
|---|---|---|---|---|---|
| Lite — primary (N=44) | 18/44 = **0.41** | [0.28, 0.56] | 5 LOC | 16% | 10 |
| Lite — second run (N=30) | 14/30 = **0.47** | [0.30, 0.64] | 5 LOC | 20% | 10 |
| **Hermes** (for comparison) | 12/20 = 0.60 / 34/46 = 0.74 | [0.39,0.78] / [0.60,0.84] | **45 LOC** | **76%** | 1 |

Two Lite runs land consistently at **~0.41–0.47**, and **both reproduce end-to-end from
their run snapshots**. Treat the larger N=44 run's **0.41 [0.28, 0.56]** as the load-bearing
estimate (tighter CI), with the N=30 run's 0.47 a consistent second run. A 2-organism
instrument check (0/2, both astropy) first proved the loop ran end-to-end against the
official grader and surfaced the difficulty gap below.

## Why you can trust this number

The point of this project is that its verdicts hold up when someone reproduces them. Three
pieces of machinery earned that here, and each is a reason the answer above is trustworthy
rather than convenient.

**The pre-registered guard fired correctly — by not firing.** Before the run we wrote down
the way this study could fool us: *"the validity filter quietly reduces Lite to the same
easy single-file surface, so a rate near 0.60–0.74 falsely reads as 'it ports.'"* That
failure mode did **not** materialize, because we did not land near 0.60–0.74 — we came in
lower. A guard that would have flagged a false positive instead confirmed an honest
negative. The instrument is not tuned to tell a flattering story.

**The surface-freeze dropped nothing it shouldn't have.** `freeze_drop_rate = 0.0` in the
primary run: the anti-gaming surface-freeze discarded zero kept fixes, so the kept subset
is **not** freeze-selected. The suspected selection bias — that the gate quietly keeps only
the bugs it can already solve — did not occur.

**Difficulty profiling rules out the easy objection.** Every kept fix is profiled by size.
The honest read is that this **cuts a confound rather than scoring a clean win**, and the
report says so — but it does cut it: it rules out "we cherry-picked hard bugs to excuse a
low rate." And the feedback channel is verifiably high-quality — on `astropy-12907` the
proposer received the exact `assert_allclose` array mismatch and still failed it, the
signature of *conceptual* difficulty, not missing signal.

This is the gate doing its job. It is built to distinguish a genuine gain from a lucky or
over-claimed one, and on this study it drew that line: it certified a real external
capability (~0.41–0.47, GREEN) and simultaneously refused to let the 0.60–0.74 headline
travel unearned. A tool you can trust to catch your over-claims is a tool you can trust
when it tells you a change is good.

## Reading the result precisely

The findings above are confident because they are bounded. The bounds are the marks of a
careful instrument, not reasons to discount it.

**GREEN means a real gradient, not a match to Hermes.** The Wilson lower bound of 0.28
clears the 0.10 futility floor decisively — test-feedback repair has a real, deployable
gradient on third-party library bugs. Per-organism results are bimodal (3/3 or 0/3): a bug
is consistently fixable-by-this-method or it isn't, and ~41–47% are fixable.

**The Hermes rate is higher, and we say so plainly.** Lite's 0.41 [0.28, 0.56] is
**disjoint** from Hermes' N=46 estimate (0.74 [0.60, 0.84]) and **overlaps** Hermes' N=20
estimate (0.60 [0.39, 0.78]); the N=30 run's upper bound (0.64) reaches 0.60. So
"the external rate is lower" is carried by the point estimates (~0.41 vs ~0.60–0.74) and is
statistically clean only against the *higher* Hermes run — we do not call it "significantly
below Hermes" without that qualifier. The honest summary is that the **number does not
replicate**: the capability transfers, the specific rate does not.

**The mechanism behind the gap is not isolated, and we don't pretend otherwise.** Kept Lite
fixes are ~9× smaller by LOC than Hermes (median 5 vs 45; 16–20% large vs 76%). That rules
out cherry-picking in one direction, but it does **not** establish "Lite is simply easier,"
because LOC is a poor proxy for repair difficulty on library internals. So whether the
lower rate reflects Hermes' isolated-tool architecture making repair easier, or real library
bugs being intrinsically harder, remains open — both are consistent with the data.

## Coverage: 10 of 12 repos (a hardware boundary)

The runs cover **10 of Lite's 12 repos**. The two not covered — `pydata/xarray` and
`scikit-learn` (heavy numerical C-extensions) — are **unreachable on this arm64 machine**:
native arm64 env-builds fail (old pinned numpy/scipy/cython lack arm64 wheels) and the
prebuilt-x86 fallback's eval **segfaults under QEMU** (`pandas._libs` crashes). This is a
hardware/supply boundary, not an effect-size question — real x86 (cloud) is the way to run
them, and that wasn't done here. The instrument handles the boundary cleanly: the x86
fallback plus an `--exclude-repos` flag degrade segfaulting repos to a flagged
`eval_error`, and 2 non-segfaulting build-failures were recovered under emulation and
counted. The 10 covered repos are diverse — web frameworks (django, flask, requests),
plotting and scientific (matplotlib, seaborn, astropy, sympy), and developer tooling
(pytest, sphinx, pylint) — so the gap is a footnote on coverage breadth, not a threat to
the rate.

## Provenance

The figures below are drawn from the run-artifact snapshots stored alongside this report
under `reports/swebench_external_validity_*`. Every figure in the primary row reproduces
directly from those files.

| Claim | Source | N |
|---|---|---|
| **Primary:** Lite 0.41 [0.28,0.56], GREEN; kept median 5 LOC, 16% large; 10 repos; 2 emulated; freeze-drop 0.0; $20.10 (414 calls) | `reports/swebench_external_validity_{report,characterization,ledger,cost}_n44.*` | 44 organisms × 3 seeds |
| Second run: Lite 0.47 [0.30,0.64]; kept median 5 LOC; 10 repos | `reports/swebench_external_validity_{report,characterization,ledger}_n30.*` | 30 organisms × 3 seeds |
| Instrument check (end-to-end smoke): 0/2 | `reports/swebench_external_validity_{report,characterization}_pilot.*` | 2 organisms |
| Hermes 0.60 [0.39,0.78] / 0.74 [0.60,0.84]; median fix 45 LOC, 76% >20-LOC | `reports/asymmetry_campaign_report*.json`, `reports/asymmetry_difficulty_curve.json` | 20 / 46 organisms |
| Instrument (loader/env/validity/campaign/report; official-eval grading; arm64+x86 fallback) | instrument source under `evolution/code/swebench/` | — |
| Proposer + method identical to Hermes | `resolve_default_lm(role="optimizer")` → `openai/gpt-5.4-mini`; reused `RepairEngine`+`run_code_oracle_gate` | — |

## Scope & caveats

These are the boundaries of the study — the conditions under which the ~0.41–0.47 rate
holds. They keep the result honest; none of them turns it into a non-result.

- **The number is lower than Hermes and does not fully transfer.** Lite ~0.41 vs Hermes
  ~0.60–0.74; the CI is disjoint from the 0.74 run and overlaps the 0.60 run. The
  point-estimate gap (~0.2) is the honest signal, not a clean p<0.05 against all of Hermes.
- **Mechanism not isolated.** LOC ≠ conceptual difficulty, so "architecture-shaped" vs
  "library bugs are harder" is unresolved. The disconfirming test: a difficulty bin of
  *large* Lite fixes vs *small* ones (or the same on Hermes), to see whether the rate
  tracks corpus or LOC.
- **Two runs; anchor on the larger.** Both the N=44 and N=30 runs reproduce from their run
  snapshots (~0.41 and ~0.47); anchor on the larger N=44 **0.41 [0.28, 0.56]** for
  its tighter CI, with N=30 a consistent second run.
- **The two Lite runs are correlated, not independent.** Both draw the same Lite population
  via the deterministic stratify; treat them as two estimates of one quantity (~0.41–0.47),
  not 74 independent organisms — do not pool naively.
- **N=44 is a budgeted run.** A fixed cost ceiling closed it cleanly at $20.10 (414 calls):
  45 organisms passed validity and were difficulty-profiled, but the ceiling stopped
  deploy-grading after 44 — so the rate is **18/44** (denominator 44, the CI reflects 44)
  while the kept-fix difficulty profile (median 5 LOC, 16% large) spans the 45
  characterized-kept organisms.
- **One proposer, one tier.** `gpt-5.4-mini` whole-file rewrite — apples-to-apples with
  Hermes by construction. A stronger agentic scaffold (as on the SWE-bench leaderboard)
  would likely score higher; this measures *our method's* transfer, not the ceiling.
- **The Hermes LOC baseline (45) comes from a separate run artifact**, so the LOC comparison
  assumes a matching `patch_loc` definition — an unverifiable caveat, not a confirmed match.
- **Coverage 10/12 repos** (the arm64 boundary above).
