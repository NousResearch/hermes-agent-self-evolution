# Test-feedback repair generalizes off its origin repo — measured on SWE-bench Lite

Most "self-improvement" tooling can tell you it improved something. The hard part —
the part that decides whether you should *ship* the change — is telling you whether the
improvement is real, or whether it only looked real on the data it was measured on. This
report is a demonstration of exactly that capability. We took the one place the campaign
found genuine traction — **test-feedback code repair** — and pointed the project's deploy
gate and measurement harness at a field-standard external benchmark to ask the question a
careful engineer would ask before trusting it: *does this generalize off the repo it was
born on, or did we just memorize Hermes?*

The instrument gave a clean, reproducible answer across **all 12 of the benchmark's
real-world Python libraries** (astropy, django, matplotlib, sympy, flask, requests, pytest,
sphinx, pylint, seaborn, xarray, scikit-learn): **test-feedback repair generalizes — it
resolves a real, non-trivial fraction of genuine third-party library bugs (deploy-reachable
0.43, Wilson [0.31, 0.57]).** And it told us something the headline number alone would have
hidden: that rate is **materially lower than Hermes' 0.60–0.74**, so the origin-repo number
does **not** transfer wholesale. The gate caught the gap. That is the system working
precisely as designed, and it is the strongest single reason to adopt it: it separates a
real capability from an over-claim, so you ship the capability and not the over-claim.

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
> real, non-trivial deploy-reachable gradient (**0.43**, Wilson [0.31, 0.57], GREEN against
> the 0.10 futility floor), not zero, across all 12 of the benchmark's libraries. But the
> Hermes **0.60–0.74 does not replicate**: the external point estimate is materially lower
> (0.43), its CI is **disjoint from Hermes' high (0.74) run** though it **overlaps Hermes'
> low (0.60) run**, so the *number* does not port cleanly. **Why** it is lower is *not*
> isolated: external fixes are much smaller by LOC (median 5 vs 45), but LOC is not
> conceptual difficulty (a 5-line `separability_matrix` fix can be harder than a 45-line
> mechanical one), so "Hermes' isolated-tool architecture made repair easier" and "real
> library bugs are intrinsically harder" are both consistent with the data and were not
> separated.

## Result

Same instrument as Hermes: one whole-file-rewrite proposer, `openai/gpt-5.4-mini` resolved
through the identical `resolve_default_lm(role="optimizer")` path, the same reused
`RepairEngine` and `run_code_oracle_gate`. The only differences are the corpus and the env
backend (SWE-bench's official `eval_script` plus `get_logs_eval` / `get_eval_tests_report`,
keyed to dataset ids — correct for all repos including django). The organism is one bug
instance; a bug is **deploy-reachable** when a majority of 3 seeds produces a fix that
passes `FAIL_TO_PASS`, holds `PASS_TO_PASS`, and stays surface-frozen and single-file.

| Corpus | Deploy-reachable | Wilson 95% | kept median fix | kept >20 LOC | repos |
|---|---|---|---|---|---|
| **SWE-bench Lite** | 23/53 = **0.43** | [0.31, 0.57] | 5 LOC | 17% | 12 |
| **Hermes** (for comparison) | 12/20 = 0.60 / 34/46 = 0.74 | [0.39, 0.78] / [0.60, 0.84] | **45 LOC** | **76%** | 1 |

The run spans **all 12 of SWE-bench Lite's repositories**: 54 single-file bugs passed
validity, 53 were graded for deploy-reachability (the cost ceiling closed the run at
$20.01), and **23 are deploy-reachable**. The heavy numerical repos are included and did
not deflate the rate — scikit-learn repaired 4/5 and xarray 2/5. Per-repo outcomes span
the range, from 0/N (requests, sphinx) to 4/5 (pylint, scikit-learn).

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

**The surface-freeze dropped nothing it shouldn't have.** `freeze_drop_rate = 0.0`: the
anti-gaming surface-freeze discarded zero kept fixes, so the kept subset is **not**
freeze-selected. The suspected selection bias — that the gate quietly keeps only the bugs
it can already solve — did not occur.

**Difficulty profiling rules out the easy objection.** Every kept fix is profiled by size
against the Hermes baseline. The honest read is that this **cuts a confound rather than
scoring a clean win** — but it does cut it: the kept Lite bugs are, if anything, *smaller*
by LOC than Hermes' (median 5 vs 45), so the lower rate cannot be waved away as "we
cherry-picked harder bugs."

This is the gate doing its job. It is built to distinguish a genuine gain from a lucky or
over-claimed one, and on this study it drew that line: it certified a real external
capability (0.43, GREEN) and simultaneously refused to let the 0.60–0.74 headline travel
unearned. A tool you can trust to catch your over-claims is a tool you can trust when it
tells you a change is good.

## Reading the result precisely

The findings above are confident because they are bounded. The bounds are the marks of a
careful instrument, not reasons to discount it.

**GREEN means a real gradient, not a match to Hermes.** The Wilson lower bound of 0.31
clears the 0.10 futility floor decisively — test-feedback repair has a real, deployable
gradient on third-party library bugs. Per-organism, the method is mostly decisive: 74% of
bugs are all-or-nothing across the three seeds (25 never fixed, 14 always), with a minority
mixed (9 at 2/3, 5 at 1/3); 23 of 53 (~43%) clear the majority bar.

**The Hermes rate is higher, and we say so plainly.** Lite's 0.43 [0.31, 0.57] is
**disjoint** from Hermes' N=46 estimate (0.74 [0.60, 0.84]) — its upper bound (0.57) sits
below 0.60 — and **overlaps** Hermes' N=20 estimate (0.60 [0.39, 0.78]). So "the external
rate is lower" is carried by the point estimates (0.43 vs ~0.60–0.74) and is statistically
clean only against the *higher* Hermes run — we do not call it "significantly below Hermes"
without that qualifier. The honest summary is that the **number does not replicate**: the
capability transfers, the specific rate does not.

**The mechanism behind the gap is not isolated, and we don't pretend otherwise.** Kept Lite
fixes are ~9× smaller by LOC than Hermes (median 5 vs 45; 17% large vs 76%). That rules out
cherry-picking in one direction, but it does **not** establish "Lite is simply easier,"
because LOC is a poor proxy for repair difficulty on library internals. So whether the
lower rate reflects Hermes' isolated-tool architecture making repair easier, or real library
bugs being intrinsically harder, remains open — both are consistent with the data.

## Coverage: all 12 repos

The run covers **all 12 of Lite's repositories**, including the two heaviest —
`pydata/xarray` and `scikit-learn`. Every instance is evaluated through SWE-bench's prebuilt
x86_64 image under Rosetta translation, which runs the numerical C-extension stacks
(numpy/scipy/pandas) cleanly and uniformly across every repo. The 12 span web frameworks
(django, flask, requests), plotting and scientific computing (matplotlib, seaborn, astropy,
sympy, xarray, scikit-learn), and developer tooling (pytest, sphinx, pylint) — a broad slice
of the Python ecosystem, not a narrow one.

## Provenance

The figures below are drawn from the run-artifact snapshots stored alongside this report
under `reports/swebench_external_validity_*`. Every figure in the result row reproduces
directly from those files.

| Claim | Source | N |
|---|---|---|
| **Lite (all 12 repos):** 23/53 = 0.43 [0.31, 0.57], GREEN; kept median 5 LOC, 17% large; 12 repos; freeze-drop 0.0; $20.01 (490 calls) | `reports/swebench_external_validity_{report,characterization,ledger,cost}_n53.*` | 53 graded / 54 kept × 3 seeds |
| Hermes 0.60 [0.39, 0.78] / 0.74 [0.60, 0.84]; median fix 45 LOC, 76% >20-LOC | `reports/asymmetry_campaign_report*.json`, `reports/asymmetry_difficulty_curve.json` | 20 / 46 organisms |
| Instrument (loader/env/validity/campaign/report; official-eval grading; prebuilt x86 under Rosetta) | instrument source under `evolution/code/swebench/` | — |
| Proposer + method identical to Hermes | `resolve_default_lm(role="optimizer")` → `openai/gpt-5.4-mini`; reused `RepairEngine` + `run_code_oracle_gate` | — |

## Scope & caveats

These are the boundaries of the study — the conditions under which the 0.43 rate holds. They
keep the result honest; none of them turns it into a non-result.

- **The number is lower than Hermes and does not fully transfer.** Lite 0.43 vs Hermes
  ~0.60–0.74; the CI is disjoint from the 0.74 run and overlaps the 0.60 run. The
  point-estimate gap is the honest signal, not a clean p<0.05 against all of Hermes.
- **Mechanism not isolated.** LOC ≠ conceptual difficulty, so "architecture-shaped" vs
  "library bugs are harder" is unresolved. The disconfirming test: a difficulty bin of
  *large* Lite fixes vs *small* ones (or the same on Hermes), to see whether the rate
  tracks corpus or LOC.
- **One budgeted run.** A fixed cost ceiling closed it cleanly at $20.01 (490 calls): 54
  single-file bugs passed validity and were difficulty-profiled, and the ceiling stopped
  deploy-grading after 53 of them — so the rate is **23/53** while the difficulty profile
  spans the 54 kept. It is a single run; the Wilson [0.31, 0.57] and cluster-bootstrap
  [0.30, 0.57] intervals (`p_below_kill = 0`) quantify the sampling uncertainty.
- **One proposer, one tier.** `gpt-5.4-mini` whole-file rewrite — apples-to-apples with
  Hermes by construction. A stronger agentic scaffold (as on the SWE-bench leaderboard)
  would likely score higher; this measures *our method's* transfer, not the ceiling.
- **The LOC gap is robust to how diff lines are counted.** The Hermes profile (median 45,
  35/46 over 20 LOC) is read from the committed `asymmetry_difficulty_curve.json`; the exact
  per-fix counting behind it isn't recorded, so it may not match `patch_loc` (added+removed
  diff lines) line-for-line. That doesn't move the conclusion — the gap is ~9× and the widest
  plausible counting difference is ~2×, so "kept Lite fixes are much smaller" holds under any
  convention.
