# A population-based evolver does not beat best-of-N at equal budget

The framework generates candidate changes and certifies them with a deploy gate; the recurring question is whether a richer *search* over candidates would find improvements that simple resampling misses. The strongest off-the-shelf answer is a population-based evolutionary loop — a maintained population, weighted parent selection, a cross-population learning log, post-mutation verification, recombination. This evaluates the actual [`imbue-ai/darwinian_evolver`](https://github.com/imbue-ai/darwinian_evolver) (AGPL-3.0), given its best shot, against best-of-N resampling at equal budget, on the bugs our single-proposer loop fails.

The result is decisive: **the evolver recovers fewer bugs than best-of-N, recovers nothing best-of-N misses, and its distinguishing machinery contributes nothing measurable.** An earlier test used a simplified proxy of population search and reached the same conclusion; this confirms it on the real tool, more cleanly.

## Method

The evolver is wired to the same machinery the shipped loop uses, so the *only* new variable is the population/selection layer:

- **Oracle (fitness) = the production code deploy gate** — bug-tests pass, oracle-match over the full upstream fix-commit test file, surface freeze, file scope, and a baseline-diff regression floor. Partial credit (fraction of bug-tests passing) guides the search, but only a real gate **deploy** clears half the score, so the search cannot bank fitness on a gate-failing local optimum.
- **Mutator = the same DSPy proposer the shipped loop uses** — identical mutation primitive across all arms. Unique per-draw tags and disabled response caching make every sample a genuinely independent draw.
- **Population = the 23 organisms the single-proposer loop fails** — authentic upstream bug→fix commits where the fix-commit test file is the generalization oracle.

Three arms, equal candidate budget of 9 draws per organism, 3 seeds each; a recovery counts as **robust** only if it reproduces across a majority of seeds:

- **Best-of-N (incumbent):** 9 independent proposer draws, gate each, stop on first deploy.
- **Controlled evolver:** the real tool with its learning log and verification *off* (3 parents × 3 iterations = 9 mutator calls) — isolates whether the population/selection machinery alone beats independent sampling.
- **Best-shot evolver:** the real tool with its intended levers *on* — ancestor learning log, post-mutation verification (pytest-only here, so it spends no candidate budget), full population dynamics.

The evaluator and verifier make zero LLM calls; all cost is the proposer. The cost ledger counts real API calls as the source of truth.

## Results

Robust recovery over the 23 organisms (3 seeds each):

| Arm | Recovered | Rate | Wilson 95% CI | LLM calls |
|-----|-----------|------|---------------|-----------|
| **Best-of-N (incumbent)** | **8 / 23** | **34.8%** | [18.8%, 55.1%] | **612** |
| Controlled evolver (levers off) | 6 / 23 | 26.1% | [12.5%, 46.5%] | 746 (+22%) |
| Best-shot evolver (levers on) | 6 / 23 | 26.1% | [12.5%, 46.5%] | 763 (+25%) |

Per-organism recovery, best-shot evolver vs best-of-N:

|                   | best-of-N recovered | best-of-N missed |
|-------------------|:-------------------:|:----------------:|
| **evolver recovered** | 6 (both)        | **0 (evolver-only)** |
| **evolver missed**    | **2 (bon-only)**| 15 (neither)     |

McNemar two-sided exact **p = 0.50** — the two discordant organisms both favor best-of-N. The controlled arm yields the identical 2×2. Total cost **$65.21**.

## Reading the result

1. **The evolver recovered fewer, not more, and nothing uniquely.** Best-of-N's recovery set strictly contains the evolver's: every organism the evolver fixed, best-of-N also fixed, plus two more. There is no bug the search reaches that resampling cannot.
2. **The distinguishing machinery is inert.** The controlled arm (no learning log, no verification) and the best-shot arm (both on) land on the *identical* 6/23 and the *identical* 2×2. The learning log, weighted parent selection, and post-mutation verification produced no measurable lift over a bare population loop on this corpus.
3. **The budgets favored the evolver.** Best-of-N stops on its first deploy, so it spent ~22–25% *fewer* calls; on the organisms it missed it ran the full draw budget. The evolver had more compute and more chances to recover, and still recovered fewer — so the comparison is conservative, and strict parity could only widen best-of-N's lead.
4. **Statistical honesty.** The Wilson intervals overlap, so this is not "best-of-N is significantly better." It is that the burden was on the engine to demonstrate a lead, and it demonstrated a deficit on every axis with zero unique recoveries. A search engine that needs more budget to find a subset of what resampling already finds is not worth adopting.

**The value is the signal plus the gate, not the search.** On this corpus, the population/selection layer adds nothing over best-of-N, and best-of-N adds nothing over the candidates the proposer already produces. What moves the needle is installing a signal the agent cannot infer and certifying it did not regress — not the sophistication of the search over candidates.

## What this means for autonomous self-improvement

A search engine does not lift either constraint that bounds autonomous self-improvement:

- **Verification.** In this evaluation, the gate and the upstream-fix test file *are* the evolver's fitness function. A self-directed loop would have to author its own oracle — and a leakage check shows roughly two-thirds of any gate pass is the test's own expected values, not a discovered invariant. A better optimizer over a missing oracle is still nothing.
- **Supply.** The evolver consumes a feed of reproducible bugs. Across a corpus of 836 agent sessions and ~41k tool calls, that feed is effectively empty of tool-internal bugs. Search does not manufacture a bug stream.

So even a win would have meant only "a better candidate generator feeding the gate on harvested bugs" — not autonomy. There was no win: at equal budget the evolver is no better than, and slightly worse than, the best-of-N already in place. The gate remains the asset; bug supply remains the bottleneck.

## Scope and reproducibility

One mutation substrate (whole-file tool code), one repository, 23 organisms, one proposer model. The driver imports the AGPL tool only in an isolated, unshipped evaluation harness — the tool is never imported into the framework's MIT modules and is added to the environment only at run time. The tool itself makes no LLM calls; all 2,121 calls and $65.21 are the proposer. Per-organism seeds, call counts, and the 2×2 are recorded alongside this report.
