# Pareto Frontier Extension — Feasibility Probe

**Question:** Can we close PLAN.md Phase 2 deviation #8 (behavioral-space frontier extension) without forking GEPA? How much work?

**Verdict:** **YELLOW**, but with a much cheaper **Step 0** to run first. If Step 0 succeeds, the work is days, not weeks. If it fails, we're at 2-4 weeks across two viable paths.

---

## The structural gap (confirmed)

GEPA's frontier *storage* is per-instance: `pareto_front_valset: dict[DataId, float]` and `program_at_pareto_front_valset: dict[DataId, set[ProgramIdx]]` (`gepa/core/state.py:154-155`). A candidate joins a val_id's frontier iff it strictly improves that instance's score.

But GEPA's *acceptance gate* (whether a proposed candidate even enters the pool) is aggregate. From `gepa/core/engine.py:491-493`:

```python
old_sum = sum(proposal.subsample_scores_before or [])
new_sum = sum(proposal.subsample_scores_after or [])
if new_sum <= old_sum:
    # rejected
```

So acceptance is `sum()`-based on the minibatch; frontier membership is per-instance. Two distinct gates. Saturated judge scores defeat the *acceptance* gate; the *frontier* gate would still discriminate if behavioral tasks were per-instance entries.

PR #52 added behavioral tasks as training instances (`evolution/core/behavioral_example.py:26-37`, `evolve_skill.py:869-880`), and the metric routes behavioral examples through `_score_behavioral_example` (`fitness.py:165`), returning 1.0/0.0. So behavioral tasks **do** contribute to the subsample sum if drawn into the minibatch — they can break ties at the acceptance gate when present. And if they're in the **valset**, they're per-instance frontier entries under the default `frontier_type="instance"`.

**The key default we found:** `closed_loop_in_valset: bool = False` (`evolve_skill.py:606`, `evolve_tool.py:366`). By default, behavioral tasks live only in the trainset — they shape the subsample sum sometimes (via batch sampling) but never enter the per-valset frontier. That's almost certainly why selection didn't move on the saturated `write_file` / `search_files` runs.

## Step 0 — re-run saturated test with `--closed-loop-in-valset` (1 day, $5-20)

Before any new code, prove the hypothesis with the flag we already have:

1. Reproduce the Phase 2 saturated `write_file` (or `search_files`) run that picked the baseline byte-for-byte
2. Add `--closed-loop-in-valset` and `--closed-loop-mode trainset` (or `both`)
3. Compare: does the evolved description now differ from baseline? Does the per-valset frontier in `gate_decision.json.knee_point.candidates` show behavioral tasks as discriminating axes?

**Outcomes:**
- **Green** — evolved differs from baseline, behavioral wins drove selection: ship a docs/default change (flip `closed_loop_in_valset` default to True, or document it as the production-recommended flag), close deviation #8. Total work: ~2 days including the rerun + doc + CHANGELOG.
- **Yellow** — frontier shows the behavioral entries but selection still ties: confirms the cartesian-objective extension is the actual fix. Escalate to Path A.
- **Red** — flag has no effect / breaks something else: confirms the deeper gap and we need Path A or B.

This Step 0 costs almost nothing and disambiguates which path matters.

## Path A — Cartesian objectives via DSPy wrapper extension (1-2 weeks IF Step 0 yellow)

`FrontierType` in `gepa/core/state.py:22` already includes `"cartesian"`. With it, GEPA tracks `pareto_front_cartesian: dict[tuple[DataId, str], float]` keyed by (example, objective) (`state.py:158`). An evaluator returning `EvaluationBatch.objective_scores: list[dict[str, float]]` (`gepa/core/adapter.py:34`) lights this up.

**But the public surface doesn't expose it.** Both `gepa/api.py` and DSPy's `dspy.GEPA(...)` (`dspy/teleprompt/gepa/gepa.py`) lack any reference to `frontier_type` or `objective_scores`. The plumbing exists in GEPA internals but isn't reachable through the wrapper we use.

So Path A actually requires one of:
- **Upstream PR to DSPy** adding `frontier_type` + `objective_scores`-passthrough kwargs to `dspy.GEPA(...)`. Small, mechanical. ~1 week including review.
- **Custom GEPA invocation in our code** that bypasses `dspy.GEPA` and constructs the adapter directly. Larger surface but no upstream dependency. ~1-2 weeks.

In either case, our side is ~40 lines: emit a `behavioral_pass` objective per example in the fitness metric, fold it into `EvaluationBatch.objective_scores`.

## Path B — New `behavioral` frontier type upstream in GEPA (3-4 weeks)

Add `"behavioral"` to the FrontierType literal, a parallel `program_at_pareto_front_behavioral_tasks` slot in GEPAState, and the corresponding updater. Cleaner semantics than Path A's repurpose-cartesian-for-this. But it's a real GEPA PR with a maintainer review cycle.

Only worth it if Path A's cartesian semantics turn out to be awkward in practice (e.g., we want different acceptance behavior for behavioral vs. judge objectives, which cartesian doesn't model).

## What the probe should NOT do

Skip the spike — even Path A is wasted work if Step 0 is green.

## Recommendation

Sequence:

1. **Day 1-2:** Run Step 0. Re-run the `write_file` saturated case with `--closed-loop-in-valset --closed-loop-mode trainset`. Document the result here.
2. **If green:** flip the default, document the flag as production-recommended, close deviation #8, move on to the diagnostic CLI (next item in the hybrid sequencing).
3. **If yellow/red:** commit to Path A. Spike for 1 week against the cartesian path; if it works, decide whether to push the DSPy PR or keep the custom adapter.
4. **Defer Path B** unless Path A reveals a semantic mismatch we can't paper over.

The original "2-3 weeks" budget in the recon plan stands as an upper bound but is likely overestimated by 2-3x. Most-likely outcome: ~1 week of work plus a docs change.

## Open questions for the spike

- Does `closed_loop_in_valset=True` change the cost shape materially? Validator runs are expensive; in-valset means every full eval pays for them.
- Does the cartesian path interact correctly with our knee-point picker (`evolution/skills/knee_point.py`), which reads `pareto_front_valset` from GEPA's `detailed_results`? It may need to also read `pareto_front_cartesian`.
- MIPROv2 fallback has no frontier at all; degraded behavior is unchanged regardless of which path we pick.

---

## Step 0 result — **RED for the original hypothesis, GREEN for understanding the real blocker**

**Run:** `evolve_tool --tool write_file --iterations 5 --closed-loop-during-evolution evolution/validation/suites/write_file.jsonl --closed-loop-hermes-repo /Users/justin/src/NousResearch/hermes-agent --closed-loop-mode trainset --closed-loop-in-valset --max-total-cost-usd 20 --seed 42`
**Cost:** $0.31. **Wall time:** 245 s. **Hermes-agent state after run:** clean (restore worked).
**Output:** `output/tools/write_file/20260520_201914/`.

### What happened
- Synthetic dataset built: 49 train / 39 val / 50 holdout (138 total).
- **Behavioral examples DID reach the valset.** GEPA logs `Using 46 examples for tracking Pareto scores`. 39 synthetic val + 7 behavioral (`evolution/validation/suites/write_file.jsonl` has 7 tasks) = 46. The `--closed-loop-in-valset` plumbing works exactly as designed.
- **Baseline scores 1.0 on all 46 valset examples** (Iteration 0 log: `Base program full valset score: 1.0 over 46 / 46 examples`). Both the 39 synthetic tool-selection tasks AND the 7 behavioral closed-loop tasks pass perfectly on the baseline description with the default `gpt-5.4-mini` agent.
- Every GEPA iteration logs `Iteration N: All subsample scores perfect. Skipping. Reflective mutation did not propose a new candidate`. GEPA never even invokes the reflection LM — there's no failure to reflect on.
- Holdout: 0.987 → 0.987 (+0.000). Description: 216 → 216 chars (byte-identical). Decision: deploy by `no_regression_only` (trivially).
- **Closed-loop validator never fired.** Because GEPA never proposed a new candidate, there was nothing to validate. (The validator does fire when a real candidate appears; with no candidate, the cache sits idle.)

### Why the original hypothesis was wrong

The deviation #8 framing assumed: "judge saturates, but behavioral tasks fail → frontier extension into behavioral space lets behavioral wins break judge ties." That's a two-condition assumption: (a) judge saturates AND (b) behavioral tasks discriminate.

**Reality: condition (b) doesn't hold on this baseline.** With a sufficiently capable agent model (`gpt-5.4-mini`) and a well-tuned baseline description, the agent picks the right tool even on the behavioral tasks too. Behavioral and judge signals are correlated, not orthogonal. There is no behavioral signal to extend the frontier into.

This means **neither Path A (cartesian) nor Path B (new frontier type) would help on this case**. Both presume non-uniform behavioral scores across candidates. With uniform 1.0, frontier shape is moot.

### The actual lever — surface signal, then mechanism matters

The framework already exposes the right knob: `--closed-loop-agent-model`. The flag exists explicitly for this: "Useful when your daily-driver Hermes model saturates the planted-bug suite at 100%, hiding the behavioral signal — run validation against a weaker model without touching your config" (CLI help string at `evolve_tool.py:1204-1209`). The team that built the closed-loop knew this could happen; the docs and code knew before us.

The blocker for the deviation #8 fix is upstream of GEPA's frontier mechanism. To get behavioral signal:
1. **Run the validator against a weaker agent model** so its tool selection has failure modes the baseline description can fix.
2. **OR** harden the behavioral suite with more adversarial tasks (descriptions that are subtly wrong about boundary cases — currently the suite has clean disambiguating tasks but the agent reads them well anyway).
3. **OR** move to a non-binary verdict — e.g., did the agent pick the optimal tool on the first try vs. only after backtracking; how confidently; how many wrong-tool calls before the right one.

Frontier extension (Path A/B) only matters AFTER signal exists. On this saturated baseline, even the existing per-instance frontier (which behavioral examples already join when `--closed-loop-in-valset` is on) can't discriminate because all candidates tie at 1.0.

### Revised verdict and next step

**Verdict:** YELLOW → escalated to **investigative**, not implementation. The "2-week Path A spike → 3-4 week Path B" recommendation from the original probe is premature. Mechanism-side fixes don't move selection on the current saturated baselines because there's no signal to mechanism over.

**Next-cheapest step ($5-15):** Re-run with `--closed-loop-agent-model` set to a weaker model (e.g., `openai/gpt-5-nano` or `openai/gpt-4o-mini`) keeping the same suite + in-valset flag. If a weaker validator agent produces non-uniform behavioral scores, then Path A becomes relevant. If even a weaker model still saturates the suite, the suite itself needs hardening.

**What to update in PLAN.md deviation #8:** the structural fix described there is necessary but not sufficient. Behavioral signal must exist first; the frontier mechanism only matters in cases where it does.

**Hidden win from this spike:** the `--closed-loop-in-valset` plumbing, the `SkillFileInstaller`-equivalent tool installer, and the cleanup/restore safety all worked end-to-end on a real run. Infrastructure is sound; the gap is in signal generation, not selection mechanism.

---

## Step 0.5 result — **deviation #8 confirmed and reframed**

**Run:** same command as Step 0 plus `--closed-loop-agent-model openai/gpt-5-nano`.
**Cost:** $1.17 tracked + ~$3-15 untracked `hermes -z` subprocess (the framework doesn't track subprocess LM spend — separate gap).
**Wall time:** 2338 s (39 min). **Output:** `output/tools/write_file/20260520_203304/`.

### What happened

- GEPA ran **116 iterations** (vs spike #1's effectively-zero), exhausted 508/510 rollouts.
- Reflection LM fired and proposed **40 distinct candidates**.
- Acceptance gate **rejected all 40**: 38× `New subsample score 2.0 is not better than old score 2.0, skipping`, 2× `1.0 is not better than 1.0`.
- **Iteration 0 base full-valset score: 0.8478** = exactly 39/46. The 7 behavioral tasks failed uniformly for the baseline; the 39 synthetic tasks all passed (judge still saturated).
- Knee-point picker received 1 candidate in band (the baseline) at val=0.848 and held the line. Holdout: 0.987 → 0.987 (judge-saturated, behavioral examples are train/val only). Manifest byte-identical to baseline.

### What the failure pattern proves

This is **the deviation #8 mechanism caught red-handed**, not as I first interpreted it. The fix path the original probe described (cartesian frontier / new frontier type) addresses **candidate selection** — which parent GEPA mutates from. But the actual bottleneck on saturated baselines is **candidate acceptance** — whether any new proposal enters the candidate pool in the first place.

GEPA's acceptance rule (`gepa/core/engine.py:491-493`):

```python
old_sum = sum(proposal.subsample_scores_before or [])
new_sum = sum(proposal.subsample_scores_after or [])
if new_sum <= old_sum:
    # rejected
```

The subsample is small (2-3 examples per `gepa/strategies/batch_sampler.py`'s default). With 7/46 = 15% of valset failing for the baseline, a random 3-example minibatch contains a failing example only ~38% of the time. The other 62%, the minibatch is all-synthetic; both candidates score sum=N.0=N.0; rejected. Even when a behavioral example IS sampled, if the new candidate also fails it (uniform failure with weak validator), the sum still ties.

In other words: **the signal exists per-instance, but stochastic small-minibatch + sum-strict acceptance discards it before it can move selection**. No frontier mechanism on the world helps if proposals can't pass acceptance to be frontier-evaluated.

### Validator-model sensitivity is real and tricky

The validator agent model is a Goldilocks knob:
- `gpt-5.4-mini` (spike #1): too strong → all 7 behavioral tasks pass → signal = 0
- `gpt-5-nano` (spike #2): too weak → all 7 behavioral tasks fail uniformly → per-instance signal exists but isn't *discriminating across candidates*

For deviation #8 to actually move selection, we need a validator model in the middle band where behavioral tasks fail for the baseline but pass for some candidate descriptions. That's a real product surface — `--closed-loop-agent-model` exists for it, but users have no guidance on how to pick.

### Revised fix paths (actual, not the original probe's)

The original probe identified Path A (cartesian via DSPy PR) and Path B (new GEPA frontier type). Both address selection, neither addresses acceptance. With acceptance as the real bottleneck, the cheaper, more direct paths are:

**Path C — Stratified minibatch sampling (1-2 weeks, GEPA-side change).** Force every minibatch to include at least one currently-failing example. Lives in `gepa/strategies/batch_sampler.py`. With the failing example always sampled, a candidate that fixes it will measurably out-sum the baseline. Compatible with the existing `sum > sum` acceptance.

**Path D — Pareto-dominance acceptance (1 week, GEPA-side change).** Replace `sum(new) > sum(old)` with weak-Pareto-dominance + strict-improvement-on-at-least-one. ~5 line change in `engine.py:491-493`. Catches candidates that win on a behavioral task without losing on synthetic. Most conservative version; doesn't trade off.

**Path E — Larger minibatch (trivial, GEPA-side change).** Bump default minibatch from 2-3 to 8+. With 15% failures, 8-example minibatch contains a failing example 73% of the time. Simplest possible fix; works with existing acceptance. Cost: 3x more eval calls per iteration.

All three need upstream changes — either a GEPA PR (cleanest) or a custom adapter in our code that bypasses `dspy.GEPA`. None require forking GEPA.

**Path F — Saturation pre-flight (1 day, OUR code, ships immediately).** Independent of any GEPA work: before spending GEPA budget, run a fast probe (e.g., 10 synthetic + the closed-loop suite) and surface a Rich panel:
- "Baseline scores 1.00 on synthetic + 1.00 on closed-loop → no measurable headroom; consider a different optimization target or harder suite."
- "Baseline scores 0.99 on synthetic + 0.85 on closed-loop → behavioral signal exists but small; expect GEPA to need more iterations or a larger minibatch."
- "Baseline scores 0.62 on synthetic → strong signal; standard config will likely improve."

This converts a $1+ wasted run into a $0.05 "save your money, try X" panel. Stand-alone valuable regardless of which mechanism fix lands.

### Recommended sequence

1. **Day 1-2:** Ship Path F (saturation pre-flight). Independent of GEPA, immediately valuable, prevents the kind of mystery this spike just diagnosed.
2. **Week 1:** Ship Path E (larger minibatch) via the cleanest available route — either a `minibatch_size` kwarg added to the DSPy GEPA wrapper, or our own GEPA invocation bypassing the wrapper. Validate with a third spike (`gpt-5-nano` validator + larger minibatch) to confirm it lets at least one proposal through.
3. **Week 2-3:** Decide between Path D (Pareto acceptance) and Path C (stratified sampling) based on Path E's residual gap. Path D is cleaner; Path C catches a wider class of cases.
4. **Defer Path A/B** until acceptance is fixed. They're worth doing for cleaner selection semantics but only matter once candidates actually pass acceptance.

### What to update in PLAN.md deviation #8

Rewrite the deviation note. The current framing ("extend Pareto frontier into behavioral space") points at the wrong layer. The actual structural problem is **stochastic small-minibatch sum-strict acceptance discards per-instance signal**. Frontier shape is downstream; acceptance gate is the load-bearing wall.

### Untracked subprocess cost — a separate but related gap

The validator runs `hermes -z` subprocesses whose LM spend is invisible to the framework's cost tracker (`evolution/core/cost_advisor.py` and `lm_timing_callback`). This run shows $1.17 tracked but probably ~$5-15 total. For `--max-total-cost-usd` to be reliable when closed-loop is enabled, the subprocess cost needs to be captured. Worth filing as its own follow-up.
