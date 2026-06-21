# Tool-description coupling on non-inferable tools — and the circular-eval limit

## Abstract

The campaign's central result is that on a capable agent, artifact *text* (skills,
tool descriptions, prompt sections) is **decoupled** from behavior in both
directions. But every tool-description test ran on **name-inferable built-ins**
(`write_file`/`patch`): the agent routes from the familiar name and ignores the
description, so the null measured *name-inference*, not *descriptions*. This probe
closes the one untested surface — tools whose **names do not telegraph behavior**,
the property real MCP tools have, where the description is the *sole* routing signal.

Two results, both decisive:

1. **Coupling is real (GREEN).** On novel, non-inferable tools the description
   *fully drives* tool selection: a clear description routes correctly (~1.0); a
   misdirecting (lying) one flips the agent to the wrong tool **100%** of the time.
   This is the first artifact→behavior coupling found, and the qualifier the
   decoupling thesis was missing: *artifact text is decoupled only when the agent
   can infer the tool from its name.*
2. **The pipeline can exploit it only with an artifact-independent eval.** The
   standard `evolve_tool` synthetic eval is **generated from the description under
   test**, so a broken baseline contaminates it: the optimizer polishes the *lie*
   and the gate deploys a behaviorally-*worse* description. A hand-authored
   behavioral eval encoding true intent produces the correct repair. This is the
   evaluation-side twin of "a self-improving loop can't author its own oracle."

This is a **proxy**: it isolates the description→routing *mechanism* with novel
in-process Hermes tools. It does not test the MCP transport (stdio/SSE,
`mcp__server__tool` namespacing, server-provided descriptions). A coupling result
says "build the transport and re-test"; the eval-contamination result transfers
directly (it is a property of synthetic-from-artifact generation, not of MCP).

## Method

Three confusable pairs of **novel** tools, registered as real Hermes tools so a
live `hermes -z` agent can route to them. Generic-verb names carry no signal about
the contract axis (so the description, not the name, must disambiguate); each pair
shares an identical parameter schema (closing the argument-shape channel):

- **scope:** `vellum_set` (one named record) vs `vellum_apply` (all records)
- **durability:** `marl_save` (provisional/reversible) vs `marl_record` (permanent)
- **read:** `korrel_get` (current value) vs `korrel_read` (full history)

Three description variants per tool: **CLEAR** (states the contract; never names the
sibling), **AMBIGUOUS** (disambiguator stripped — identical across the pair, the
leakage detector), **MISDIRECTING** (the sibling's contract — a lie). Tasks are
hand-authored, goal-worded (never naming the tool), scored by the existing zero-LM
**membership verdict** (expected vs forbidden tool — no judge, so no judge
saturation). Pairs are run in **isolation** (only one pair's two tools exposed at a
time) so the choice is the strictly-within-pair binary decision; A/A noise floor and
per-task discrimination labeling reused from the validation harness. Agent:
gpt-5-mini. Reps=8.

## Result 1 — coupling is real (Stages 1–2)

**Discrimination (CLEAR vs AMBIGUOUS), 15 discriminator tasks:** 10/15
DISCRIMINATIVE, 0 unfillable, sensitivity canary discriminative. When both tools in
a pair carry identical (AMBIGUOUS) descriptions the agent guesses; a CLEAR
description routes it correctly:

| pair | AMBIGUOUS | CLEAR |
|---|---|---|
| scope | ~0.60–0.80 | 1.00 |
| durability | 0.60 | 1.00 |
| read | ~0.25–0.62 | 0.88–1.00 |

**Effect + direction (vs the A/A floor):** passive (CLEAR→AMBIGUOUS drop) moved
11/15; **active (CLEAR→MISDIRECTING flip) moved 15/15 = 100%** — every discriminator
went to 0.00, and a spot-check confirmed the agent *actively calls the forbidden
sibling* (following the lie), not erroring out. The 5 TOO_EASY tasks are the design
working: they align with a name prior (the agent picks the right tool without the
description), so they are excluded from the coupling estimate — the description earns
its keep precisely where the name misleads.

Contrast with the decoupling null: misdirecting a **name-inferable** built-in
(`write_file`) moved 1/7, within noise. Misdirecting a **non-inferable** tool moves
100%. Same rig, opposite result — the difference is name-inferability.

## Result 2 — repairing the description: the eval is the determining factor (Stage 3)

We then ran the real `evolve_tool` pipeline to *repair* a deliberately broken
`vellum_set` description (sibling `vellum_apply` fixed CLEAR).

- **Synthetic-from-artifact eval → anti-fix.** With a MISDIRECTING baseline, the
  synthetic dataset generator (which reads the description to make tasks) inherited
  the lie: it produced *all-record* tasks where the lying tool is "correct." GEPA
  optimized toward a *clearer version of the lie*, the synthetic holdout "improved"
  +37%, and the gate **deployed** it. Post-hoc on the live agent, the deployed
  "repair" routed single-record tasks **0.54 — worse than the 0.71 broken baseline**
  it started from (a true fix scores 1.00). The synthetic gate was not just blind;
  it was anti-correlated with live behavior, because its eval was generated from the
  artifact it was judging.
- **Artifact-independent eval → correct fix.** Feeding the same reflective proposer
  the *behavioral* failures from the hand-authored suite (which encodes true intent —
  single-record → `vellum_set` — regardless of the description) produced a correct
  single-record description; live routing went **8/10 → 10/10** (all proposals
  converged). Same proposer, two evals, opposite outcomes — the eval is the
  determining factor.

A mechanistic corollary surfaced along the way: in a *mixed* setup (one tool broken,
the sibling clear), an **ambiguous** target description is **not** behaviorally
broken — the clear sibling disambiguates by elimination ("the other tool says
all-records, so this must be the single-record one"). Only an *active* lie misroutes.
The Stages 1–2 ambiguity bit only because *both* tools were ambiguous (isolation).

## Honest scope and caveats

- **Proxy, not transport.** Tests the coupling mechanism via novel in-process Hermes
  tools; the MCP transport itself is untested. A coupling GREEN warrants building it.
- **One model** (gpt-5-mini); coupling strength and the noise floor are model-specific.
- **Selection only.** The verdict is which tool was chosen, not whether it was used
  correctly.
- **Gate conservatism is real.** In the artifact-independent repair, the noise-aware
  deploy gate *rejected* the correct fix ("gain +2 < required 3"): the misdirecting
  baseline only *partially* breaks routing, so the suite-level gain sits inside the
  A/A noise floor. The gate correctly refuses to certify a within-noise gain; a clean
  deploy would need a more-uniformly-broken baseline or multi-rep noise estimation,
  not a re-tuned measurement.
- **The CL-primary (live-behavioral) gate is hard to trigger deliberately,** because
  the synthetic judge and the live agent are correlated (both read the description):
  a description that breaks one tends to break both, landing the `healthy` band
  rather than the `weak_signal` band the live gate needs.

## What a real MCP-description evolver would need

The coupling is real and exploitable, but the optimizer must be gated on a
**live-behavioral, artifact-independent oracle** (a hand-authored suite encoding true
intent — exactly the Stages 1–2 machinery), **not** a synthetic eval generated from
the description, which inherits the artifact's errors and can deploy an anti-fix. It
also needs enough reps to lift a real repair above the suite's noise floor. The
durable contribution is the boundary: *self-referential evaluation cannot certify a
contract it was generated from* — the eval-side analogue of the self-improvement
verification wall.
