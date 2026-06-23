# The hard axis: capable Claude saturates standard debugging — no reasoning-methodology headroom (NULL)

**Verdict: NULL-saturated.** On standard algorithmic-debugging tasks, both **Claude
Opus and Claude Sonnet** solve every task perfectly *unaided* — they root-cause **and
generalize to held-out edge cases** with no skill at all. There is no headroom for a
methodology skill to move, so reasoning-quality coupling cannot even be measured on this
substrate. This extends the campaign's saturation finding from gpt-5-mini to the **top
Claude models**, on a rig built specifically to be leak-clean and floor-clean.

This is the **reasoning/quality axis** — distinct from the convention axis. We separately
showed (G1/G2) that a thick skill's body *does* couple to a capable agent's adherence to
a **non-inferable convention** (0→1, GREEN). The asymmetry is the campaign's through-line:
artifact text drives a capable agent exactly where it carries a signal the agent **lacks**
(an arbitrary convention) — and is inert where the agent **already has** the signal
(good reasoning process).

## What was tested

The open question: does a skill's *methodology* content (HOW to reason — root-causing,
edge-case enumeration, verification; NOT conventions, NOT answers) improve a **capable**
agent's task **outcomes**? Scope: the COUPLING question (does a hand-written GOLD
methodology skill beat a WEAK one), not evolvability.

Rig (designed against the three walls that sank prior reasoning-quality measurements;
spike: `spikes/reasoning_quality_coupling/`, local):
- **Substrate** — 8 hard-debugging tasks in Python spanning a difficulty range
  (touching-interval merge, leftmost binary search, exact-fit word wrap, RPN operand
  order, numerically-unstable variance, point-touch interval intersection, unreachable
  min-coins, order-preserving dedup).
- **Hidden, leak-proof, zero-LM oracle** — the agent gets the buggy code + a *symptom*
  (one failing example), never the grading test. The held-out grader is materialized only
  **after the agent exits**, at a random temp path, with inputs **different** from the
  symptom — so passing requires fixing the root cause, not transcribing (the
  code-evolution leakage trap, neutralized). No LM judge anywhere.
- **Floor-clean** — `compile_suite_floor` is **inert** (len 0) on this suite, so a skill
  gain here would be reasoning, not a restated convention ("suite-states-the-win" wall,
  neutralized).
- **Arms** — GOLD methodology skill (strong root-cause discipline) vs WEAK (vague "try
  edits"), frontmatter byte-identical so selection is unconfounded. Neutral prompt — the
  skill body, not the prompt, carries the generalization discipline.
- **Gate A (per model)** — the no-skill baseline must land in the 0.20–0.70 *headroom*
  band for ≥4 tasks, else there is nothing for a skill to move (cheap kill before any
  contrast spend).

## Result

| model  | tasks solved (no skill) | per-task rate | headroom band (0.20–0.70) |
|--------|-------------------------|---------------|----------------------------|
| Opus   | 8/8 (40/40 runs)        | 1.00 each     | **0/8** |
| Sonnet | 8/8 (40/40 runs)        | 1.00 each     | **0/8** |

Total spend: **$7.24** (Opus $4.55 + Sonnet $2.69). The Gate-A kill-shot did its job —
the line was falsified for the price of a sandwich, before any GOLD-vs-WEAK contrast.

Both models passed every **held-out** edge case unaided — they did **not** overfit the
symptom example. So the failure mode a methodology skill would fix (overfitting / skipping
edge cases) simply did not occur.

## Why the hard axis is hard (the structural reason)

A methodology skill can only help where the agent **can** solve a task but **fails on
process** — rushing, not enumerating edge cases, not verifying. Capable agents have
**internalized** that process. So:
- The tasks where methodology would help are exactly the ones the agent already aces →
  saturation, no headroom.
- The tasks the agent fails need **capability/knowledge**, which a "be systematic" skill
  cannot supply.

The "process gap" methodology fills is closed in a capable agent. This is the
reasoning-axis form of the campaign's decoupling thesis, and it is *why* the convention
axis couples but the quality axis does not: a convention is non-inferable (the agent
lacks it); good methodology is inferable/internalized (the agent has it).

## Honest scope (what this does and does not establish)

CAN claim: on **standard algorithmic-debugging** tasks, capable Claude (Opus + Sonnet) is
saturated — no reasoning-methodology headroom; a methodology skill has nothing to add.
The rig is leak-clean, floor-clean, and the kill was made by an explicit headroom gate.

CANNOT claim: that methodology *never* helps any capable agent on any task. A harder or
different task class **might** have a headroom band. But the path is fraught: tasks hard
enough to fail Opus most likely fail for **knowledge/insight** reasons (which methodology
cannot fix), so a null there would be uninterpretable (the **process-vs-knowledge
confound**). Finding a clean "process-gap" task that is hard for Opus *and* fixable by
methodology is a narrow, high-effort target — and the structural reason above predicts it
is rare.

## Reusable rig

The spike is reusable for any future harder-substrate attempt: leak-proof hidden
held-out grader, floor-inertness check, per-model Gate-A headroom gate, a mandatory
positive canary (GOLD-vs-baseline sensitivity), and a `validation_runs.jsonl` audit trail.
What would flip the verdict: a substrate where capable Claude lands in the 0.20–0.70 band
*and* the failure mode is process, not knowledge — at which point the canary tests whether
GOLD lifts it. Not found on standard debugging.

## Bottom line

The project's promise holds on the **convention** axis (skills install non-inferable
conventions a capable agent can't guess — G1/G2 GREEN) and is **saturated** on the
**reasoning-methodology** axis for capable Claude (this result). Evolving a skill's
*methodology* to make a capable agent reason *better* has no measurable headroom on
standard tasks, because the agent already reasons well. The honest product surface for
skill evolution is convention/instruction installation, not reasoning improvement.
