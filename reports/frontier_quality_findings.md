# Open-ended review quality resists non-saturating measurement

The three-region map leaves one regime open: **region 3 — open-ended generative quality.** Where a task is pass/fail and the agent is already competent, evolving a skill has nothing to take hold of (the binary cliff); where the text supplies a non-inferable signal, evolution + the gate deliver. Region 3 — does a better skill or prompt yield *better open-ended output* — stayed unresolved because its only metric, a holistic LLM judge, saturates and cannot see the gradient even if one exists.

This probe tested region 3 concretely: **does an evolved code-review skill make a capable agent (Sonnet) propose sounder fixes, measured by signals that do not saturate?**

## Design
The probe escaped the judge with checkable axes, the primary one grounded in execution:

- **recall** — does the review identify a planted issue (a real upstream fix-commit's defect)?
- **fix-soundness** (the region-3 claim) — apply the review's proposed fix and run the bug's hidden test; sound iff it passes. A zero-LM execution oracle.
- **precision** — a guard against flagging everything.

A trust gate had to pass before any evolution spend: the detection checker's false-positive rate ≤ 0.10, a bidirectional metric-discriminates control, independent-labeler reliability, and a baseline sitting inside a 0.30–0.95 headroom band — on a frozen, file-disjoint train/holdout split of real fix-commits.

## The instrument
A negative result is only as trustworthy as the metric, so the harness was adversarially reviewed and hardened against every way the result could be an artifact. The review fixture is history-sterile (the buggy file alone, no git history) and the reviewer has read-only tools, so it cannot reach the upstream fix and copy the answer. Fix-extraction splices into the whole file and fails closed. Detection is a strict LM judge **validated by the false-positive calibration** — measured at **0/8**, so it does not over-fire. Soundness is decoupled from recall (scored per caught issue, not recall-weighted). The reviews that were scored are real and substantial (4.8–25 KB).

## Result
With genuine reviews and a calibrated checker, the agent identified the specific planted issue in **0 of 8 cases**. This is not poor reviewing — the reviews are strong. On a case whose upstream fix added missing `sudo` privilege-flag detection, the agent's review flagged a prompt-injection vector, an unsynchronized lock read, and a resource leak — all real defects, **none of them the planted one**.

## Why, and the boundary it draws
Open-ended review surfaces the agent's *own* salient defects, not a pre-chosen target. This is the nature of the task, not a fixable flaw in the metric, and it forces a general conclusion:

> The execution oracle — the only non-saturating quality metric available — can only score a fix to a *specific* bug that has a test. Getting the agent to address that specific bug requires **targeting** it (handing it the symptom), which is the *repair* framing. Open-ended generative quality, by definition, is not targeted, so it offers no pre-chosen oracle.

So region 3 resists **both** instruments we have: the execution oracle needs a targeted task (recall on open-ended review is ~0), and the LLM judge saturates (the original obstacle). The cliff is not only binary-correctness saturation — the one metric immune to judge-saturation is available only for targeted work, and open-ended quality is intrinsically un-targeted.

## Scope
One substrate, one repository, whose fix-commits were predominantly security hardening — adding edge-case coverage rather than correcting wrong-line behavior — which a reviewer would not flag as defects in the pre-fix code, compounding the recall floor. A repository of genuine wrong-behavior bugs might lift baseline recall, but the deeper mismatch (open-ended review does not target a pre-chosen issue) would likely recur.

This is **not** a claim that region 3 is empty. It is a claim, with evidence, that *execution-checked quality of open-ended output against planted issues* cannot measure it. A measurable variant does exist — target the agent at the bug so recall is ~1 by construction, then measure fix-soundness — but that is the repair framing, where methodology skills already appear saturated on capable agents. Region 3, as open-ended quality, remains bounded on both sides: un-targeted output has no execution oracle, and the judge that could grade it saturates.
