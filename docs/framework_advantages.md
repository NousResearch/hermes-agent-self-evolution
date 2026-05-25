# Why this framework, not raw DSPy + GEPA?

## Who this is for

You've used DSPy and GEPA. You're looking at this fork and wondering whether the wrapping layer earns its keep — or whether you'd be better off calling `dspy.GEPA` directly. This document is the deltas, why they exist, and when they don't matter. No marketing copy: the goal is to give you enough information to decide.

## The regime mismatch

GEPA was validated on benchmarks with 111-300 validation examples per task (HotpotQA, IFBench, HoVer, PUPA — see [Agrawal et al. 2025][gepa]). At that scale, the rule "ship the candidate with the highest aggregate validation score" is statistically defensible — the standard error of the mean is small enough that a 2-3% gap between rank-1 and rank-2 is real signal.

SKILL.md evolution lives at N=20-60. The standard error of a mean validation score scales as roughly `1/√N`, so at N=30 a 2-3% gap is *inside* the sampling jitter. Argmax-on-aggregate becomes a coin flip: the candidate ranked #1 on val often isn't the one that generalizes best on a held-out evaluation. Recent work on LLM-as-judge noise ([Miller et al. 2025][miller-2025]; [Boyeau et al. 2026][boyeau-2026]) confirms that paired prediction noise at this scale routinely exceeds the differences being measured.

That's the regime this framework targets. Three layers on top of stock GEPA make the candidate that ships statistically defensible at small N rather than just argmax-of-noise.

## What this framework adds

### Knee-point ε-band selection

Stock GEPA returns the candidate with the highest aggregate validation score. This framework instead defines a **band** of "statistically indistinguishable" candidates — everything within ε of the top score — and walks that band in a deterministic order to pick a winner. The default ε is `1/n_val` (one validation example's worth of disagreement); the default order is highest-val-first with body size as tiebreak. A `smallest` strategy is also available for users who want explicit compression at iso-quality.

The ε-band idea is the modern descendant of the 1-SE rule (Hastie, Tibshirani & Friedman 2009 §7.10; Breiman et al. 1984). It's also what current knee-point Pareto work does in genetic-programming pipelines facing the same noisy / small-N problem ([Zhang et al. 2024][zhang-2024], EuroGP; [Zhang et al. 2025][zhang-2025], GPEM). The framework's twist is the static-validator inner loop: a candidate must pass static constraints (size limits, structure checks) to be picked, with a clean fallback to stock GEPA's choice if every band candidate fails.

File: `evolution/skills/knee_point.py`.

### Paired-bootstrap deploy gate

Stock GEPA returns a winner; this framework asks "is the winner provably not worse than baseline?" Before a candidate ships, it goes through a paired-bootstrap CI on per-example score deltas against the baseline skill on a held-out split. Three decision rules are selectable:

- `dual_check` — both the bootstrap mean and the lower CI bound must clear thresholds tied to skill-size growth. The default for runs that grow the skill body.
- `no_regression` — bootstrap mean must be ≥ 0.
- `non_inferiority` — Decagon-style; bootstrap lower CI bound must be > -tolerance. Ships variants that are statistically not-worse than baseline by more than the configured tolerance.

The non-inferiority option is the right call for compression-focused runs: at small N the bootstrap CI is too wide to detect tiny improvements, but a tight lower bound on "not worse" is exactly what a compression decision needs.

File: `evolution/core/constraints.py` — see `_check_growth_with_quality_gate` and `resolve_decision_rule`.

### BudgetAwareProposer + composite judge fitness

Two pieces of the GEPA loop are customized so the reflection step gets useful gradient.

`BudgetAwareProposer` is a custom GEPA instruction proposer that injects a character budget — with a configurable safety margin — into the reflection prompt. Stock GEPA reflects on traces without size constraints; this framework's reflection prompt knows the target.

The fitness function is a composite LLM-as-judge metric: separate scores for correctness, procedure-following, and conciseness, combined as `0.5·correctness + 0.3·procedure + 0.2·conciseness − length_penalty`. A binary metric tells GEPA "this failed"; a composite tells it "the answer was right but you wandered into 4 paragraphs of preamble." That's the gradient the reflective step actually consumes when proposing the next mutation.

Files: `evolution/skills/budget_aware_proposer.py`, `evolution/core/fitness.py`.

### Saturation pre-flight that refuses to spend budget on hopeless runs

GEPA will happily burn an hour optimizing a target that has no measurable headroom — every reflective mutation gets rejected because the minibatch ties at 100%, and you end up with the baseline byte-for-byte plus a bill. The framework's pre-flight (`evolution/core/saturation_check.py`) catches this BEFORE GEPA starts: scores the baseline on the holdout (and the closed-loop suite, if configured), classifies into `healthy` / `no_headroom` / `weak_signal` / `uniform_failure`, and either prompts the user (interactive) or default-denies with a `--force-saturation-check` override (non-interactive). Net cost is ~zero — the probe's holdout scores are reused at the post-GEPA evaluation site. When the run does proceed, the user has band-specific suggestions for the warn cases (try a stronger validator model, try a harder suite, increase iterations). Raw `dspy.GEPA` has no equivalent.

Files: `evolution/core/saturation_check.py`.

## Telemetry as a first-class feature

Every run writes `gate_decision.json` (schema_version `"5"`) capturing the deploy decision, the paired-bootstrap statistics, the static-constraint results, the knee-point band roster, and an explicit comparison against the candidate stock GEPA would have picked. Combined with `metrics.json` (deploy summary) and `run.log` (every LM call timing), this means a deploy decision is auditable post-hoc and the system can be re-calibrated on accumulated runs. Most upstream users won't realize they're missing this until they need to debug a bad ship.

## When raw GEPA is the right choice

Don't fork if all three of these are true:

- Your validation set is N≥300.
- Your metric is programmatic and high-signal — exact-match, unit-test pass, BLEU on a curated reference set.
- You don't ship variants behind a paired-comparison gate against a baseline.

In that regime, stock `dspy.GEPA` is the right tool and the wrapping layer adds complexity without buying you anything. The wrapping layer earns its keep specifically when (a) N is small, (b) the metric is noisy LLM-judged, and (c) the artifact ships into a long-running system where regressions are expensive to back out — which describes SKILL.md evolution exactly.

## Going deeper

- [`docs/research/knee_point_analysis.md`](research/knee_point_analysis.md) — full literature recon on knee-point selection and small-N model selection, walks the GEPA paper and DSPy source for the original argmax behavior, pros/cons table comparing stock vs. the framework's selector, and the verdict for the modification.
- [`AGENTS.md`](../AGENTS.md) — repo-level component map and conventions.
- [`docs/architecture.md`](architecture.md) — statistical substrate and design-decision log.

If you're convinced and want to run something, jump back to [`README.md`](../README.md#quick-start) for the Quick Start.

## References

- [Agrawal et al. 2025][gepa]. *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning.* arXiv:2507.19457. ICLR 2026 Oral.
- Hastie, Tibshirani & Friedman 2009. *The Elements of Statistical Learning*, 2nd ed. — 1-SE rule discussion in §7.10.
- Breiman, Friedman, Olshen & Stone 1984. *Classification and Regression Trees.* — original 1-SE rule.
- [Zhang, Chen, Xue, Banzhaf, Zhang 2024][zhang-2024]. *Improving Generalization of Evolutionary Feature Construction with Minimal Complexity Knee Points in Regression.* EuroGP 2024.
- [Zhang, Chen, Xue, et al. 2025][zhang-2025]. *Adaptive Complexity Knee Point Selection in Multi-objective Genetic Programming for Improving Generalization.* Genetic Programming and Evolvable Machines.
- [Miller et al. 2025][miller-2025]. *Measuring all the noises of LLM Evals.* arXiv:2512.21326.
- [Boyeau et al. 2026][boyeau-2026]. *Noisy but Valid: Robust Statistical Evaluation of LLMs with Imperfect Judges.* arXiv:2601.20913.

For a deeper literature recon — including the broader knee-point Pareto and small-N model-selection work that the framework's selector sits inside — see [`docs/research/knee_point_analysis.md`](research/knee_point_analysis.md).

[gepa]: https://arxiv.org/abs/2507.19457
[zhang-2024]: https://link.springer.com/chapter/10.1007/978-3-031-56957-9_9
[zhang-2025]: https://link.springer.com/article/10.1007/s10710-025-09525-6
[miller-2025]: https://arxiv.org/abs/2512.21326
[boyeau-2026]: https://arxiv.org/abs/2601.20913
