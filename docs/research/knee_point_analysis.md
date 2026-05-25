> **Superseded.** Historical analysis from 2026-05-03. The knee-point ε-band selector was empirically dropped as a no-op for the val-best path in May 2026 (10/10 mode agreement across 5 ε modes on a regenerated calibration corpus). The selector survives only for the `--knee-point-strategy smallest` opt-in path. See [`reports/calibration_findings.md`](../../reports/calibration_findings.md) for current status.

# Knee-point Pareto Selection: Analysis

This document evaluates the framework's custom Pareto selector at
`evolution/skills/knee_point.py` against (a) what the GEPA paper and DSPy
implementation actually do, and (b) the modern multi-objective and
cross-validation model-selection literature.

---

## 1. High-level technical overview

### What GEPA's default does

GEPA (Agrawal et al., 2025, arXiv:2507.19457; ICLR 2026 Oral) runs a
reflective evolutionary loop that maintains a population of candidate prompts
and a per-instance Pareto frontier over a held-out validation set
(`D_pareto`). The frontier is used **during** evolution to decide which
parent to mutate next — `candidate_selection_strategy="pareto"` samples from
candidates that win on at least one validation instance.

At termination, the algorithm collapses the Pareto front to a single output.
Algorithm 1, line 23 of the paper specifies: *"return Φ\* maximizing average
score on D_pareto."* The DSPy implementation
([`stanfordnlp/dspy:dspy/teleprompt/gepa/gepa.py`](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/gepa/gepa.py))
realises this as:

```python
@property
def best_idx(self) -> int:
    scores = self.val_aggregate_scores
    return max(range(len(scores)), key=lambda i: scores[i])
```

So the *exploration* uses the full Pareto front, but the *output* is a plain
argmax over aggregate validation score. There is no built-in alternative
final-selection knob; `candidate_selection_strategy` only governs in-loop
mutation.

### What the framework's selector does

`select_knee_point` replaces the final argmax with an ε-band sweep:

1. ε defaults to `1 / n_val` — "one example's worth of disagreement," sized
   to the resolution of the validation set rather than pretending a 20-item
   valset can distinguish 0.02 differences.
2. The band is `{i : val_score_i ≥ best - ε}`.
3. `strategy="val-best"` (default) walks the band by `(-val_score, body_chars,
   idx)` — highest val first, parsimony as tiebreak. `strategy="smallest"`
   inverts the priority for explicit compression-chasing.
4. A static validator gates each candidate; the first to pass is returned.
5. If the entire band fails static checks, the function falls back to GEPA's
   `best_idx`.
6. Telemetry (band size, picked rank, ε, GEPA's pick for comparison) is
   captured for post-hoc ε calibration.

### Intuition

On a 20-50-example valset, the standard error of an aggregate accuracy near
0.5 is roughly `sqrt(0.25/N) ≈ 0.07–0.11`. The gap between rank-1 and rank-3
candidates is routinely smaller than that. Picking strict argmax in this
regime is choosing on noise. ε = 1/n_val is a coarse but defensible
"resolution floor" — it absorbs differences of one example or fewer, and the
deterministic tiebreak (highest val, then smallest body) picks the candidate
most likely to generalise inside the noise band. The fallback strategy
`smallest` lets a user trade val cost for parsimony when that's the goal.

---

## 2. GEPA / DSPy source recon

**GEPA paper** (arXiv:2507.19457, [HTML](https://arxiv.org/html/2507.19457v1)):

- The paper uses `D_pareto` sizes of **300** (HotpotQA, IFBench, HoVer) and
  **111** (PUPA) — meaningfully larger than the n_val ≈ 20-50 the framework
  runs.
- The paper acknowledges: *"the majority of GEPA's rollouts are expended for
  candidate validation, which can be performed well even on smaller Pareto
  datasets. Future works should explore the impact of the hyperparameter
  Pareto-validation set size."* This is an open question the authors flag,
  not a defended choice.
- Section 5, Observation 2: *"reflectively evolved instructions now
  demonstrate a lower generalization gap, underscoring both advancements in
  model capabilities and the benefits of GEPA's design."* This is a claim
  about the **prompts being more generalisable**, not about the final
  selection rule preventing val-overfit.
- Algorithm 1 line 23: argmax aggregate validation. No defence of the rule
  itself; the Pareto framing is for in-loop diversity, not the final pick.

**DSPy implementation**
([`gepa.py`](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/gepa/gepa.py)):

- `DspyGEPAResult.best_idx` and `best_candidate` are pure argmax; no
  alternative exposed.
- `candidate_selection_strategy: Literal["pareto", "current_best"] = "pareto"`
  controls which parent to mutate next — this is the in-loop knob, not a
  final selector.
- DSPy's
  [GEPA docs](https://dspy.ai/api/optimizers/GEPA/overview/) recommend a
  valset minimum tied to `auto_budget()`'s default minibatch (**35**), and
  advise *"Provide the smallest valset that is just large enough to match
  your downstream task distribution."* They give no advice for handling small-N
  selection noise.

**GEPA reference repo** ([`gepa-ai/gepa`](https://github.com/gepa-ai/gepa))
exposes `ParetoCandidateSelector`, `CurrentBestCandidateSelector`, and
`EpsilonGreedyCandidateSelector` — all in-loop selectors. No final-selection
alternative.

**Issues / PRs / community discussion**: no open or closed DSPy issue
raises val/test gap, small-N val overfitting, or alternative final-selection
rules for GEPA. The only related issue is
[#8782](https://github.com/stanfordnlp/dspy/issues/8782) ("Update docs for
GEPA train/val split advise"), still open and empty. The framework's
modification is therefore not contradicting an established recommendation —
it's filling a gap the GEPA authors have flagged but not addressed.

---

## 3. Knee-point and small-N selection literature

Both founding lines of work — Branke et al. on knee-point Pareto selection
and the Breiman / Hastie-Tibshirani 1-SE rule for cross-validation model
selection — remain active in the current literature. The framework's
heuristic sits inside the modern consensus.

### Knee-point Pareto selection

Branke et al. 2004 ([PPSN VIII chapter](https://link.springer.com/chapter/10.1007/978-3-540-30217-9_73))
has **568 citations** on Semantic Scholar and continues to be cited in
current MOEA work:

- **Heydari & Branke 2022, "Finding Knees in Bayesian Multi-objective
  Optimization"** ([PPSN XVII chapter](https://link.springer.com/chapter/10.1007/978-3-031-14714-2_8))
  — direct extension to BO, where evaluations are expensive and exactly the
  noisy/small-budget regime this framework operates in.
- **Yu et al. 2019** "Benchmark Problems and Performance Indicators for
  Search of Knee Points" ([PubMed 30762578](https://pubmed.ncbi.nlm.nih.gov/30762578/)).
- **He et al. 2024**, "A Knee Point-Driven Many-Objective Evolutionary
  Algorithm" ([Wiley 2024/4737604](https://onlinelibrary.wiley.com/doi/10.1155/2024/4737604)).
- **Zhang, Chen, Xue, Banzhaf, Zhang 2024**, "Improving Generalization of
  Evolutionary Feature Construction with **Minimal Complexity Knee
  Points** in Regression" (EuroGP 2024, [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-031-56957-9_9)).
- **Zhang et al. 2025**, "Adaptive Complexity Knee Point selection in
  multi-objective genetic programming for improving generalization"
  (Genetic Programming and Evolvable Machines, [Springer
  article](https://link.springer.com/article/10.1007/s10710-025-09525-6)).

The 2024 EuroGP and 2025 GPEM papers are the most directly relevant: they
explicitly choose between knee candidates by *minimum complexity* to
**improve generalisation** in **noisy and sample-limited** regimes.
That is essentially the framework's `strategy="smallest"` plus the
parsimony tiebreak in `strategy="val-best"`. The 2025 paper claims to
significantly outperform nine established model-selection strategies
*"particularly when dealing with sample-limited and noisy datasets"* —
which is exactly the framework's regime.

### 1-SE / ε-band model selection

Hastie/Tibshirani's 1-SE rule remains the standard cross-validation
heuristic and is implemented in mainstream tooling (R `mgcv::one.se.rule`,
glmnet, mlr). Modern critical assessment:

- **Chen & Yang 2021**, "The One Standard Error Rule for Model Selection:
  Does It Work?" (Stats 4(4):51, [MDPI](https://www.mdpi.com/2571-905X/4/4/51)).
  Findings: 1-SE *helps* in sparse variable selection / structure recovery,
  *can hurt* in pure prediction (especially when the candidate model is
  well-specified). Standard error estimates have 50-100% bias in some
  regimes. Verdict: useful but not a free lunch — the ε is a
  regularisation choice and should be context-tuned.
- The framework's default ε = 1/n_val is *not* the textbook 1-SE; it's a
  resolution-based ε, more conservative on small N (1/20 = 0.05; one SE on
  N=20 binomial is ~0.11). So the framework picks a tighter band than
  classical 1-SE, which mitigates the "underfitting / underperforming"
  risk Chen & Yang flag.

### Validation noise in modern LLM evaluation

- **Miller et al. 2025**, "Measuring all the noises of LLM Evals"
  ([arXiv:2512.21326](https://arxiv.org/abs/2512.21326)) — paired
  prediction noise typically *exceeds* paired data noise on benchmarks
  like MATH500. Implication: aggregate val score is a noisy ranking
  signal, especially below N≈100.
- **Boyeau et al. 2026**, "Noisy but Valid: Robust Statistical Evaluation
  of LLMs with Imperfect Judges"
  ([arXiv:2601.20913](https://arxiv.org/html/2601.20913v1)) —
  variance-corrected critical thresholds for model comparison, motivated
  by exactly the "best on val ≠ best in deployment" concern.
- **Lakera blog, "Your validation set won't tell you if a model
  generalizes"** ([link](https://www.lakera.ai/blog/your-validation-set-wont-tell-you-if-a-model-generalizes))
  — practitioner-level confirmation that argmax-on-val is fragile.
- **Sukhbaatar et al. 2026**, "Don't stop me now: Rethinking Validation
  Criteria for Model Parameter Selection"
  ([arXiv:2602.22107](https://arxiv.org/abs/2602.22107)) — argues
  validation-accuracy-based selection is the worst rule among common
  options, and post-hoc selection across all checkpoints generally
  outperforms the rule-based pick. This *partially undermines* both raw
  GEPA and the framework's modification, but its concrete
  recommendation (loss-based criteria + post-hoc evaluation) is more
  applicable to gradient training than to discrete prompt populations
  where loss isn't available.

### Synthesis

The framework is sitting on a stack of *current* literature:
- knee-point selection on Pareto fronts (Branke 2004 → Heydari & Branke
  2022 → Zhang et al. 2024-2025);
- ε-band / "good-enough" selection from CV (Hastie/Tibshirani → Chen &
  Yang 2021);
- documented LLM-eval noise on small N (Miller et al. 2025; Boyeau et al.
  2026).

None of this *requires* the modification, but the modification is on the
inside of the scientific consensus, not in opposition to it.

---

## 4. Pros / cons table — raw GEPA vs knee-point

| Dimension | Raw GEPA (`best_idx`) | This framework's knee-point (`select_knee_point`) |
|---|---|---|
| Statistical robustness on small N (20-50) | Weak — argmax over noisy aggregates; rank-1 vs rank-3 often inside one SE | Strong — ε = 1/n_val absorbs sub-resolution differences; deterministic tiebreak |
| Risk of overfitting to valset | High in small-N — argmax selects on noise | Lower — band collapses noise; parsimony tiebreak biases toward simpler bodies |
| Risk of underfitting / missing real winners | Low — always picks the visible peak | Low for `val-best` (highest val in band first); meaningful for `smallest` (deliberate compression trade) |
| Determinism / reproducibility | Deterministic (max + first-tie semantics) | Deterministic (sort key includes idx) |
| Computational cost | Free (one argmax) | Negligible — band scan + at most `band_size` static-validator calls |
| Telemetry / debuggability | None — no insight into runner-up gap | Strong — full band roster, picked rank, ε, GEPA-default-comparison fields all logged |
| Alignment with GEPA paper framing | Exactly matches Algorithm 1 line 23 | Diverges from the paper's literal output rule, but consistent with the paper's own concern that small `D_pareto` sizes need study |
| Alignment with broader literature | Pure argmax-on-val is the most-criticised rule in CV / model-selection literature | Endorsed by 1-SE family (Hastie/Tibshirani, Chen & Yang 2021) and by recent knee-point work (Zhang et al. 2024, 2025) |
| Behaviour at large N (e.g. N=200) | Argmax becomes legitimate — SE shrinks below typical candidate gaps | Modification quietly degrades to a near-no-op: ε = 1/200 = 0.005 makes most bands singleton, telemetry still useful, val-best still picks the GEPA default in most cases |
| Static-validation safety | None — could deploy a candidate that violates static constraints | Built-in — falls back to GEPA default only after every band candidate fails static |

---

## 5. Open empirical questions

Things worth measuring against deployments:
1. Per-run, what is the rank-of-picked-candidate-by-val-score? If it's
   consistently 1 (i.e. knee-point and raw GEPA agree), the modification is
   load-bearing only via static-validator gating.
2. Holdout-score distribution: holdout(picked) − holdout(GEPA-default).
   Mean and per-run sign. Negative means knee-point regresses.
3. Body-size distribution: chars(picked) / chars(GEPA-default). Quantifies
   the parsimony bias.
4. Fallback-fired rate: how often does every band candidate fail static
   validation? If high, the static-validator side of the system is doing
   more work than the knee-point side.
5. Sensitivity to ε: for the same runs, would ε ∈ {0.5/n_val, 1/n_val,
   2/n_val, 1-SE binomial} change picks? Cheap to compute post-hoc from the
   recorded `band_roster`.
6. Strategy comparison: for each run, what does `val-best` vs `smallest`
   pick, and how do their holdout scores compare?
7. Scaling: does the modification still beat raw GEPA at n_val = 100, 200?
   If parity by N=100, the modification is small-N-only and the
   recommendation could be to auto-disable above some threshold.

---

## References

- Agrawal et al. 2025. *GEPA: Reflective Prompt Evolution Can Outperform
  Reinforcement Learning.* arXiv:2507.19457. [paper](https://arxiv.org/abs/2507.19457),
  [HTML](https://arxiv.org/html/2507.19457v1),
  [OpenReview](https://openreview.net/forum?id=RQm2KQTM5r) (ICLR 2026 Oral).
- DSPy GEPA implementation:
  [`stanfordnlp/dspy:dspy/teleprompt/gepa/gepa.py`](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/gepa/gepa.py).
- DSPy GEPA docs: <https://dspy.ai/api/optimizers/GEPA/overview/>.
- GEPA reference repo: <https://github.com/gepa-ai/gepa>.
- Branke, Deb, Dierolf, Osswald 2004. *Finding Knees in Multi-objective
  Optimization.* PPSN VIII. [Springer](https://link.springer.com/chapter/10.1007/978-3-540-30217-9_73).
- Heydari & Branke 2022. *Finding Knees in Bayesian Multi-objective
  Optimization.* PPSN XVII. [Springer](https://link.springer.com/chapter/10.1007/978-3-031-14714-2_8).
- Zhang, Chen, Xue, Banzhaf, Zhang 2024. *Improving Generalization of
  Evolutionary Feature Construction with Minimal Complexity Knee Points
  in Regression.* EuroGP 2024.
  [Springer](https://link.springer.com/chapter/10.1007/978-3-031-56957-9_9).
- Zhang, Chen, Xue, et al. 2025. *Adaptive Complexity Knee Point Selection
  in Multi-objective Genetic Programming for Improving Generalization.*
  Genetic Programming and Evolvable Machines.
  [Springer](https://link.springer.com/article/10.1007/s10710-025-09525-6).
- He et al. 2024. *A Knee Point-Driven Many-Objective Evolutionary
  Algorithm with Adaptive Switching Mechanism.* Journal of Applied
  Mathematics. [Wiley](https://onlinelibrary.wiley.com/doi/10.1155/2024/4737604).
- Yu, Jin, Olhofer 2019. *Benchmark Problems and Performance Indicators for
  Search of Knee Points in Multiobjective Optimization.* IEEE
  T-Cybernetics. [PubMed](https://pubmed.ncbi.nlm.nih.gov/30762578/).
- Hastie, Tibshirani, Friedman 2009. *The Elements of Statistical
  Learning*, 2nd ed. — 1-SE rule discussion in §7.10.
- Breiman, Friedman, Olshen, Stone 1984. *Classification and Regression
  Trees.* — original 1-SE rule.
- Chen & Yang 2021. *The One Standard Error Rule for Model Selection:
  Does It Work?* Stats 4(4):51. [MDPI](https://www.mdpi.com/2571-905X/4/4/51).
- Miller et al. 2025. *Measuring all the noises of LLM Evals.*
  arXiv:2512.21326. <https://arxiv.org/abs/2512.21326>.
- Boyeau et al. 2026. *Noisy but Valid: Robust Statistical Evaluation of
  LLMs with Imperfect Judges.* arXiv:2601.20913.
  <https://arxiv.org/abs/2601.20913>.
- Sukhbaatar et al. 2026. *Don't stop me now: Rethinking Validation
  Criteria for Model Parameter Selection.* arXiv:2602.22107.
  <https://arxiv.org/abs/2602.22107>.
- DSPy GEPA train/val split docs issue:
  [stanfordnlp/dspy#8782](https://github.com/stanfordnlp/dspy/issues/8782).
