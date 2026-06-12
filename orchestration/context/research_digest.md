# Research digest — verified frontier findings

Adversarially verified (3-vote per claim) June 2026 sweep of the 2025–2026
literature on LLM agent self-evolution. Confidence labels survive
verification; caveats are part of the finding. Cite by [G1]-style tag.

## Core lineage

- **[G1] GEPA peer-validated (ICLR 2026 Oral, arXiv:2507.19457).** Reflective
  prompt evolution beats GRPO RL by ~6% avg at up to 35× fewer rollouts;
  beats MIPROv2 by >10%. Caveat: Decagon deployment blog reports GEPA can
  overfit with many iterations. HIGH confidence.
- **[G2] gskill (gepa-ai, Feb 2026)** — the closest published analogue to this
  framework: GEPA `optimize_anything` + SWE-smith synthetic verifiable-task
  generation evolving `.claude/skills/` SKILL.md for coding agents. 300-rollout
  budget: Mini-SWE-Agent (gpt-5-mini) resolve 55%→82% (Jinja), 24%→93%
  (Bleve). Tasks are synthetic-but-verifiable (planted, test-checked), ~300/repo.
  HIGH confidence; self-reported, simple synthetic tasks, repo-specific skills.
- **[G3] Weak-to-strong skill transfer (gskill).** Skills evolved with a weak
  model on a simple scaffold transfer to Claude Code: Bleve Haiku 4.5
  79.3%→98.3% (and faster: 173s→142s); Sonnet 4.5 94.8%→100%. Caveat:
  Sonnet delta ≈ run noise; Jinja Sonnet dipped slightly; 2 repos only.
  HIGH confidence on the Haiku-scale effect. **Evolve cheap → validate
  transfer → deploy expensive.**

## Selection under small-N noisy fitness

- **[S1] RoboPhD (arXiv:2604.04347, preprint, MEDIUM).** Replaces train/val
  split with Elo tournaments on training data (evaluation = selection).
  Beat GEPA Pareto selection on 3/4 benchmarks under a fixed 1,500-eval
  budget. Caveats: unreviewed, single-run, GEPA at defaults. Pilot-worthy
  hypothesis, not established.
- **[S2] HGM (ICLR 2026 Oral, arXiv:2510.21614).** Metaproductivity–
  Performance Mismatch: current benchmark score is a weak proxy (r≈0.27–0.44)
  for an agent's potential to yield good descendants. Clade-aggregated CMP
  parent selection beats per-candidate fitness (56.7% vs 53.3% DGM on
  SWE-bench-Verified-60 at 2.4–6.9× fewer CPU-hours). Applies to lineage
  trees generally; evidence is coding-agent archives only. HIGH confidence.
- **[S3] DGM ablations (ICLR 2026 poster, arXiv:2505.22954).** Archive-based
  parent selection (sample any archived candidate, weighted by promise)
  significantly beats greedy always-branch-from-best (50.0% vs 39.7%) and
  no-archive (23.0%). Single runs, ~$22k each, gaps >> measured run SD.
  HIGH confidence.
- **[S4] SICA (arXiv:2504.15228).** Minimal fully self-referential loop works:
  best archived agent becomes the meta-agent editing its own code. Gate is an
  explicit scalar utility U = 0.5·score + 0.25·(1−cost/$10) + 0.25·(1−time/300s).
  17%→53% on its own fitness subset (NOT held-out). Pattern value: cost/latency
  as first-class gate objectives. HIGH confidence in mechanism, not generalization.

## Evaluation-budget mechanics

- **[E1] ShinkaEvolve (ICLR 2026 poster, arXiv:2509.19349).** Three transferable
  sample-efficiency mechanisms: (a) parent sampling balancing explore/exploit,
  (b) code-novelty rejection-sampling — reject near-duplicate candidates BEFORE
  spending eval budget, (c) bandit selection over an ensemble of LLM mutation
  operators. Reached SOTA circle-packing in 150 evaluations (vs thousands).
  Also: the agent scaffold itself is an evolvable artifact. HIGH (SOTA margin
  razor-thin, 2-1 vote).
- **[E2] OpenEvolve (code-level verified).** Production open-source
  code-evolution infra: MAP-Elites quality-diversity + island model with ring
  migration; cascade (multi-stage) evaluation gating expensive stages behind
  cheap thresholds. NOT a prompt evolver (that claim was REFUTED 0-3 — do not
  cite the "+23% HotpotQA" number). HIGH confidence.
- **[E3] AlphaEvolve (DeepMind).** Heterogeneous LLM ensemble as mutation
  operator (fast model for breadth, strong model for depth; ensemble beats
  small-only in ablations). Boundary: fitness must be machine-gradeable —
  their own stated limitation; adaptation required for LLM-judged regimes.
  HIGH confidence.

## Methodology (noisy small-N LLM-judged eval)

- Sources verified in the sweep: arXiv:2411.00640, 2503.01747, 2512.21326
  (noise decomposition of LLM evals), 2601.20913 (valid inference with
  imperfect judges), 2512.06710, 2601.05420. The repo already cites two of
  these; the sweep confirms the area is active and the repo's noise-aware
  stance is current best practice.

## Refuted during verification

- OpenEvolve as a GEPA-comparable prompt evolver (+23% HotpotQA): 0-3 REFUTED.

## Open questions the frontier has not answered

1. Does Elo selection's budget advantage survive LLM-judged noisy fitness at
   N≈10–50 (this repo's regime)?
2. Can clade-level CMP be estimated for skill/prompt lineages (cheaper to
   expand, noisier per eval than coding-agent archives)? How many descendants
   per node before CMP beats gate scores?
3. How far do SWE-smith-style synthetic verifiable tasks extend beyond code
   repos to behavioral/discipline skills? (This repo found fixture synthesis
   intractable for most correction-mined task types.)
4. What is GEPA's overfitting trajectory under many iterations, and do
   noise-aware gates + held-out transfer checks catch it before promotion?
