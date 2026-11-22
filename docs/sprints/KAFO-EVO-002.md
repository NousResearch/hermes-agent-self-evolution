# KAFO-EVO-002 — Real-World Evidence for Adaptive Hermes: Sprint Plan

Status: **draft plan** (owner-approved for drafting, 2026-11-22). Nothing in
this document has been built yet unless a gate says so.

## Objective

Find out whether the KAFO-EVO-001 loop improves Hermes on **real** code
review, not only on hand-written synthetic cases. Concretely:

1. Measure skills and strategy routing on a benchmark mined from real bug-fix
   commits, large enough to detect a 0.05 utility gain.
2. Stop GEPA from overfitting the skill to the benchmark's format or cases.
3. Give the bandit arms that differ in cost and quality, so routing can save money
   at equal quality, instead of choosing between near-identical prompts.
4. Replay real sessions and run new skills in shadow mode before anything
   reaches users.
5. Route an admitted skill upstream only as a human-reviewed hermes-agent PR.

Preserve everything KAFO-EVO-001 delivered: the sealed-split guard,
admission gate, checkpoint/rollback and spend ledger stay in force.

## Lessons from KAFO-EVO-001

These shape the gates below. They come from the KAFO-EVO-001 sprint report
(`reports/kafo-evo-001/`) and the full live run review reported by the
orchestrator (2026-11-22). No sealed test case content is reproduced here.

| Lesson | Consequence for this sprint |
|---|---|
| The benchmark was **synthetic**: 48 hand-written Python cases, single author. | Gate 2: real bug-fix commits, several languages, multi-file diffs. |
| **10 of 48 labels were ambiguous** (more than one defensible location or category). | Gate 2: two or more independent labellers, measured agreement, multiple acceptable categories per defect. |
| The admitted GEPA skill passed admission only narrowly (**CI low bound 0.006**) on 10 validation cases. | Gate 2: power calculation up front; at least 50 validation cases. |
| GEPA **overfit to output format and memorised train cases** (the evolved skill echoed train-case specifics). | Gate 3: anti-overfitting guards before admission. |
| The bandit **did not beat `fixed:skill`** (and on synthetic data no adaptive policy beat the best fixed strategy on validation). | Gate 4: arms that differ in cost and quality (model, passes, tools), with contexts observable upfront. |
| Run state was held in local files and per-process SQLite. | Gate 1: persistent run storage. |
| `evolve_skill` now defaults to an LLM judge that needs judge credentials (PR #18). | Gate 7: fix the default so runs fail fast with a clear message or use an explicit choice. |

## Architecture changes (summary)

- **Knowledge:** run storage becomes persistent and queryable across runs
  (gate 1), instead of per-run directories.
- **Monitor / Analyse:** a real-world benchmark with location-first scoring
  (gate 2) replaces the synthetic Python set for all claims.
- **Plan:** strategy arms become cost-routing arms (gate 4).
- **Legitimise:** admission gains anti-overfitting checks and a fixed
  regression set (gate 3).
- **Execute:** shadow mode runs candidates next to the live skill without
  affecting users (gate 5).
- **Slow evolution:** an admitted skill leaves the system only as a
  human-reviewed PR to hermes-agent (gate 6).

## Sprint 02 acceptance gates

Priority order. A gate is ticked only when its evidence is merged and green
on remote CI; live results also need the owner-approved spend cap.

- [ ] **1. Persistent run storage.** In progress by worker -3 (separate
      workstream; reference its PR here once open). Requirements this sprint
      relies on:
      - every run, admission decision, checkpoint, rollback and ledger entry is
        stored with run ID, git SHA, skill version and policy version;
      - storage is append-only for decisions and ledger rows;
      - results from different runs can be compared without re-running;
      - the sealed split and its results are stored separately and never
        joined into optimisation queries.
- [ ] **2. Real-world benchmark from bug-fix commits.**
      - **Source:** real bug-fix commits from permissively licensed public
        repositories, with the licence and commit URL recorded per case. Each
        case is the pre-fix diff under review; the fix commit supplies the
        ground-truth location.
      - **Coverage:** multi-file diffs; at least three languages (for example
        Python, TypeScript and Go), with each language's share recorded.
      - **Size:** at least 50 validation and 50 sealed test cases (plus a train
        split), fixed by the power calculation below **before** any labelling
        result is looked at. If the calculation asks for more, use more.
      - **Labelling:** at least 2 independent labellers per case, blind to each
        other. Agreement is measured and reported per field (location: Cohen's κ
        or Krippendorff's α on line-range overlap; category: κ). Target κ ≥ 0.6
        on location. Disagreements go to a third adjudicator. Cases still
        ambiguous after adjudication are dropped or marked, never silently kept.
      - **Multiple acceptable categories:** each defect stores a set of
        acceptable categories, not one.
      - **Scoring:** location first. A finding matches a defect when it
        overlaps the defect's line range in the right file. Primary score:
        location precision, recall and F1. Category is a secondary score,
        counted correct when it falls in the acceptable set.
      - **Separation:** same sealed-split machinery as KAFO-EVO-001 (manifest
        hashes, `optimisation_phase()` guard, logged one-time unsealing).
        Leakage check: no sealed case shares a repository commit or file with
        train or validation.
      - **Licensing and privacy:** no private code, no secrets (run the
        existing secret detector over every case), with attribution kept.
- [ ] **3. Anti-overfitting guards for GEPA candidates** (checked before the
      KAFO-EVO-001 admission run; a failure rejects the candidate with reasons):
      - **No case leakage:** reject a candidate that mentions any benchmark
        case ID, or quotes benchmark code (n-gram overlap with any train or
        validation case above a fixed threshold, for example 8 consecutive
        tokens).
      - **Core sections preserved:** the skill's declared core sections
        (headings listed in a fixed manifest per skill) must still be present
        and non-empty.
      - **Size limit:** an absolute byte cap and a maximum growth over the
        parent (reuse `max_prompt_growth`); no gutting either, so a minimum
        size of 50% of the parent is enforced as well.
      - **Regression set:** a fixed, versioned set of GitHub-workflow tasks for
        the skill's primary mode (Mode A: the `github-code-review` PR-review
        workflow), scored separately. A candidate may not fall below the
        parent on it by more than a fixed margin, so Mode A cannot be gutted
        to win the benchmark. The regression set is never used by GEPA for
        reflection.
      - Each guard has a test that a crafted bad candidate is rejected.
- [ ] **4. Cost-routing strategy arms.**
      - **Arms:** cheap vs strong model; single pass vs two-pass
        (review, then self-check); with tools (allowlisted, read-only) vs
        none. Start with at most 4 arms to keep the comparison powered.
      - **Contexts observable before the decision:** diff size (lines, files),
        language(s), domain (for example web, data, infra), and nothing derived
        from outcomes or labels.
      - **Same budgets for every arm:** budgets are set per executor, not per
        arm; any cost cap applies to all arms equally.
      - **Report:** quality and cost per arm, and for each adaptive policy the
        cost saving at non-inferior quality (non-inferiority margin fixed
        upfront), with a CI. "No saving detected" must be reported as such.
- [ ] **5. Real-session replay and shadow mode.**
      - **Replay:** use the existing session importers
        (`evolution/core/external_importers.py`: Claude Code, Copilot, Hermes)
        to build a replay set of real review requests. Secret detection and
        PII scrubbing run first; the user must opt in to which sources are
        read.
      - **Shadow mode:** a candidate skill runs next to the live skill on the
        same request. Its output is logged and scored but never shown to the
        user and never acts (read-only tools only). Shadow results feed
        evidence, not automatic promotion.
      - **Agreement metrics:** shadow-vs-live finding overlap, cost and latency,
        reported per context.
- [ ] **6. Upstream path to hermes-agent.**
      - An admitted skill version becomes a **pull request** to hermes-agent
        containing the skill diff, the admission evidence (decision, CI,
        regression-set result, guard results) and the run ID.
      - Always human-reviewed; **never auto-merged**. The automation has no
        merge permission on hermes-agent.
      - A rejected or rolled-back candidate never produces a PR.
- [ ] **7. Operations.**
      - **Pin dspy** (and gepa) to tested versions in `pyproject.toml`. The
        offline GEPA tests depend on dspy 3.5 internals (engine API, the
        ChainOfThought `plan` field name); upgrades become deliberate PRs.
      - **Nightly live canary:** a small scheduled workflow (a few validation
        cases, one strategy, strict spend cap per night) on a GitHub-hosted
        runner within the org's USD 250/month Actions budget. LLM API spend is
        separate from Actions minutes and needs an owner-approved cap and an
        API-key secret (owner decision). The canary fails loudly; it never
        admits or deploys anything.
      - **Fix the `evolve_skill` judge default:** with no judge credentials,
        `--metric judge` (default since PR #18) fails late. Add a preflight
        check that fails fast with a clear message (and is reported by
        `--dry-run`), or require an explicit `--metric` choice. Never fall back
        silently to the keyword proxy.

## Power calculation (gate 2)

Admission tests a paired per-case utility difference `d` (candidate minus
baseline, mean over seeds). To detect a mean gain of δ = 0.05 with a
two-sided α = 0.05 and 80% power:

```
n = ((z_{1-α/2} + z_{1-β}) · σ_d / δ)²  =  ((1.96 + 0.84) · σ_d / 0.05)²  ≈  3136 · σ_d²
```

| σ_d (per-case SD of the gain) | Cases needed |
|---|---|
| 0.10 | 32 |
| 0.126 | 50 |
| 0.15 | 71 |
| 0.20 | 126 |
| 0.25 | 196 |

So 50 validation cases are enough only if σ_d ≤ 0.126. Procedure:

1. Estimate σ_d on a pilot (the first 20 labelled train cases, two arms).
   Do not use validation or sealed cases for this.
2. Set `n = max(50, ceil(3136 · σ̂_d² · 1.1))` (10% margin) for validation and
   the same for sealed test. Record σ̂_d, n and the date in this document
   before labelling the remaining cases.
3. If the required n cannot be labelled within the sprint, report the
   detectable effect size for the n achieved instead of claiming power.

## Scientific acceptance criteria

1. Every claim of improvement is made on the real-world benchmark (gate 2).
   Synthetic results stay labelled synthetic and are never used as evidence.
2. Sample sizes are fixed by the power calculation before results are seen.
3. Labels have at least two independent labellers with reported agreement;
   ambiguous cases are adjudicated or excluded, and the counts reported.
4. Location is the primary score; category is secondary with sets of
   acceptable categories.
5. No access to sealed test cases during optimisation, selection or
   admission; one logged final evaluation only.
6. Compared policies and arms get equivalent task distributions and budgets;
   contexts are observable before the decision.
7. Record action probabilities, policy, skill and benchmark versions for
   every decision (persistent storage, gate 1).
8. Report absent improvement explicitly ("no improvement detected", "no
   saving detected"); never infer generalisation from one split.
9. Evaluate policy adaptation, skill evolution and shadow agreement
   separately.

## Safety constraints

- CLI demonstrations stay offline and read-only by default; live runs need
  an owner-approved spend cap and the spend ledger.
- The agent cannot modify reward logic, evaluation data, labels, regression
  sets, permissions or budgets. Guards and thresholds are fixed in reviewed
  code and fingerprinted, as in KAFO-EVO-001.
- Violations stay hard disqualifiers for candidate admission.
- No automatic deployment: an admitted skill reaches hermes-agent only as a
  human-reviewed PR, never auto-merged (gate 6).
- Shadow mode never shows output to users and never takes actions; tools in
  shadow and replay are read-only and allowlisted.
- Real-session data: opt-in sources only, secret and PII scrubbing before
  storage, no raw session content in reports.
- Mined benchmark code: permissive licences only, attribution recorded, no
  secrets.
- **Never quote sealed test case content** in documents, reports, logs, PRs
  or skill text.

## Out of scope

- Training model weights.
- Auto-merging or auto-deploying skills.
- Changing the KAFO-EVO-001 sealed split or re-running its final evaluation.
