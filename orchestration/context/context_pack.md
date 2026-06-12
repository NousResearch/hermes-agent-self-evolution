# Context pack — agent-self-evolution

Point-in-time briefing for panel/research subagents. The orchestrator must
refresh this before each campaign; a stale pack produces confident nonsense.

## What the project is

A standalone optimization pipeline that improves an agent's artifacts —
SKILL.md files, tool descriptions, system-prompt sections — via DSPy + GEPA
reflective evolutionary search, with statistical deploy gates sized for the
small-N (20–60 examples), noisy-LLM-judge regime GEPA wasn't designed for.
Operates ON a target agent repo (Hermes reference; Claude Code supported),
never inside it. Everything is API calls; no GPU training. Typical run $1–20.

## Pipeline in one paragraph

Artifact text is wrapped as a `dspy.Module`; GEPA mutates instruction text
using execution-trace feedback; candidates are scored by a composite LLM judge
(correctness/procedure/conciseness); a knee-point band picks among
statistically indistinguishable candidates; a paired-bootstrap quality gate on
a held-out split decides deploy. An orthogonal closed-loop surface runs a real
agent (`hermes -z` or `claude -p`, splice-and-restore install with
flock/backup/sha256 guards) against JSONL behavioral task suites — usable as
post-gate veto, reflection feedback, or trainset score channel. A saturation
pre-flight refuses to spend budget when the baseline already aces both
signals. Every run writes `gate_decision.json` (v5), lineage, and a
non-attributing review dossier.

## Shipped (do not re-propose)

- Knee-point ε-band selection (ε variants measured a no-op on val-best; selector
  dropped from that path, `smallest` strategy kept for compression users)
- Paired-bootstrap gate with no-regression / dual-check / non-inferiority rules
  (non-inferiority tolerance 0.05 measured strictly dominant for compression)
- GEPA acceptance `improvement_or_equal` default (upstream gepa PR, ties accepted)
- `--gepa-minibatch-size` exposure (larger minibatch = pareto report "Path E")
- Saturation pre-flight with healthy/no_headroom/weak_signal/uniform_failure
  bands; default-deny non-interactive
- Closed-loop validation: 3 modes (veto / reflection feedback / trainset
  behavioral examples that join GEPA's acceptance sums and per-instance frontier
  via `--closed-loop-in-valset`)
- Noise-aware CL gate: A/A probe writes `<suite>.noise.json`; win threshold
  rises above measured flip floor
- Per-task suite discrimination labeler (too_easy / discriminative /
  baseline_fails / noise_limited)
- Suite-constraint compiler: zero-LM "constraint floor" prompt from suite
  metadata (measured: captures ~79–85% of SKILLS_GUIDANCE headroom — "the
  suite states the win")
- Hermes agent-subprocess LM spend captured from state.db `actual_cost_usd`
  into `--max-cost-usd` (Claude backend: not captured)
- PR automation (`--create-pr`, direct-push, branches from `origin/<base>`);
  deferred for prompt sections (local-splice PR would carry unrelated diff)
- Evidence-linked dossiers: lineage.json + diff + selection metadata; per-hunk
  attribution deliberately rejected as statistically fragile
- Claude Code backend: CLAUDE.md sentinel-region evolution, hermetic
  `claude -p` validation via `--append-system-prompt-file`, sandbox containment
- SessionDB mining for tool misselections (Hermes only, confidence-banded judge)
- Cost ledger + ceiling kill-switch, per-attempt LM observability, cache-aware

## Completed-experiment findings (nulls bind; cite when relevant)

1. **Quality-section prompt headroom: NULL** on capable agents (memory-content
   sections). Discipline/convention sections DO have headroom — but two
   necessary conditions: schema-inert (tool schema doesn't already enforce it)
   AND promptable behavior.
2. **Model-tier binarity:** closed-loop signal flips all-pass ↔ all-fail
   between adjacent model tiers (mini=7/7, nano=0/7 on the write_file suite).
   No smooth middle observed yet; the Goldilocks validator model is unsolved.
3. **Saturated baselines:** on hand-tuned artifacts (write_file, search_files,
   live MEMORY_GUIDANCE) the pipeline is regression-catching, not
   improvement-finding. Demonstrated improvements required actively-misdirected
   adversarial baselines.
4. **Acceptance, not frontier, was the GEPA bottleneck** on saturated runs:
   stochastic 3-example minibatch + sum acceptance discards per-instance
   behavioral signal (~38% sampling probability at 7/46 failing). Frontier
   extension (cartesian objectives) is moot until proposals pass acceptance.
   Remaining unimplemented fix: stratified minibatch sampling ("Path C").
5. **"The suite states the win":** a zero-LM compiler over suite constraints
   captures most measured headroom; GEPA's residual over that floor is within
   noise on the conventions suite. Suite authorship is where the value lives.
6. **Synthetic dataset generation caps out:** LM returns far fewer distinct
   valid cases than requested (drop rates 51–94%); sub-2500-char skills are
   unsuitable for the synthetic pipeline regardless of requested N.
7. **Passive weakening of a baseline doesn't move validators; active
   misdirection does.** Ablation baselines must redirect agent intent.
8. **A/A noise floors are real and measurable** (flip rates on identical
   artifacts); gates that ignore them ship noise.

## Open problems (verified current as of this pack)

- **Path C unimplemented:** no stratified/failure-aware minibatch sampling;
  behavioral signal still randomly missed at acceptance on small minibatches.
- **Goldilocks validator model:** no automated search/guidance for the model
  tier where behavioral tasks discriminate (binary tier-flip observed).
- **Binary behavioral verdicts:** pass/fail per task; no partial credit
  (first-try vs backtrack, wrong-call count), so tier-flips are cliffs.
- **Suite lifecycle:** discrimination labeler ships, but no closed loop from
  labels → suite repair/generation; correction-mining tractable only for
  "use X not Y" convention tasks (fixture synthesis blocks the rest).
- **Serialized closed-loop:** one shared live install ⇒ flock-serialized
  candidate evaluation; no parallel worktree/container isolation.
- **Splice machinery untested under concurrency/staleness/YOLO-agent
  mutation;** stale-backup recovery is manual.
- **Phase 4 (code evolution) and Phase 5 (continuous loop): empty stubs**
  (`evolution/code/`, `evolution/monitor/`). Original-plan sketches exist
  (Darwinian Evolver external CLI; monitor/triage/cron) but are not binding.
- **Saturation thresholds are magic numbers** (0.99/0.95/0.15), uncalibrated.
- **Claude backend gaps:** no skill closed-loop suites, no session-log mining,
  no subprocess cost capture, prompt-section PR automation deferred.
- **MIPROv2 fallback loses lineage/dossier narrative.**
- **Layer-2 (content) verdicts wired for prompts only;** skill closed-loop is
  membership-only.

## Extension seams (cheap plug-in points)

`SkillSource` / `ToolSource` / `PromptSource` protocols (new framework
adapters ~50–200 lines); `ArtifactInstaller` (new install surface);
`AgentRunner` (new agent backend); `PromptBackend` factory (source + installer
+ runner triple per target); proposer modes; fitness dimensions; Layer-2 judge
factories; provider probers; `--benchmark-cmd` external gate hook;
`--baseline-override-file` ablation hook; behavioral examples → trainset/valset
plumbing; suite compiler + discrimination labeler as suite-lifecycle building
blocks.

## License constraint

MIT repo. AGPL tools (Darwinian Evolver) integrate as external CLI only —
no Python imports.
