# Proposal-debate workflow

How to wire the panel prompts into a multi-agent run that produces a vetted
roadmap. Works with any orchestrator that can spawn parallel subagents (Claude
Code Workflow/Agent tools, or manual copy-paste into separate sessions).

## Stages

```
context pack ──► [5 domain experts, parallel, blind] ──► proposals
proposals ──► [red-team skeptic] ──► verdicts (KILL / CUT DOWN / ADVANCE)
verdicts ──► [orchestrator pressure-test vs codebase] ──► verified set
verified set ──► [synthesis] ──► ranked roadmap, each item with kill experiment
roadmap ──► [staff-engineer synthesis review] ──► sequencing/merge/omission fixes
```

The synthesis-review stage reviews the PACKAGING, not the individual items:
wave ordering, hidden contradictions between merged items, omissions the panel
under-weighted, re-proposals of shipped work, and whether the roadmap
compounds toward the project's purpose. Item-level relitigating is out of
scope — the skeptic already did that. Observed yield: a single review pass
caught an internally contradictory merge, a buried highest-EV-per-dollar item,
a calibration promise the data couldn't support, and a missing signal
generator — defects invisible at item granularity.

## Rules that make it work

1. **Round-1 blindness.** Experts must not see each other's proposals; shared
   context comes only from the context pack. Convergent proposals from blind
   experts are evidence of importance; engineered consensus is not.
2. **The skeptic never proposes.** A reviewer with a competing proposal is a
   conflicted reviewer. The skeptic's output is verdicts + minimal-version
   prescriptions only.
3. **The orchestrator owns verification.** Subagent claims about GEPA/DSPy
   internals, repo seams, shipped capabilities, or experiment history are
   checked against source before synthesis (grep/read, file:line evidence).
   Claims failing verification are dropped or explicitly downgraded — never
   silently repaired, never passed through on authority.
4. **Convergence + survival ranks items.** An item proposed independently by
   ≥2 experts that survives the skeptic outranks a brilliant orphan. Orphans
   survive only with a cheap kill experiment attached.
5. **Carry nulls forward.** Killed proposals and the evidence that killed them
   go into the campaign record, so the next campaign's context pack lists them
   and no one re-litigates.

## Structured output schema (per proposal)

```json
{
  "title": "...",
  "claim": "...",
  "mechanism": "...",
  "signal_source": "...",
  "kill_experiment": "...",
  "effort": "S|M|L",
  "cost_to_validate_usd": "1|10|100",
  "verdict": "ADVANCE|CUT_DOWN|KILL",
  "verdict_reason": "...",
  "verified_against_code": true,
  "verification_notes": "file:line evidence"
}
```

## Cost notes (observed)

- One panel round (5 experts + skeptic, sonnet-class): ~$1-3, ~5-10 min wall.
- Research fan-out (5 scouts + verification): ~$2-5, ~10-20 min wall.
- Orchestrator verification: cheapest stage, highest value — do not skip.
