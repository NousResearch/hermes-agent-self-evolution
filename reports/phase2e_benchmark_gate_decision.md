# Phase 2E Benchmark Gate Decision

Decision: defer benchmark gate until Phase 3 or active apply.

Benchmark gate executed now: no
Blocking for Phase 2 candidate-only closeout: no
Candidate-only/no-apply: yes

## Deferred benchmarks

- TBLite
- YC-Bench

## Current evidence

- 45-case formal gate is green.
- 9-case SessionDB holdout is available and privacy-safe.
- Combined tool-selection slice case count: 54
- Heldout review passed.
- Expanded holdout decision does not require a 100+ slice before Phase 2 closeout.
- CI smoke is available for the candidate-only Phase 2D/2E path.

## Rationale

- Phase 2E is closing a candidate-only tool-description/tool-selection gate, not applying active Hermes tool schemas or system prompts.
- The 45-case formal gate, 9-case SessionDB holdout, heldout review, and CI smoke cover the Phase 2E closeout risk directly.
- TBLite and YC-Bench are broader runtime/system-prompt regression benchmarks and are more appropriate as Phase 3 or active-apply prerequisites.
- Running those benchmarks now would expand Phase 2E into benchmark-infrastructure validation rather than closing the current candidate-only workstream.

## Required before

Run or explicitly re-evaluate benchmark gates before:

- `phase3_execution`
- `active_tool_schema_apply`
- `default_gate_promotion`
- `system_prompt_evolution_acceptance`

## Remaining Phase 2 closeout items

- `human_review_checkpoint`
