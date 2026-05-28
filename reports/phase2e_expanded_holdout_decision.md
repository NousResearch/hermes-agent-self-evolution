# Phase 2E Expanded Holdout Decision

Decision: current 45+9 slice is sufficient for Phase 2 closeout.

100+ held-out quality slice required before Phase 2 closeout: no
Candidate-only/no-apply: yes

## Coverage snapshot

- Default gate cases: 45
- SessionDB holdout cases: 9
- Combined slice cases: 54
- Combined expected tools: 29
- Combined confusion pairs: 88

## Coverage delta from holdout

- New expected tools from holdout: none
- New confusion pairs from holdout: none
- Overlapping user requests: none
- Overlapping categories: none

## Evidence

- Default case count OK: true
- Holdout case count OK: true
- Holdout disjoint from default: true
- Holdout review passed: true
- Holdout review contract OK: true
- Holdout failed-checks contract OK: true
- Holdout metric deltas: `{"argument_cue_coverage": -0.0502, "constraint_pass_rate": 0.0182, "selection_accuracy": 0.7778, "wrong_tool_avoidance": 0.7778}`

## Policy

- 45-case default gate plus 9-case SessionDB holdout is sufficient for candidate-only Phase 2 closeout when the holdout adds no new expected tools/confusion pairs and the heldout review passes.
- Defer 100+ slice until: before any default-gate promotion, active tool-schema apply, or broader Phase 3/benchmark expansion that needs lexical diversity beyond the current candidate-only gate.
- Rationale: The SessionDB holdout adds sanitized real-session variants but no new expected-tool or confusion-pair coverage beyond the default gate; current evidence supports closeout without turning Phase 2E into a 100+ case expansion project.

## Remaining Phase 2 closeout items

- `benchmark_gate_decision`
- `human_review_checkpoint`
