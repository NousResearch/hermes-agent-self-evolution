# Phase 5 P1 Finite Cron Soak Approval Packet

- Created: 2026-06-08T16:04:18+0200
- Status: `P1_FINITE_CRON_SOAK_APPROVAL_PACKET_READY_NOT_EXECUTED`
- PR: https://github.com/NousResearch/hermes-agent-self-evolution/pull/108
- Repo branch: `hse/phase5-continuous-loop-prep`
- Head at packet creation: `fab14ecf80a3d583b86b3cf242a8ead6c475f4c6`
- Checks caveat: `MISSING_NOT_CI_PASS`

## Scope

This is a packet-only approval artifact for a future P1 finite cron soak. No cron job was created, no scheduler was enabled, no optimizer was started, and no automatic PR loop was enabled.

The current action is limited to writing and verifying reviewer-facing packet artifacts.

## Evidence chain inputs

- Production-loop activation gate: `PRODUCTION_LOOP_ACTIVATION_GATE_DEFINED_NOT_ENABLED`
- G1 one-shot canary run: `G1_PASS_NO_ACTION`
- G2 bounded supervised run: `G2_PASS_NO_ACTION`
- G3 limited cron pilot run: `G3_CRON_PILOT_PASS_NO_ACTION`
- G3 side-effect ledger: `APPROVED_SIDE_EFFECTS_ONLY`
- G3 independent review acceptance: `G3_INDEPENDENT_READ_ONLY_REVIEW_ACCEPTED`
- P0 production enablement packet: `P0_PRODUCTION_ENABLEMENT_PACKET_READY_NOT_EXECUTED`
- P0 independent review acceptance: `P0_INDEPENDENT_READ_ONLY_REVIEW_ACCEPTED`

## Proposed future P1 soak bounds

These are proposed defaults for a later separately approved execution, not active settings:

- max fire count: `3`
- interval: `30 minutes`
- total TTL: `120 minutes`
- per-fire timeout: `<=1800 seconds`
- cron shape: Hermes cron, script-only, `no_agent=true`, repeat-bounded, profile `default`, delivery `origin`
- lock: skip-if-running with repo-local lock label
- kill switch: repo-local kill-switch label, fail-closed before pipeline execution
- allowed writes: repo-local output/report evidence only

Per-fire pipeline:

1. preflight
2. lock and kill-switch check
3. provenance dataset or latest sanitized input
4. performance snapshot
5. auto triage
6. scheduler dry-run
7. side-effect ledger
8. run summary report

Allowed terminal statuses:

- `P1_SOAK_PASS_NO_ACTION`
- `P1_SOAK_REVIEW_REQUIRED_NO_ACTION`
- `P1_SOAK_FAIL_BLOCKED`
- `P1_SOAK_ABORTED_BY_KILL_SWITCH`

## Preserved OFF boundaries

The following remain OFF and are not approved by this packet:

- production continuous loop
- long-running or unbounded cron
- threshold-triggered optimizer execution
- automatic PR loop
- auto-merge/deploy
- network/API/model budget spend
- credential modification
- active runtime/source/config/skill/memory mutation

## Future execution requirements

Actual P1 finite cron soak execution requires a later explicit approval naming P1 finite cron soak or equivalent bounds. Before execution, the invocation method, schedule, max fire count, interval, TTL, timeout, delivery target, lock path, kill switch, allowed write roots, and stop conditions must be re-confirmed.

Any weak metric or review-required state must stop escalation. Optimizer execution and automatic PR updates remain disallowed.

## Next gate

Recommended next step after this packet is verified is path-limited PR publication of the P1 approval packet. Actual P1 execution remains blocked pending separate explicit approval.

Production activation remains `NO_GO`.
