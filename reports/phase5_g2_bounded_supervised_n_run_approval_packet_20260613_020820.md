# Phase 5 G2 Bounded Supervised N-run Approval Packet — 20260613_020820

## Status

```text
status: G2_BOUNDED_SUPERVISED_N_RUN_APPROVAL_PACKET_READY_NOT_EXECUTED
G2 execution started: false
cron/optimizer/model/API/PR/merge/deploy: false
```

## Current interpretation

Sunwoo approved creating the G2 approval packet now. This packet does **not** execute G2 by itself. Execution requires the next explicit phrase:

```text
G2 bounded supervised N-run execute GO under phase5_g2_bounded_supervised_n_run_approval_packet_20260613_020820.json
```

## Current repo / PR state

```text
branch: hse/phase5-continuous-loop-prep
HEAD: 2fe70c987f1ffc30c06c453f6b8ab4684e1b01b8
PR #108 head: 2fe70c987f1ffc30c06c453f6b8ab4684e1b01b8
PR mergeable: MERGEABLE
PR checks: 0 — MISSING_NOT_CI_PASS
```

## Prerequisite evidence

Fresh G1:

```text
terminal_status: G1_PASS_NO_ACTION
repeat_count: 1
side_effect_count: 0
g2_status before packet: BLOCKED_PENDING_HUMAN_REVIEW_AND_EXPLICIT_APPROVAL
```

Local verification gate:

```text
targeted Phase 5/tool-selection: 66 passed in 0.64s
full HSE suite: 360 passed, 11 warnings in 11.39s
```

Historical G2 pattern reference:

```text
status: G2_PASS_NO_ACTION
repeat_count: 3
side_effect_status: SIDE_EFFECT_ZERO
side_effect_count: 0
```

## Proposed G2 contract

```text
N: 3
N_max: 3
mode: manual_supervised_local_zero_budget
per_run_timeout_seconds_max: 1800
total_timeout_seconds_max: 5400
lock: output/phase5-continuous-loop/.phase5-g2-bounded-supervised.lock
```

Per-run pipeline:

1. `evolution.monitor.provenance_dataset`
2. `evolution.monitor.performance_snapshot`
3. `evolution.monitor.auto_triage`
4. `evolution.monitor.scheduler_dry_run`

Disallowed during G2 execution:

```text
cron_enabled_or_created: false
optimizer_execution_allowed: false
model_or_paid_api_budget_allowed: false
network_calls_allowed: false
external_pr_update_allowed: false
git_push_allowed: false
auto_merge_allowed: false
deploy_allowed: false
active_runtime_mutation_allowed: false
production_continuous_loop_enablement_allowed: false
```

## Stop conditions

- any component exits nonzero
- any run status is not `PASS_NO_ACTION`
- any `side_effect_count > 0`
- `review_required_count > 0`
- `dry_run_action_count > 0`
- lock already exists or cannot be removed
- unexpected worktree changes outside approved output/report artifacts
- timeout is hit

## Acceptance criteria before/for execution

| AC | Status before execution | Meaning |
|---|---|---|
| P5-G2-01 explicit G2 execution authorization | PENDING_NEXT_EXPLICIT_EXECUTE_GO | Need exact execute phrase. |
| P5-G2-02 fresh G1 reviewed and side-effect zero | PASS | Fresh G1 is PASS/zero side effect. |
| P5-G2-03 bounded N supervised runs | READY | N=3, manual foreground, no cron. |
| P5-G2-04 overlap/runaway controls | READY | Lock and timeout defined. |
| P5-G2-05 zero side-effect boundary | READY | Optimizer/network/model/PR/cron/mutation all disallowed. |
| P5-G2-06 parseable evidence packets | READY | JSON/Markdown output contract defined. |
| P5-G2-07 G3 blocked without review | READY | G3 remains blocked after G2 unless separately approved. |

## Recommended next action

If Sunwoo wants to execute the bounded supervised run, say:

```text
G2 bounded supervised N-run execute GO under phase5_g2_bounded_supervised_n_run_approval_packet_20260613_020820.json
```

Otherwise keep this as approval packet only.
