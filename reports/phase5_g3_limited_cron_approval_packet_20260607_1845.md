# Phase 5 G3 Limited Cron Approval Packet — Ready Not Executed

## Conclusion

Status: `G3_LIMITED_CRON_APPROVAL_PACKET_READY_NOT_EXECUTED`.

Sunwoo's `rec action GO` is interpreted narrowly: prepare and publish a G3 limited cron approval packet for PR evidence review. This packet does **not** create a cron job, enable a production loop, start an optimizer, enable an automatic PR loop, merge, deploy, spend budget, or mutate the active Hermes runtime.

## Current Evidence Chain

- PR: https://github.com/NousResearch/hermes-agent-self-evolution/pull/108
- Branch: `hse/phase5-continuous-loop-prep`
- Head before packet: `f5e89d0997739a356c4c6374ee32aba061665ff2`
- Prior gate: `reports/phase5_production_loop_activation_gate_20260607_1550.json` → `PRODUCTION_LOOP_ACTIVATION_GATE_DEFINED_NOT_ENABLED`
- G1 run: `reports/phase5_g1_one_shot_canary_run_20260607_161618.json` → `G1_PASS_NO_ACTION`
- G2 run: `reports/phase5_g2_bounded_supervised_run_20260607_183053.json` → `G2_PASS_NO_ACTION` / `BLOCKED_PENDING_INDEPENDENT_REVIEW_AND_EXPLICIT_APPROVAL`
- G2 ledger: `reports/phase5_g2_bounded_supervised_side_effect_ledger_20260607_183053.json` → `SIDE_EFFECT_ZERO`, side-effect count `0`

## What This Packet Allows Now

Only these actions are in scope for this packet publication step:

1. write reviewer-facing G3 approval packet JSON/Markdown under `reports/`;
2. validate the packet and focused Phase 5 monitor tests;
3. path-limited commit/push of the packet into PR #108 evidence chain;
4. update PR #108 metadata/body to point reviewers at the new packet.

## What Remains Disabled Now

- production continuous loop
- cron creation or scheduling
- threshold-triggered optimizer execution
- automatic PR creation/update loop
- auto-merge
- deploy
- network/model/API spend
- active Hermes runtime/source/config/skill/memory mutation
- credential or permission changes

## Future G3 Limited Cron Pilot Contract

Actual G3 execution remains blocked until independent review and a separate explicit approval.

Proposed future pilot bounds:

- cron type: expiring one-shot or repeat-limited only
- max fire count: `1`
- max wall-clock per fire: `1800s`
- expiry: first fire completion or 2 hours after creation, whichever comes first
- concurrency: skip-if-running and abort-on-existing-lock
- lock path: `output/phase5-continuous-loop/.phase5-g3-limited-cron.lock`
- kill switch path: `output/phase5-continuous-loop/.phase5-g3-kill-switch`
- allowed writes only under:
  - `output/phase5-continuous-loop/g3-limited-cron-pilot-<stamp>/`
  - `reports/phase5_g3_limited_cron_pilot_run_<stamp>.json`
  - `reports/phase5_g3_limited_cron_pilot_run_<stamp>.md`
  - `reports/phase5_g3_limited_cron_pilot_side_effect_ledger_<stamp>.json`

Future pipeline:

```text
preflight -> provenance_dataset -> performance_snapshot -> auto_triage -> scheduler_dry_run -> semantic_safety_verifier -> side_effect_ledger -> summary_report -> auto_disable_or_expire_cron
```

## Threshold / Optimizer Handling

If a weak metric appears during the future G3 pilot, the pilot must log `review_required` and stop. The threshold-triggered optimizer remains a dry-run signal only. Actual optimizer execution requires a later candidate-only optimizer approval packet and separate explicit approval.

## Automatic PR Loop Handling

Automatic PR creation/update remains disabled in G3. Reviewer-facing PR evidence updates remain manual, path-limited, and separately approved.

## Acceptance Criteria

- Packet status is `G3_LIMITED_CRON_APPROVAL_PACKET_READY_NOT_EXECUTED`.
- Packet does not create cron jobs or enable production automation.
- Future G3 pilot is bounded by max fire count, timeout, lock, kill switch, and allowed write roots.
- Threshold-triggered optimizer and automatic PR loop remain disabled in G3.
- Human review and separate explicit approval remain required before actual cron pilot.
- JSON/Markdown reports contain no private absolute paths or sensitive material.

## Recommended Next Action

Obtain independent reviewer acceptance of PR #108 G2/G3 evidence. If accepted and Sunwoo still wants to proceed, request a separate explicit approval for exactly one G3 limited cron pilot with `max_fire_count=1`.
