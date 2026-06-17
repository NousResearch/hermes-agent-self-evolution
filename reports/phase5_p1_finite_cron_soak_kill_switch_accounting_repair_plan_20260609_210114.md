# HSE Phase 5 P1 Kill-switch / Cron-accounting Repair Plan — 20260609_210114

- Status: `P1_FINITE_CRON_SOAK_KILL_SWITCH_ACCOUNTING_REPAIR_PLAN_READY_NOT_EXECUTED`
- Verdict: `PLAN_ONLY_READY`
- Generated: 2026-06-09 21:01:14 +0200 CEST
- Target job: `b24aca09f168`
- Current action: plan only; no PR update

## Current state

- Cron job: `state=paused`, `enabled=False`, `repeat=2/3`
- Actual soak state: `1/3` completed fire in `state.json` / run report
- Kill switch: present at `output/phase5-continuous-loop/p1-finite-cron-soak-20260608-164455/P1_FINITE_CRON_SOAK_KILL_SWITCH`
- Lock: `False`
- `fire-02`: `False`
- Last resume attempt: `FAIL_CLOSED_BLOCKED_BY_EXISTING_KILL_SWITCH`, successful additional fires `0`

## Diagnosis

The previous resume approval triggered the cron job once, but the script failed closed before creating `fire-02` because the existing run-root kill switch was still present. Hermes cron nevertheless advanced repeat accounting to `2/3`; actual soak evidence still shows only `1/3` completed fire. Therefore future action must choose an explicit repair strategy instead of blindly resuming the old job.

## Boundaries preserved now

- Kill switch cleared now: `false`
- Cron accounting edited now: `false`
- Cron resumed now: `false`
- Remaining fires run now: `false`
- Optimizer: `off`
- Automatic PR loop: `off`
- PR update: `not performed`

## Repair options for future separate approval

### R1 — Recommended: replacement job, no old accounting edit

Leave `b24aca09f168` paused as historical evidence. After explicit approval, archive/clear the kill switch with evidence, then create a fresh bounded `no_agent` replacement job for exactly two remaining actual fires using the existing state/report lineage. This avoids editing `repeat.completed` on the old cron job.

### R2 — Same-job accounting repair

Back up cron storage, edit `b24aca09f168` repeat accounting from `2/3` back to `1/3`, archive/clear the kill switch, then resume/trigger the same job. This preserves the job id but is riskier because it mutates cron accounting.

### R3 — One-fire salvage

Do not repair accounting; only archive/clear the kill switch and run the old job once more. This is not recommended because it cannot satisfy max additional fires `2`.

## Recommended future plan: R1

1. Reconfirm `b24aca09f168` remains paused/enabled=false and no lock exists.
2. Back up and hash `jobs.json`, `state.json`, run report, side-effect ledger, kill switch, and attempt artifacts.
3. With explicit approval, archive or remove the kill switch while preserving its hash/content in repair evidence.
4. Create a fresh bounded `no_agent` replacement cron job with `repeat=2`, `workdir=<hse-repo>`, `profile=default`, `deliver=origin`, same script, optimizer off, automatic PR loop off.
5. Trigger one fire at a time and inspect state/report/scheduler output after each fire.
6. Stop immediately on `review_required=true`, `dry_run_action_count>0`, `side_effect_count>0`, disallowed side effects, optimizer/network/runtime mutation request, lock, or kill-switch recurrence.
7. Leave cron paused/disabled or expired at the end and write local evidence only unless PR publication is separately approved.

## Still blocked without new approval

- clear/archive/remove kill switch
- edit `jobs.json` or cron repeat accounting
- resume or run `b24aca09f168`
- create replacement cron job
- run remaining fires
- start optimizer
- enable automatic PR loop / production loop
- commit, push, or update PR

## Recommended next step

If Sunwoo wants to continue, approve the recommended R1 repair execution explicitly: archive/clear the kill switch, create a replacement repeat=2 no_agent job, keep b24aca09f168 paused/retired, run at most two one-at-a-time NOOP-guarded fires, optimizer off, no automatic PR loop, no PR update.
