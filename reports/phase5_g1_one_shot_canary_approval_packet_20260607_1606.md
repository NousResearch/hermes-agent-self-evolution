---
title: "HSE Phase 5 G1 One-shot Canary Approval Packet"
created: "2026-06-07T18:08:20+0200"
source: Hermes EvAH
channel: discord
type: hse-phase5-approval-packet
status: final
tags:
  - eva/response
  - hse/phase5
  - production-loop
  - canary
  - approval-packet
---

# HSE Phase 5 G1 One-shot Canary Approval Packet

## 결론

`G1_ONE_SHOT_CANARY_APPROVAL_PACKET_READY_NOT_EXECUTED` 상태의 승인 패킷을 작성했다.

이 문서는 **G1 실행 승인서가 아니라, G1을 실행하기 전에 검토·승인해야 하는 실행 경계 계약**이다. 현재 시점에는 canary 실행, cron 생성, production loop enablement, optimizer 실행, network/model/API budget 사용, active Hermes runtime mutation, external PR automation, auto-merge/deploy를 수행하지 않았다.

## 근거 상태

- Repo: `/Users/snw/.hermes/evolution/repos/hermes-agent-self-evolution`
- Branch: `hse/phase5-continuous-loop-prep`
- HEAD: `9a5e077119e7cd9cec58f141b26a8d0acecb73ab`
- PR #108: https://github.com/NousResearch/hermes-agent-self-evolution/pull/108
- PR state: `OPEN`
- PR checks reported: `0`
- Formal completion: `FORMAL_PHASE5_COMPLETE_LOCAL_WITH_EXPLICIT_WAIVER`
- Production gate: `PRODUCTION_LOOP_ACTIVATION_GATE_DEFINED_NOT_ENABLED`
- Prior bounded L1 run: `PASS_NO_ACTION`

## 현재 activation flags

- `production_continuous_loop_enabled`: `false`
- `continuous_loop_enabled`: `false`
- `cron_jobs_created`: `false`
- `optimizer_execution_started`: `false`
- `active_runtime_mutation`: `false`
- `external_calls_performed`: `false`
- `network_calls_performed`: `false`

## 지금 승인된 작업

- G1 one-shot canary approval packet 작성
- repo JSON/Markdown artifact 작성
- Obsidian/Discord Markdown copy 작성
- JSON parse, side-effect flag, secret scan, attachment identity 검증

## 지금 승인되지 않은 작업

- execute the G1 canary
- create or enable cron jobs
- start production continuous loop
- execute optimizer or model/API benchmark calls
- mutate active Hermes runtime/source/config/skill/memory
- update external PRs automatically from a canary run
- auto-merge or deploy

## G1 실행 계약 초안

- Execution mode: `manual_one_shot_local_cli`
- Repeat count max: `1`
- Wall-clock timeout max: `1800s`
- Concurrency lock: `output/phase5-continuous-loop/.phase5-g1-one-shot-canary.lock`
- Network/model/API budget: `0` by default
- Optimizer execution: `false`
- Cron creation: `false`
- Production loop enablement: `false`
- External PR update: `false`
- Auto-merge/deploy: `false`
- Active runtime mutation: `false`

Allowed write roots for a future separately approved G1 run:

- `output/phase5-continuous-loop/production-canary-<stamp>/`
- `reports/phase5_g1_one_shot_canary_run_<stamp>.json`
- `reports/phase5_g1_one_shot_canary_run_<stamp>.md`
- `reports/phase5_g1_one_shot_canary_side_effect_ledger_<stamp>.json`

## Proposed pipeline

1. `preflight` — confirm branch/head/status, lock absence, allowed roots, and no activation flags
2. `provenance_dataset` — generate sanitized local metrics input from approved local evidence
3. `performance_snapshot` — evaluate metric thresholds without side effects
4. `auto_triage` — rank any weak targets without taking action
5. `scheduler_dry_run` — simulate scheduler decision only; create no cron job
6. `side_effect_ledger` — record all side-effect flags and fail closed if any unexpected true flag appears
7. `summary_report` — write G1 JSON/Markdown run summary and human handoff

## Logical command template — not executed

```bash
cd /Users/snw/.hermes/evolution/repos/hermes-agent-self-evolution
PY=~/.hermes/evolution/venvs/self-evolution/bin/python
STAMP=$(date -u +%Y%m%d-%H%M%S)
RUN_ROOT=output/phase5-continuous-loop/production-canary-$STAMP
LOCK=output/phase5-continuous-loop/.phase5-g1-one-shot-canary.lock
abort if $LOCK exists; create lock; run exactly one provenance_dataset -> performance_snapshot -> auto_triage -> scheduler_dry_run pipeline; write reports/phase5_g1_one_shot_canary_run_$STAMP.{json,md}; remove lock after completion
```

## Pre-execution checklist

- [ ] `G1-PRE-01` Sunwoo gives explicit G1 execution approval naming this packet or equivalent bounds
- [ ] `G1-PRE-02` git worktree is clean or dirty scope is declared and excluded from canary writes
- [ ] `G1-PRE-03` concurrency lock path is absent before start
- [ ] `G1-PRE-04` allowed write roots exist or can be created under repo-local output/reports only
- [ ] `G1-PRE-05` timeout wrapper and kill switch are printed before run
- [ ] `G1-PRE-06` network/model/API budget remains zero unless a separate budget approval is recorded
- [ ] `G1-PRE-07` cron/optimizer/production-loop/external-PR/auto-merge/deploy remain disabled

## Acceptance criteria

- `P5-G1-01` **explicit_one_shot_authorization** — G1 execution may start only after explicit approval of this packet or equivalent bounds.
- `P5-G1-02` **bounded_runtime_and_concurrency** — Exactly one run, timeout <=1800s, abort-on-lock concurrency guard, and documented kill switch.
- `P5-G1-03` **side_effect_zero_by_default** — No cron, optimizer, production loop, network/API calls, runtime mutation, external PR update, auto-merge, or deploy.
- `P5-G1-04` **reviewable_evidence_packet** — Run emits parseable JSON/Markdown summary, side-effect ledger, command ledger, and reviewer handoff.
- `P5-G1-05` **fail_closed_status_contract** — Any unexpected side effect, parse failure, lock conflict, budget breach, or unsafe data finding yields blocked/review-required status, not PASS.
- `P5-G1-06` **next_stage_blocked_without_review** — G2 bounded supervised runs remain blocked until G1 evidence is reviewed and explicitly approved.

## Allowed terminal statuses for future G1 run

- `G1_PASS_NO_ACTION`
- `G1_PASS_REVIEW_REQUIRED_NO_ACTION`
- `G1_FAIL_BLOCKED`
- `G1_ABORTED_BY_KILL_SWITCH`

## Kill switch contract

- Before start: print process/session id, timeout, lock path, and run root.
- During run: terminate tracked process/session on user stop, timeout, or unsafe output.
- After stop: verify process stopped, write abort summary, then remove lock only if owned by this run.

## Post-run review requirements

- JSON parse for every report
- Side-effect ledger all false for blocked flags
- High-risk secret/private-data scan has no findings
- Scheduler dry-run created no cron and took no action
- Independent read-only review before G2
- Human decision recorded before cron or optimizer activation

## Recommended next step/action

이 패킷을 PR #108 evidence chain에 포함하려면 다음은 **path-limited commit/push**가 안전하다. 실제 G1 canary 실행은 이 문서만으로는 승인되지 않으며, 별도 명시적 실행 승인 후에만 진행한다.
