---
title: "HSE Phase 5 Production Loop Activation Gate"
created: "2026-06-07 15:50:25 UTC"
source: Hermes EvAH
channel: Discord
type: response
status: final
tags:
  - eva/response
  - hse/phase5
  - production-loop
  - activation-gate
related_artifacts:
  - reports/phase5_production_loop_activation_gate_20260607_1550.json
  - reports/phase5_production_loop_activation_gate_20260607_1550.md
---

# HSE Phase 5 Production Loop Activation Gate

## 결론

이 문서는 **production continuous loop / cron / optimizer enablement를 실제로 켜기 위한 문서가 아니라**, 켜기 전에 반드시 통과해야 하는 **별도 activation gate**입니다.

현재 상태는 다음과 같이 유지합니다.

- Formal Phase 5 local evidence: `FORMAL_PHASE5_COMPLETE_LOCAL_WITH_EXPLICIT_WAIVER`
- Production continuous loop enabled: `false`
- Cron jobs created: `false`
- Optimizer execution started: `false`
- Active runtime mutation: `false`
- External/network calls: `false`

따라서 이 문서의 완료 상태는:

```text
PRODUCTION_LOOP_ACTIVATION_GATE_DEFINED_NOT_ENABLED
```

즉, **gate는 정의됐고 activation은 아직 OFF**입니다.

## 현재 근거 상태

| 항목 | 값 |
|---|---|
| Repo | `/Users/snw/.hermes/evolution/repos/hermes-agent-self-evolution` |
| Branch | `hse/phase5-continuous-loop-prep` |
| HEAD | `8921affc22cac0978f95318b7dc4daa3bc82ef8a` |
| Git status | `## hse/phase5-continuous-loop-prep` |
| Formal artifact | `reports/phase5_formal_completion_20260607_091723.json` |
| Formal status | `FORMAL_PHASE5_COMPLETE_LOCAL_WITH_EXPLICIT_WAIVER` |
| Readiness manifest | `reports/phase5_continuous_loop_readiness_manifest.json` |
| Readiness status | `local_preparation_started_not_deployed` |
| L1 dry-run | `reports/phase5_bounded_unattended_run_20260607_090920.json` / `PASS_NO_ACTION` |
| L2 decision | `reports/phase5_l2_budgeted_decision_cycle_20260607_091723.json` / `PASS_NO_OP_DECISION` |
| Waiver | `reports/phase5_optimizer_handoff_explicit_waiver_20260607_091723.json` / `WAIVER_APPROVED_AND_RECORDED` |

## 왜 별도 gate가 필요한가

`production continuous loop + cron + optimizer`는 개별 기능보다 결합 위험이 큽니다.

- Cron은 반복 실행, overlap, silent failure, 로그/디스크 누적 위험을 만든다.
- Optimizer는 metric overfit, prompt/tool description drift, Goodhart 리스크를 만든다.
- Continuous loop는 작은 실패를 반복 증폭할 수 있다.
- External PR automation 또는 auto-merge가 결합되면 외부 상태 변경 위험이 급증한다.

따라서 activation은 한 번에 켜지 않고 아래 3단계로 분리합니다.

```text
G1 one-shot canary → G2 bounded N supervised run → G3 limited cron
```

## G0 — Gate document only

**상태:** 이 문서 작성 단계.
**결과:** `DEFINED_NOT_ENABLED`.

허용:

- activation gate JSON/Markdown 작성
- read-only validation
- Obsidian/Discord handoff artifact 작성

금지:

- cron 생성/활성화
- production continuous loop 시작
- optimizer 실행
- paid/network benchmark 실행
- active Hermes runtime/source/config/skill/memory 변경
- external PR automation
- auto-merge/deploy

## G1 — One-shot canary

**목표:** 실제 cron 없이, 수동 invocation으로 production-like pipeline을 정확히 1회만 실행한다.

### Entry requirements

- Sunwoo의 별도 승인: invocation method, timeout, budget, model/provider allowlist, delivery target 명시
- clean worktree 또는 dirty-scope waiver
- concurrency lock path 정의 및 시작 전 lock 부재 확인
- write root 제한:
  - `output/phase5-continuous-loop/production-canary-<stamp>/`
  - `reports/phase5_production_canary_<stamp>.*`
- kill switch command 문서화
- no auto-merge / no deploy / no active runtime mutation

### Bounds

| 항목 | 제한 |
|---|---|
| repeat count | max 1 |
| wall-clock timeout | max 1800s |
| cron | disabled |
| network/model calls | canary approval packet에 명시된 경우만 허용, 기본 false |
| optimizer | 명시 budget 승인 시 candidate-only, 기본 false |
| external PR update | false |
| auto-merge/deploy | false |
| active runtime mutation | false |

### Required outputs

- preflight report
- canary run summary JSON/Markdown
- side-effect ledger
- budget ledger, network/model calls가 승인된 경우
- static/secret scan result
- human review handoff

### 다음 단계로 갈 수 있는 조건

- side-effect flag failure 없음
- raw credential/private-data finding 없음
- 모든 JSON parse 통과
- scheduler decision이 `PASS_NO_ACTION` 또는 action 없는 `REVIEW_REQUIRED`
- optimizer output이 있다면 candidate-only이고 human review 완료
- Sunwoo가 G2를 별도 승인

Fail-closed statuses:

```text
G1_FAIL_BLOCKED
G1_REVIEW_REQUIRED
G1_ABORTED_BY_KILL_SWITCH
```

## G2 — Bounded N supervised run

**목표:** cron 없이 작은 유한 반복을 감독 하에 실행해 repeatability, lock, budget, evidence quality를 검증한다.

### Entry requirements

- G1 packet review accepted
- 별도 승인된 `N`, schedule window, budget cap, supervision owner
- run ledger 및 idempotency key 전략
- overlap lock 테스트
- rollback/checkpoint 절차 문서화

### Bounds

| 항목 | 제한 |
|---|---|
| N default max | 3 |
| N absolute max without new gate | 5 |
| invocation | manual or supervised only |
| cron | disabled |
| per-run timeout | max 1800s |
| total budget cap | required |
| optimizer | candidate-only |
| external PR update | 별도 승인 없으면 manual handoff only |
| auto-merge/deploy | false |
| active runtime mutation | false |

### 다음 단계로 갈 수 있는 조건

- 모든 N회 run이 parseable evidence packet 생성
- overlap/runaway 없음
- 같은 class의 `REVIEW_REQUIRED`가 반복될 경우 fix 또는 waiver 기록
- budget cap 준수
- independent review blocker 없음
- Sunwoo가 limited cron을 별도 승인

Fail-closed statuses:

```text
G2_FAIL_BLOCKED
G2_REVIEW_REQUIRED
G2_STOPPED_SUPERVISOR_DECISION
```

## G3 — Limited cron

**목표:** G1/G2가 안전하게 통과된 뒤, 좁은 범위의 만료되는 cron만 활성화한다.

### Entry requirements

- G2 accepted
- cron expression, timezone, expiry date, max fire count, delivery target 승인
- cron job prompt/script가 self-contained 및 fail-closed
- kill switch 테스트 완료
- alert threshold 및 no-action silence semantics 문서화
- state/watermark storage path 제한

### Bounds

| 항목 | 제한 |
|---|---|
| initial frequency | weekly or slower 권장 |
| initial max fire count | 2 |
| expiry | required |
| concurrency | skip if previous running |
| optimizer | candidate-only |
| external PR update | 승인된 reviewer handoff 범위만 |
| auto-merge/deploy | false |
| active runtime mutation | false |

### 다음 단계로 갈 수 있는 조건

- cron이 승인된 schedule에서만 fire
- duplicate jobs / overlapping runs 없음
- unexpected writes/network calls 없음
- review packet이 honest/fail-closed 유지
- broader production loop 전 manual review에서 value 확인

Fail-closed statuses:

```text
G3_CRON_DISABLED_BY_GUARD
G3_REVIEW_REQUIRED
G3_FAIL_BLOCKED
```

## Global hard blocks

다음 중 하나라도 발생하면 즉시 중단하고 Sunwoo 확인이 필요합니다.

- raw credential 접근/입력/출력
- unbounded recurring execution
- auto-merge 또는 deploy
- dedicated apply gate 없는 active Hermes runtime/config/skill/memory 변경
- optimizer가 production artifact에 직접 write
- metric/scheduler triage가 필요한데 `PASS`로 표기
- 명시 budget 없는 network 또는 paid benchmark call

## Acceptance criteria

| ID | Criterion | Requirement |
|---|---|---|
| P5-PAG-01 | stage separation | G1/G2/G3는 별도 승인 없이는 합칠 수 없다. |
| P5-PAG-02 | bounded execution | 각 stage는 repeat count, timeout, lock, write roots, budget, kill switch를 먼저 정의한다. |
| P5-PAG-03 | candidate-only optimizer | optimizer output은 human review 전 candidate-only로 남는다. |
| P5-PAG-04 | cron expiry and lock | limited cron은 expiry/max-fire count와 skip-if-running lock을 가져야 한다. |
| P5-PAG-05 | reviewable evidence | 모든 run은 JSON/Markdown evidence, side-effect ledger, fail-closed status를 남긴다. |
| P5-PAG-06 | no auto-merge/deploy | human review/merge는 필수이며 deploy는 범위 밖이다. |

## G1 canary approval packet 초안

다음 단계에서 실제 canary를 승인하려면 최소한 아래 항목을 별도 문서/명령으로 확정해야 합니다.

```yaml
stage: G1_ONE_SHOT_CANARY
repeat_count_max: 1
wall_clock_timeout_seconds_max: 1800
cron_enabled: false
optimizer_execution:
  allowed: false  # true로 바꾸려면 candidate-only, budget, model/provider allowlist 필요
network_calls_allowed: false
external_pr_update_allowed: false
auto_merge_allowed: false
active_runtime_mutation_allowed: false
write_roots:
  - output/phase5-continuous-loop/production-canary-<stamp>/
  - reports/phase5_production_canary_<stamp>.*
required_ledgers:
  - side_effect_ledger
  - run_summary
  - secret_static_scan
  - budget_ledger_if_applicable
kill_switch: required_before_run
human_review_required_before_G2: true
```

## 결론

이 문서로 production-loop activation은 다음처럼 분리됩니다.

1. **G1 one-shot canary** — manual, one-shot, bounded, no cron.
2. **G2 bounded N supervised run** — cron 없이 N회 이하 감독 실행.
3. **G3 limited cron** — 만료/횟수/lock/kill switch가 있는 제한 cron.

현재 완료된 것은 **G0 gate definition**입니다. 실제 activation은 아직 발생하지 않았고, 다음 단계는 **G1 one-shot canary approval packet 작성 및 별도 승인**입니다.

## Recommended next step/action

**G1 one-shot canary approval packet**을 다음 artifact로 작성하는 것을 권장합니다. 아직 cron/optimizer/production loop는 켜지지 않았고, 별도 승인 전까지 OFF를 유지해야 합니다.
