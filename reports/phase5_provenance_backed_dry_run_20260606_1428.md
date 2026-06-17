---
title: HSE Phase 5 Provenance-backed Read-only Dry-run
date: 2026-06-06 14:28 CEST
status: REVIEW_REQUIRED
phase: 5
---

# HSE Phase 5 Provenance-backed Read-only Dry-run

## 결론

REVIEW_REQUIRED. synthetic fixture metric을 실제 provenance-backed sanitized dataset으로 대체하는 read-only generator를 작성했고, 그 입력으로 Phase 5 dry-run을 1회 다시 실행했습니다. 안전 invariant는 통과했지만, `tool_selection_accuracy=0.8889 < 0.9`, `performance_status=NEEDS_TRIAGE`, `auto_triage_status=REVIEW_REQUIRED`, `scheduler_status=DRY_RUN_REVIEW_REQUIRED`였으므로 이것은 gate-passing dry-run으로 해석하면 안 됩니다.

단, 이것은 **Phase 5 formal completion이 아닙니다**. scheduler/cron/optimizer/외부 PR 업데이트는 계속 OFF입니다.

## 구현 변경

- `evolution/monitor/provenance_dataset.py`
  - `tool_selection_accuracy` per-example rows 생성
  - `prompt_contract_warning_rate` per-contract-check rows 생성
  - `provenance_metrics_input.json` 생성
- `evolution/monitor/performance_snapshot.py`
  - `source.kind=provenance_backed_sanitized_dataset` 입력 허용
- `tests/monitor/test_phase5_provenance_dataset.py`
  - RED→GREEN 계약 테스트 추가

## Dry-run output

- output root: `output/phase5-continuous-loop/provenance-backed-dry-run-20260606-122725`
- metrics source kind: `provenance_backed_sanitized_dataset`
- performance status: `NEEDS_TRIAGE`
- auto triage status: `REVIEW_REQUIRED`
- scheduler status: `DRY_RUN_REVIEW_REQUIRED`
- scheduler dry-run actions: `1`
- scheduler side effects: `0`

## Metrics

### tool_selection_accuracy

- value: `0.8889`
- threshold: `0.9`
- baseline: `0.1111`
- sample_count: `45`
- row evidence: `40 pass / 5 fail / 45 total`

### prompt_contract_warning_rate

- value: `0.0`
- threshold: `0.05`
- baseline: `0.0`
- sample_count: `20`
- row evidence: `0 warnings / 20 checks`

## 검증

- RED observed: `5 failed` due missing `evolution.monitor.provenance_dataset`
- focused tests: `11 passed in 0.23s`
- monitor tests: `31 passed in 0.45s`
- full tests: `356 passed, 11 warnings in 6.06s`
- compileall: PASS
- json semantics + secret scan: PASS
- git diff check: PASS

## Safety invariants

- cron_jobs_created=false
- benchmark_cron_enabled=false
- optimizer_execution_started=false
- automated_pr_created_or_updated=false
- active_runtime_mutation=false
- external_calls_performed=false
- network_calls_performed=false

## Recommended next step/action

Review the single remaining `tool_selection_accuracy` dry-run target manually. Do **not** enable scheduler/cron/optimizer until separate approval and Phase 5 go/no-go gates are satisfied.
