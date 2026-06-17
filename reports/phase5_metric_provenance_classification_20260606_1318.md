---
title: HSE Phase 5 Metric Provenance Classification
created: 2026-06-06-1318
tags: [HSE, Phase5, EvAH, provenance, triage]
---

# HSE Phase 5 Metric Provenance Classification

## 결론

- **완료:** Phase 5 dry-run이 지목한 2개 metric에 대해 provenance tracing 및 sanitized classification artifact를 작성했다.
- **핵심 판정:** 두 metric의 dry-run 값은 현재 **sanitized local fixture aggregate** 이며, 그대로 “실측 per-example/per-warning evidence”로 간주하면 안 된다.
- **tool_selection_accuracy:** 정확히 `sample_count=90`에 대응하는 per-example row는 발견하지 못했다. 다만 가장 가까운 실제 case-level evidence로 `45 cases / 40 pass / 5 fail / accuracy 0.8889` 보고서를 확인했고, 5개 failure를 분류했다.
- **prompt_contract_warning_rate:** 정확한 Phase 3 prompt-contract per-warning row는 발견하지 못했다. warning-like Phase 3 artifact는 manifest/risk note 성격이고, tool-description warning analog는 component mismatch다.
- **안전 상태:** cron/scheduler/optimizer/network/PR update/runtime mutation은 모두 OFF로 유지했다.

## Source evidence

- `dry_run_metrics_input`: `output/phase5-continuous-loop/unattended-dry-run-20260606-124310/sanitized_metrics_input.json`
- `performance_snapshot`: `output/phase5-continuous-loop/unattended-dry-run-20260606-124310/performance/performance_snapshot_report.json`
- `auto_triage`: `output/phase5-continuous-loop/unattended-dry-run-20260606-124310/auto-triage/auto_triage_report.json`
- `scheduler_dry_run`: `output/phase5-continuous-loop/unattended-dry-run-20260606-124310/scheduler/scheduler_dry_run_report.json`
- `manual_triage_plan`: `reports/phase5_manual_triage_plan_20260606_1251.json`
- `nearest_tool_selection_per_example_report`: `output/tool-description/phase2e-heldout-review/run/candidate_only_report.json`
- `heldout_review_summary`: `output/tool-description/phase2e-heldout-review/heldout_review.json`
- `tool_description_warning_analog_report`: `output/tool-description/phase2d_gate_final/candidate_only_report.json`

## Metric classification

### `tool_selection_accuracy`

- Component: `tool_descriptions`
- Dry-run aggregate: value `0.86`, threshold `0.9`, baseline `0.88`, sample_count `90`, status `FAIL`
- Provenance: `sanitized_local_fixture_aggregate`
- Exact 90-row per-example evidence found: `False`
- Nearest concrete evidence: `output/tool-description/phase2e-heldout-review/run/candidate_only_report.json`
- Nearest case-level result: `40/45` pass, `5` fail, accuracy `0.8889`
- Failure class counts: `{"insufficient_discrimination_margin": 2, "wrong_tool_selected": 3}`

| Case | Category | Expected | Selected | Class | Margin | Cue | Sanitized request |
|---:|---|---|---|---|---:|---:|---|
| 2 | `search-vs-shell` | `search_files` | `search_files` | `insufficient_discrimination_margin` | `0.0` | `1.0` | Find Python files mentioning browser_navigate in the tools directory. |
| 3 | `shell-execution` | `terminal` | `terminal` | `insufficient_discrimination_margin` | `0.0` | `1.0` | Run the focused pytest target for the new tool evaluation tests. |
| 4 | `edit-vs-overwrite` | `patch` | `write_file` | `wrong_tool_selected` | `-0.1375` | `1.0` | Make a targeted replacement in one Python file and preserve surrounding content. |
| 16 | `file-read-range-vs-shell` | `read_file` | `search_files` | `wrong_tool_selected` | `-0.005` | `1.0` | Read only lines 120 through 180 of gateway/run.py. |
| 28 | `terminal-package-manager-vs-python` | `terminal` | `execute_code` | `wrong_tool_selected` | `-0.0445` | `0.6` | Install the project dependencies and run the npm build script. |

### `prompt_contract_warning_rate`

- Component: `system_prompts`
- Dry-run aggregate: value `0.07`, threshold `0.05`, baseline `0.06`, sample_count `20`, status `FAIL`
- Provenance: `sanitized_local_fixture_aggregate`
- Exact per-warning evidence found: `False`
- Phase3 warning-like artifact count: `4`; these are manifest/risk-note artifacts, not per-warning rows.
- Tool-description warning analog count: `43` with type counts `{"Missing confusing tool candidate": 1, "Parameter description length constraint failed": 42}`
- Actionable gap: create sanitized prompt-contract warning rows with check_id, prompt_clause, warning_type, and pass/fail outcome

## Safety invariants

- `read_only`: `true`
- `raw_private_session_data_committed`: `false`
- `raw_credentials_recorded`: `false`
- `active_runtime_mutation`: `false`
- `external_calls_performed`: `false`
- `network_calls_performed`: `false`
- `cron_jobs_created`: `false`
- `benchmark_cron_enabled`: `false`
- `scheduler_or_cron_side_effects_performed`: `false`
- `optimizer_execution_started`: `false`
- `automated_pr_created_or_updated`: `false`

## Recommended next step/action

replace synthetic fixture metrics with provenance-backed sanitized per-example/per-warning input before any real scheduler enablement or optimizer run

Scheduler/cron enablement, optimizer execution, network/API benchmark execution, external PR updates, and active runtime mutation still require separate explicit approval.
