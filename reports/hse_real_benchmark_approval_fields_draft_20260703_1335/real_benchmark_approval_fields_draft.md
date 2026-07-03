# HSE Real Benchmark Approval Fields Draft

Status: `APPROVAL_FIELDS_DRAFT_RECORDED_NOT_EXECUTABLE`

This is not approval to execute. It is a conservative human-review draft only.

## Execution State

- approval_complete=false
- real_benchmark_execution_approved=false
- execution_ready=false
- strict_plan_gate_closed=false
- execution_started=false
- real_benchmarks_executed=false
- NO_GITHUB_WRITE

## Conservative Draft Fields

- `benchmark_suites` conservative default: `["TBLite", "YC-Bench", "Phase2 PLAN-scale tool-selection triples"]`
- `max_budget_usd_or_krw` conservative default: `{"max_budget_krw": 0, "max_budget_usd": 0}`
- `max_runtime_minutes` conservative default: `0`
- `network_provider_api_spend_allowed` conservative default: `false`
- `baseline_materialization_allowed` conservative default: `false`
- `current_materialization_allowed` conservative default: `false`
- `regression_thresholds` conservative default: `{"Phase2 PLAN-scale tool-selection triples": "no_aggregate_or_per_tool_regression_beyond_gate", "TBLite": "within_2_percent_or_better", "YC-Bench": "no_material_regression"}`
- `allowed_write_roots` conservative default: `[]`
- `rollback_plan` conservative default: `null`
- `human_approval_source` conservative default: `null`

## Still Blocked By

- awaiting_explicit_human_benchmark_approval
- max_budget_usd_or_krw
- max_runtime_minutes
- network_provider_api_spend_allowed
- baseline_materialization_allowed
- current_materialization_allowed
- human_approval_source
- allowed_write_roots
- rollback_plan
