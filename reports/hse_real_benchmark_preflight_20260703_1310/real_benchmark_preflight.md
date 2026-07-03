# HSE Real Benchmark Preflight

Status: `PREFLIGHT_RECORDED_NOT_EXECUTABLE`

This preflight is not approval to execute. It records a dry-run-only command preview and guarded plans.

## Execution State

- preflight_passed=true
- execution_ready=false
- dry_run_only=true
- execution_started=false
- real_benchmarks_executed=false

## Suites

- TBLite
- YC-Bench
- Phase2 PLAN-scale tool-selection triples

## Blockers

- awaiting_explicit_human_benchmark_approval
- max_budget_usd_or_krw
- max_runtime_minutes
- network_provider_api_spend_allowed
- baseline_materialization_allowed
- current_materialization_allowed
- human_approval_source
- allowed_write_roots
- rollback_plan

## Boundaries

- NO_GITHUB_WRITE
- benchmark process started: false
- provider/model/API spend performed: false
- worktree materialization performed: false
- gateway restart/reload performed: false
