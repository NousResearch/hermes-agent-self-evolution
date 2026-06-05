# Phase 5 Performance Monitor Report Contract

Status: implemented as a read-only local snapshot contract.

Module: `evolution.monitor.performance_snapshot`

## Purpose

The Phase 5 performance monitor converts sanitized aggregate metrics into a machine-readable snapshot report for manual triage. It is the first implementation slice of the continuous-loop phase, but it does not enable the continuous loop.

## Report identity

Required report identity fields:

- `schema_version`: `phase5-performance-snapshot-v1`
- `phase`: `5`
- `mode`: `phase5-readonly-performance-monitor-snapshot`
- `status`: `PASS` or `NEEDS_TRIAGE`
- `generated_at`: caller-supplied or current UTC timestamp

## Input contract

The input JSON must use `schema_version=phase5-performance-input-v1` and contain only sanitized aggregate metrics:

- `window.start`
- `window.end`
- `source.kind=sanitized_local_fixture`
- `source.label`
- non-empty `metrics[]`

Each metric requires:

- `id`
- `component`
- finite numeric `value`
- finite numeric `threshold`
- finite numeric `baseline`
- boolean `higher_is_better`
- positive integer `sample_count`

No raw session data, local private paths, or credentials are allowed in inputs or emitted reports.

## Safety invariants

The report must keep these safety invariants:

- `read_only=true`
- `raw_private_session_data_committed=false`
- `raw_credentials_recorded=false`
- `active_runtime_mutation=false`
- `external_calls_performed=false`
- `network_calls_performed=false`
- `cron_jobs_created=false`
- `optimizer_execution_started=false`
- `automated_pr_created_or_updated=false`

For quick review, the Markdown output must also surface:

- `cron_jobs_created=false`
- `optimizer_execution_started=false`

## Output boundary

The CLI writes only under `output/phase5-continuous-loop/<run-id>/` and produces:

- `performance_snapshot_report.json`
- `performance_snapshot_report.md`

The output directory must be below the Phase 5 output root and must not already contain stale artifacts.

## Triage semantics

Metrics are normalized to `PASS` or `FAIL` against their threshold. Baseline regression is also recorded. Any failing or regressing metric becomes a weak area with recommendation:

`manual_triage_required_no_optimizer_started`

This recommendation is deliberately non-automated. It is evidence for human review only and is not approval to create cron jobs, run benchmark/API spending, start GEPA/DSPy or Darwinian optimizers, or update external pull requests automatically.
