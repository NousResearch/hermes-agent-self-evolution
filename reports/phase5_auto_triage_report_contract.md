# Phase 5 Auto-Triage Report Contract

Status: implemented as a read-only local ranking contract.

Module: `evolution.monitor.auto_triage`

## Purpose

The Phase 5 auto-triage layer converts a sanitized Phase 5 performance monitor snapshot into a ranked manual-review target list. It is intentionally not an optimizer trigger and not a scheduler.

## Report identity

Required report identity fields:

- `schema_version`: `phase5-auto-triage-ranking-v1`
- `phase`: `5`
- `mode`: `phase5-readonly-auto-triage-ranking`
- `status`: `REVIEW_REQUIRED` or `NO_ACTION`
- `generated_at`: caller-supplied or current UTC timestamp

## Input contract

The input must be a `phase5-performance-snapshot-v1` report produced by the read-only performance monitor. It must preserve read-only safety invariants before auto-triage will run.

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
- `scheduler_or_cron_side_effects_performed=false`
- `auto_optimizer_triggered=false`
- `optimizer_execution_started=false`
- `automated_pr_created_or_updated=false`
- `automated_apply_ready=false`

For quick review, the Markdown output must also surface:

- `auto_optimizer_triggered=false`
- `scheduler_or_cron_side_effects_performed=false`

## Scoring contract

Ranked targets are produced from metrics that fail their threshold or regress against baseline.

Priority formula:

`severity * sample_count`

Tie-breakers:

1. `priority_score desc`
2. `sample_count desc`
3. `metric_id asc`

Each ranked target records rank, metric id, component, status, severity, sample count, baseline regression flag, priority score, reasons, and recommendation.

## Output boundary

The CLI writes only under `output/phase5-continuous-loop/<run-id>/` and produces:

- `auto_triage_report.json`
- `auto_triage_report.md`

The output directory must be below the Phase 5 output root and must not already contain stale artifacts.

## Triage semantics

When ranked targets exist, `status=REVIEW_REQUIRED` and the recommended next step is:

`manual_review_required_no_optimizer_started`

This recommendation is deliberately non-automated. It is evidence for human review only and is not approval to create cron jobs, run benchmark/API spending, start GEPA/DSPy or Darwinian optimizers, update external pull requests automatically, or apply changes to active Hermes runtime state.
