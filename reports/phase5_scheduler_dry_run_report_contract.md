# Phase 5 Scheduler Dry-Run Report Contract

Status: implemented as a read-only local no-side-effect scheduler dry-run contract.

Module: `evolution.monitor.scheduler_dry_run`

## Purpose

The Phase 5 scheduler dry-run layer converts a sanitized Phase 5 auto-triage ranking report into hypothetical manual-review scheduling actions. It is intentionally not a cron installer, notification sender, benchmark runner, optimizer trigger, or PR updater.

## Report identity

Required report identity fields:

- `schema_version`: `phase5-scheduler-dry-run-v1`
- `phase`: `5`
- `mode`: `phase5-readonly-scheduler-dry-run`
- `status`: `DRY_RUN_REVIEW_REQUIRED` or `DRY_RUN_NOOP`
- `generated_at`: caller-supplied or current UTC timestamp

## Input contract

The input must be a `phase5-auto-triage-ranking-v1` report produced by the read-only auto-triage layer. It must preserve read-only safety invariants before the scheduler dry-run will run.

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
- `benchmark_cron_enabled=false`
- `scheduler_or_cron_side_effects_performed=false`
- `notifications_sent=false`
- `auto_optimizer_triggered=false`
- `optimizer_execution_started=false`
- `automated_pr_created_or_updated=false`
- `automated_apply_ready=false`

For quick review, the Markdown output must also surface:

- `cron_jobs_created=false`
- `benchmark_cron_enabled=false`
- `scheduler_or_cron_side_effects_performed=false`

## Dry-run policy

Scheduler enablement policy:

`never_enable_in_this_slice`

Real scheduler enablement requires all of the following before any future non-dry-run change:

1. explicit human approval for scheduler enablement
2. Phase 4 formal handoff reviewed or waived
3. benchmark/API budget approval
4. cron target and delivery channel review

Each dry-run action records the target metric, component, priority score, manual-review-only cadence, required approval, and explicit false flags for cron creation, benchmark cron enablement, optimizer start, external notification, and external PR update.

## Local candidate bundle queue

The scheduler dry-run also emits a local candidate bundle queue that maps ranked auto-triage targets to the local bundle contract:

- `local_candidate_bundle_contract.schema_version`: `hse-local-candidate-bundle-v1`
- `decision_json_required_before_apply=true`
- `runner_execution_started=false`
- `active_apply_ready=false`
- `github_publication_performed=false`

Optional `--candidate-bundle-decision-json <path>` inputs let the dry-run consume existing local bundle `decision.json` files. Consumed decisions must be candidate-only, keep `apply_ready=false`, record no GitHub PR/push/merge side effects, and match a ranked queue item by both phase and exact canonical candidate bundle target. Broad metric/component aliases do not count as a match. The dry-run only records whether a decision is already available for a ranked target; it does not start runners, create local bundles, apply candidates, or publish to GitHub.

Queue entries include the target metric/component, mapped candidate bundle phase/target, runner hint, decision state (`DECISION_AVAILABLE` or `MISSING_DECISION`), decision status/run id when available, and `requires_human_review_before_apply=true`.

## Output boundary

The CLI writes only under `output/phase5-continuous-loop/<run-id>/` and produces:

- `scheduler_dry_run_report.json`
- `scheduler_dry_run_report.md`

The output directory must be below the Phase 5 output root and must not already contain stale artifacts.

## Triage semantics

When dry-run actions exist, `status=DRY_RUN_REVIEW_REQUIRED` and the recommended next step is:

`human_review_required_before_scheduler_enablement`

This recommendation is deliberately non-automated. It is evidence for human review only and is not approval to create cron jobs, enable benchmark cron, run benchmark/API spending, start GEPA/DSPy or Darwinian optimizers, send external notifications, update external pull requests automatically, or apply changes to active Hermes runtime state.
