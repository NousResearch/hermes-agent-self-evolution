# Phase 5 Supervised Candidate Runner Contract

Status: implemented as an explicit-approval local handoff from scheduler dry-run queue evidence to one local candidate runner decision.

Module: `evolution.monitor.supervised_candidate_runner`

## Purpose

The Phase 5 supervised candidate runner closes the local evidence loop:

```text
scheduler dry-run queue -> explicit single-target approval -> existing candidate bundle decision.json -> re-consumed report
```

It is not an unattended scheduler and not a production apply step. It records only one approved queue target per run. The approval phrase is a public sentinel for human/operator intent, not a credential or secret.

## Report identity

Required report identity fields:

- `schema_version`: `phase5-supervised-candidate-runner-v1`
- `phase`: `5`
- `mode`: `phase5-supervised-local-candidate-runner`
- `status`: `SUPERVISED_DECISION_RECONSUMED`
- `generated_at`: caller-supplied or current UTC timestamp

## Approval contract

The CLI requires all of the following:

- `--scheduler-report-json <path>` from the read-only scheduler dry-run layer
- `--approved-queue-id <queue-id>` selecting exactly one queue target
- `--approval-token APPROVE_LOCAL_CANDIDATE_RUNNER`
- `--candidate-bundle-decision-json <path>` produced by or supplied for the approved local runner
- `--output-dir output/phase5-continuous-loop/<run-id>`

The approval scope is exactly:

`single_local_candidate_queue_target_only`

The approval is not approval for active runtime apply, GitHub publication, deployment, cron creation, external notification, or multi-target unattended execution.

## Inline execution policy

The tool only re-consumes an existing `decision.json`.

Inline command execution is deliberately disabled in this safety slice. If `--execute-approved-runner` or `--runner-command-json` is supplied, the CLI must fail before writing a report or running any command. Operators who need to create a new local bundle must run the approved local runner separately under its own review/approval step, then pass the resulting `decision.json` back to this tool.

The report records:

- `execution_mode=manual_decision_reconsume_only`
- `runner_execution_started=false`
- `runner_returncode=null`
- `raw_command_recorded=false`

## Candidate bundle decision contract

The consumed `decision.json` must use the local candidate bundle contract:

- `schema_version=hse-local-candidate-bundle-v1`
- `candidate_only=true`
- `apply_ready=false`
- GitHub PR/push/merge booleans are false
- active runtime/skill/tool schema/prompt mutation booleans are false
- credentials, external publication, and deployment booleans are false

The decision must match the approved queue target by both phase and exact canonical `candidate_bundle_target`. Same-phase evidence for a broad component/metric alias or unrelated target must remain `MISSING_DECISION` / rejected.

## Safety invariants

The report must keep these safety invariants:

- `active_runtime_mutation=false`
- `active_apply_ready=false`
- `credentials_accessed=false`
- `cron_jobs_created=false`
- `scheduler_or_cron_side_effects_performed=false`
- `external_calls_performed=false`
- `network_calls_performed=false`
- `github_publication_performed=false`
- `automated_pr_created_or_updated=false`
- `deployment_performed=false`

For quick review, the Markdown output must also surface:

- `active_apply_ready=false`
- `github_publication_performed=false`
- `automated_pr_created_or_updated=false`
- `cron_jobs_created=false`
- `deployment_performed=false`

## Output boundary

The CLI writes only under `output/phase5-continuous-loop/<run-id>/` and produces:

- `supervised_candidate_runner_report.json`
- `supervised_candidate_runner_report.md`

The output directory must be below the Phase 5 output root and must not already contain stale artifacts.
