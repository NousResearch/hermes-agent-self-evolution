# HSE Sanitized Publication-Safe Packet

This bundle is a local-only, path-limited, sanitized no-auto-merge publication packet for Hermes Self-Evolution evidence review.

## Status

```text
status: SANITIZED_PUBLICATION_PACKET_CREATED_LOCAL_ONLY
packet: hse_publication_safe_packet_20260705_040031
branch: hse/publication-safe-packet-20260705_040031
base_ref: origin/main
base_commit: 0a929e3aa20e15cf04dc7c28492a7d41a5139125
strict_status: HSE_PROJECT_STRICT_COMPLETE
manifest_status: HSE_PROJECT_STRICT_COMPLETE
highest_strict_complete_phase: 5
```

## Safety / publication boundary

```text
github_write_performed: false
branch_pushed: false
pull_request_created: false
merge_performed: false
auto_merge_performed: false
provider_or_model_spend_performed: false
gateway_restart_or_reload_performed: false
cron_mutation_performed: false
human_review_required_before_publication_or_merge: true
```

This packet is not a merge claim and not a CI-pass claim. If it is later published as a PR, auto-merge must remain disabled and CI/check status must be classified separately.

## PLAN source caveat

Local operational PLAN and remote upstream PLAN diverge. This packet freezes the audit basis to the local operational PLAN SHA plus the user-provided Phase 1-5 table.

```text
local_plan_sha256: edfdf202dd164c9d4618909f18ecff1016dd187d98e3bd46ac784613a6e6b662
remote_plan_sha256: 5928030b470710480a4c0fd7e702af1b2352c9470adddc437c3bc965a23a71f8
remote_equals_local: False
```

## Phase evidence summary

| Phase | Classification | Basis | Caveat |
|---|---|---|---|
| Phase 1 | `PASS_LOCAL_CURRENT_HEAD_STRICT` | latest strict frontier marks phase1 strict_complete=true and current-head manifest is hash-verified | External/upstream publication status is not claimed by this index. |
| Phase 2 | `PASS_LOCAL_CURRENT_HEAD_STRICT` | latest strict frontier marks phase2 strict_complete=true; Phase 2E decision artifacts record holdout/gate/human-review decisions | Candidate/apply boundary remains explicit; active tool schema apply is not implied by candidate-only evidence. |
| Phase 3 | `PASS_LOCAL_CURRENT_HEAD_STRICT_WITH_BOUNDED_LOCAL_EVIDENCE` | latest strict frontier marks phase3 strict_complete=true; local full completion manifest records bounded local active source apply and smoke validation | Real remote/provider-spend benchmark execution remains not claimed unless explicitly approved and run. |
| Phase 4 | `PASS_LOCAL_CURRENT_HEAD_STRICT` | latest strict frontier marks phase4 strict_complete=true; Phase 4 clean-worktree gate records known-bug red→green and freeze-surface preservation | Human review/merge upstream is not claimed here. |
| Phase 5 | `PASS_LOCAL_CURRENT_HEAD_STRICT_PRODUCTION_SURROGATE` | latest strict frontier marks phase5 strict_complete=true; latest production strict loop records unattended detect→optimize→local PR-ready handoff with GitHub write deferred | Local PR-ready handoff is publication-safe evidence; GitHub branch push, PR creation, merge, auto-merge, deploy are not performed. |

## Files

- `README.md` — this packet overview.
- `hse_evidence_reconciliation_index.sanitized.json` — publication-safe evidence index.
- `hse_evidence_reconciliation_index.sanitized.md` — human-readable reconciliation report.
- `audit_refresh_manifest.sanitized.json` — sanitized latest current-head audit refresh manifest.
- `strict_frontier_audit.sanitized.json` — sanitized strict frontier audit.
- `phase5_production_cron_verified_loop_run.sanitized.json` — sanitized production strict loop run evidence.
- `phase5_production_cron_verified_strict_unattended_loop_report.sanitized.json` — sanitized Phase 5 unattended loop report.
- `phase5_production_cron_verified_loop_state.sanitized.json` — sanitized loop state.
- `sanitized_source_manifest.json` — source-to-packet file mapping and hashes.
- `sanitized_publication_packet_summary.json` / `.md` — machine/human summary.
- `sanitized_publication_packet_verification.json` — verification result.
- `PR_BODY_DRAFT.md` — draft PR body for later explicit GitHub-write approval.
