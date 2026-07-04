# HSE Operational Completion Contract Preflight

## Conclusion

This contract freezes HSE strict operational completion semantics as local-only preflight evidence. It does **not** perform active apply, GitHub query/write, cron/gateway mutation, deploy/publication, provider/API spend, or overall HSE completion claim.

```text
status=HSE_OPERATIONAL_COMPLETION_CONTRACT_PREFLIGHT_READY_LOCAL_ONLY
strict_current_frontier=Phase 3
overall_hse_completion_claimed=false
phase4_strict_complete=false
phase5_strict_complete=false
```

## Completion Model

- `local_hse_repo_completion`: HSE repo artifacts/source/tests prove phase gates inside the self-evolution repository.
- `active_hermes_evah_operational_completion`: Current active Hermes/EvAH runtime/source/profile/skills/tool schemas/prompt/cron state match the evidence subject and pass current-active gates.
- `strict_overall_hse_completion`: Only true when Phase 1-5 all strict-complete under current active baseline and no forbidden shortcut is used.

## Forbidden Shortcuts

- candidate-only output counted as active apply
- local smoke-only benchmark counted as full benchmark unless explicitly approved as strict-equivalent
- waiver-only completion counted as strict completion
- read-only/no-op cron counted as unattended optimize-to-PR loop
- archive/claim artifact counted as active runtime completion
- historical completion counted as current-active completion after active baseline drift
- provider/API budget or network benchmark run without explicit bounded approval
- GitHub publication/query/write without explicit packet approval
- cron/gateway mutation without explicit packet approval
- overall HSE completion claim before Phase 1-5 strict gates pass

## Allowed Evidence Classes

- hash-backed artifact manifests and evidence JSON
- current git branch/head/clean status snapshots
- current-active source/skill/schema/prompt readback with SHA-256
- baseline-vs-candidate-vs-active deterministic scoring tables
- targeted and full test output with return codes
- benchmark logs, command argv, elapsed time, budget ledger, and no-regression summary
- Time Rewind anchor/inspect/release evidence for local mutations
- manual human review acceptance for active apply/PR/merge boundaries
- cron/process/job status evidence only when the task is explicitly about automation state

## Phase Exit Criteria

### phase1_skill_evolution

- strict_goal: At least one skill measurably improved, no benchmark regression, reusable skill-evolution pipeline present.
- current_contract_status: `requires_current_active_revalidation_before_operational_claim`

Minimum exit checks:
- selected active skill baseline/candidate/current SHA fixed
- holdout score delta >= 10% or preserved applied improvement explicitly proven
- human-sensible diff review pass
- active default-profile skill apply or current-active match verified when claiming operational completion
- benchmark regression=false under approved strict/strict-equivalent gate

### phase2_tool_descriptions

- strict_goal: Tool-selection accuracy improved, no benchmark regression, active schema/source state matches claim when operational completion is claimed.
- current_contract_status: `candidate_only_complete_active_apply_missing`

Minimum exit checks:
- 45-case default gate pass
- 9-case or newer holdout pass
- accuracy improvement >= 5%
- per-tool regression count = 0
- description <= 500 chars and parameter description <= 200 chars
- active Hermes schema/source apply or current-active match verified
- benchmark regression=false under approved gate

### phase3_system_prompt

- strict_goal: Behavioral tests pass, benchmarks hold or improve, current active prompt/source matches claim subject.
- current_contract_status: `local_phase3_strict_claim_complete_but_current_active_revalidation_recommended_before_overall_claim`

Minimum exit checks:
- behavioral tests pass for identity/tool discipline/memory/session-search/skill-loading/privacy posture
- prompt caching boundary preserved
- current active Hermes HEAD matched against Phase 3 claim subject
- active source apply or no-op current-active match verified
- benchmark regression=false under approved gate
- claim/archive/extract evidence remains hash-valid

### phase4_code_evolution

- strict_goal: At least one known bug fixed by evolution; tests pass; benchmarks hold; signatures/registries stable; human review pass.
- current_contract_status: `not_strict_complete`

Minimum exit checks:
- known active bug reproduced with RED test
- Darwinian Evolver or approved code-evolution engine invoked for the fix
- candidate-only mutation bounded before active apply
- GREEN target/module/full tests pass
- TBLite + TerminalBench2/approved substitute + YC-Bench hold or improve
- function signatures and registry.register surfaces unchanged unless explicitly approved
- human review acceptance recorded
- active apply/PR/merge only after separate approval

### phase5_continuous_loop

- strict_goal: Automated pipeline runs unattended: weekly benchmarks, auto-triage, at least one detect->optimize->PR cycle, human merge retained.
- current_contract_status: `not_strict_complete`

Minimum exit checks:
- weekly or approved bounded benchmark job runs unattended with lock/expiry/budget/kill switch
- auto-triage identifies underperforming target from sanitized metrics
- optimizer actually runs candidate-only without manual intervention after trigger
- PR or PR-ready reviewer handoff produced by the unattended cycle
- human merge boundary preserved and auto-merge/deploy=false
- side-effect, budget, secret, and private-data ledgers pass
- cron/job status proves bounded repeated operation, not only a read-only/no-op run

## Rollback and Abort Gates

- `before_local_mutation`: create Time Rewind anchor, record scope, verify clean repos
- `before_active_apply`: backup/checksum target files, dry-run apply, explicit approval, rollback branch or restore plan
- `before_benchmark_or_provider_spend`: explicit budget/runtime/provider approval and abort threshold
- `before_github_query_or_write`: explicit GitHub packet; no automatic PR/merge/deploy by default
- `before_cron_or_gateway_mutation`: explicit schedule/expiry/kill-switch packet and post-change status verification
### abort_if
- repo unexpectedly dirty outside approved paths
- active Hermes baseline no longer matches evidence subject and no revalidation exists
- benchmark regression appears
- secret/private-data scan fails
- provider/API budget unavailable or exceeds cap
- side-effect ledger records unapproved mutation
- Phase criteria would need weakening to pass

## Execution Order Contract

- P0 freeze this operational completion contract
- P1 Phase 1 current-active skill evolution revalidation/apply closure
- P2 Phase 2 active tool-description revalidation/apply closure
- P3 Phase 3 current active prompt revalidation/apply closure
- P4 Phase 4 true code-evolution completion with full gates
- P5 Phase 5 strict unattended detect->optimize->PR loop completion
- P6 only then issue overall HSE completion claim

## Safety Flags

```json
{
  "active_apply_performed": false,
  "active_hermes_repo_modified": false,
  "cron_mutation_performed": false,
  "deploy_or_publication_performed": false,
  "gateway_mutation_performed": false,
  "github_query_performed": false,
  "github_write_performed": false,
  "overall_hse_completion_claimed": false,
  "provider_or_api_spend_performed": false
}
```

## Source Artifacts

- `audit_plan`: `/Users/snw/Library/Mobile Documents/iCloud~md~obsidian/Documents/SnwEvAH/Response/20_Hermes_HSE_and_Ouroboros/2026-07-05-0009-hse-strict-operational-audit-and-completion-plan.md` sha256=`d057955b3c95ef685ecffbbee1d3c935cdad799a9ffd6953336d25f301eff4da` bytes=`19266`
- `phase3_claim_emit_manifest`: `reports/hse_phase3_official_completion_claim_emit_20260704_2305_local_only/phase3_official_completion_claim_emit_manifest.json` sha256=`743938166ebf58d2dab026d734b6bb2757f8b61d10f9a8cf8c201cdbc7d10644` bytes=`14776`
- `phase3_archive_extract_verify_manifest`: `reports/hse_phase3_claim_local_archive_disposable_extract_verify_20260704_2352_local_only/phase3_claim_local_archive_disposable_extract_verify_manifest.json` sha256=`db57c86ee0849a2b90a022ed0985da3e910829c412c493e38ce358e7acf24127` bytes=`9644`
- `phase5_readiness_manifest`: `reports/phase5_continuous_loop_readiness_manifest.json` sha256=`0217e9eb69783ee8321f2cd6431cb4f15fbf5f61c394a2a9312aca3d2d3fca4a` bytes=`4798`
- `phase5_formal_completion_with_waiver`: `reports/phase5_formal_completion_20260607_091723.json` sha256=`5f41178be7d48ae4c481e20156ea0500cd71c7d35f03ed91bde480e5a7ee8b9a` bytes=`4920`

## Next Exact Packet

`hse_phase1_current_active_revalidation_preflight_go_no_active_apply_no_github_write_no_cron_mutation_no_provider_spend`

## Verification Contract

- JSON must parse.
- Contract must include all five phase exit criteria.
- Forbidden safety flags must remain false.
- HSE repo mutation must be limited to this report directory until local commit.
- Active Hermes repo must remain clean and unmodified.
