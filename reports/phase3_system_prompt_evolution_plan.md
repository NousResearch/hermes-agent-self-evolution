# Phase 3 System Prompt Evolution Design Plan

Status: planned, not executed

Design-only boundary: no active system-prompt mutation or apply.

## Goal

Design Phase 3 system prompt evolution scope, acceptance criteria, and benchmark gate reactivation before any execution or active apply.

## Scope

Phase 3 may later design section-as-DSPy-parameter wrappers for these five evolvable prompt sections:

- `DEFAULT_AGENT_IDENTITY`
- `MEMORY_GUIDANCE`
- `SESSION_SEARCH_GUIDANCE`
- `SKILLS_GUIDANCE`
- `PLATFORM_HINTS`

The following remain non-evolvable in this Phase 3 design boundary:

- durable user/private memory
- auto-generated skills index
- project context files such as AGENTS.md / CLAUDE.md / .cursorrules
- canonical identity or SOUL artifacts
- secrets, credentials, and active runtime configuration

## Acceptance criteria

1. Phase 3 design starts only after the Phase 2E human review checkpoint records Phase 2 closeout as complete.
2. This artifact is design-only: no runtime mutation, no GEPA run, no active prompt/source apply.
3. The five evolvable sections are explicitly scoped.
4. Private/generated/context/identity artifacts are explicitly excluded.
5. The future behavioral evaluator covers tool-use discipline, memory/session-search behavior, skill loading, platform formatting, identity/safety, and prompt-cache compatibility.
6. Benchmark gate reactivation is required before Phase 3 execution, system-prompt evolution acceptance, active system-prompt apply, or default-gate promotion.
7. Any active apply requires a separate human-approved PR or patch.
8. Prompt-cache boundaries and identity/safety constraints are preserved; drift is a reject condition.

## Benchmark gate reactivation

Benchmark gate reactivation is required before Phase 3 execution.

TBLite and YC-Bench are not blocking this design-only plan. They become blocking before:

- Phase 3 execution
- system prompt evolution acceptance
- active system prompt apply
- default gate promotion

This follows `reports/phase2e_benchmark_gate_decision.json`, which deferred benchmark execution for Phase 2 candidate-only closeout but kept it as a prerequisite before broader system-prompt/runtime changes.

## Seed

The design-only Seed is recorded at:

- `seeds/phase3_system_prompt_evolution_seed.yaml`

This Seed is intentionally not an execution/apply Seed. It fixes scope, acceptance criteria, constraints, and reactivation boundaries for a future Phase 3 execution plan.

## Safety boundary

- `plan_only`: yes
- `execution_started`: no
- `active_system_prompt_mutation`: no
- `active_system_prompt_apply_approved`: no
- `raw_private_session_data_committed`: no
- `secrets_or_credentials_required`: no
