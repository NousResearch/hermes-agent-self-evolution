# Phase 3 Real Benchmark Readiness Manifest

Status: recorded, not executed.

This manifest records the non-executing readiness contract for switching Phase 3 from local dry-run fixtures to real TBLite/YC-Bench evidence. It does not run GEPA/DSPy, does not run real benchmarks, does not edit Hermes Agent prompt source, and does not apply any evolved prompt to the active runtime.

- Machine-readable manifest: `reports/phase3_real_benchmark_readiness_manifest.json`
- Source execution Seed draft: `seeds/phase3_system_prompt_evolution_execution_seed_draft.yaml`
- Source execution report: `reports/phase3_execution_seed_draft.json`
- Planning authorization: `rec action GO`

## Current readiness state

- `real_benchmark_ready_now=false`
- `active_apply_ready_now=false`
- `phase3_execution_ready=false`
- Current authorized spend is `$0`; current authorized spend is `$0` until separate approval records a nonzero benchmark/API budget.

## Required inputs before real benchmark execution

Real benchmark execution remains blocked until the future run has all required prompt artifacts, candidate scaffold evidence, local dry-run preflight evidence, benchmark runner readiness, and fixture/real-case evidence under the Phase 3 output root.

Required input classes:

1. baseline prompt artifact;
2. candidate prompt artifact;
3. candidate scaffold report;
4. dry-run preflight report;
5. TBLite fixture or real-case evidence;
6. YC-Bench fixture or real-case evidence;
7. TBLite runner implementation;
8. YC-Bench runner implementation.

## Real benchmark transition

The current benchmark adapters are still `dry_run_only_real_mode_not_implemented`. Real-mode command templates are recorded for review only and are not runnable now. Network calls, external benchmark calls, and nonzero spend remain disallowed until separate approval.

Real benchmark results are required before:

- Phase 3 execution;
- system prompt evolution acceptance;
- active system prompt apply;
- default-gate promotion.

## Approval, cost, and runtime gate

Separate approval is required before running real TBLite commands, running real YC-Bench commands, spending any nonzero benchmark/API budget, running GEPA/DSPy optimization, editing Hermes Agent prompt source, applying an evolved prompt to active runtime, or promoting a default gate.

The proposed first real-benchmark budget cap is `$25`, the hard stop is `$50`, and maximum wall clock is `8` hours. Any higher budget requires reapproval.

## Rollback and write boundary

Before real benchmark execution, the future run must record a checkpoint, baseline prompt snapshot, baseline prompt checksum, git status snapshot, and rollback handle. Before active apply, active runtime rollback coverage must also be available.

Allowed writes before active apply are restricted to the Phase 3 output root and the readiness manifest artifacts. Writes to canonical identity artifacts, Hermes config, active prompt-builder source, skills, memories, and profile directories remain prohibited without separate apply approval.

## Go/no-go conditions

The manifest fails closed until all go/no-go conditions are satisfied:

- `RBM-1-input-artifacts-and-checksums`
- `RBM-2-real-mode-runner-implemented-or-pinned`
- `RBM-3-explicit-approval-and-budget`
- `RBM-4-fresh-output-root`
- `RBM-5-rollback-handle`
- `RBM-6-candidate-only-no-apply`

The current state is intentionally blocked: `real_benchmark_ready_now=false` and `active_apply_ready_now=false`.
