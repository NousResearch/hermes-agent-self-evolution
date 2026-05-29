# Phase 3 Execution Seed Draft

Status: drafted, not executed.

This artifact records the future Phase 3 system prompt evolution execution contract. It is a preparation artifact only: it does not run GEPA/DSPy, does not run real benchmarks, does not edit Hermes Agent prompt source, and does not apply any evolved prompt to the active runtime. It does include read-only dry-run fixture adapters for validating the benchmark command/report contract.

- Execution Seed draft: `seeds/phase3_system_prompt_evolution_execution_seed_draft.yaml`
- Machine-readable report: `reports/phase3_execution_seed_draft.json`
- Source design plan: `reports/phase3_system_prompt_evolution_plan.json`
- Source design Seed: `seeds/phase3_system_prompt_evolution_seed.yaml`
- Phase 2E closeout dependency: `reports/phase2e_human_review_checkpoint.json`
- Authorization for drafting: `rec action GO`

## Non-execution boundary

GEPA/DSPy execution: not started.

The execution draft keeps these safety flags false:

- `execution_started: false`
- `run_gepa_now: false`
- `run_dspy_now: false`
- `mutate_active_system_prompt_now: false`
- `active_system_prompt_apply_approved: false`
- `apply_ready: false`

Planning authorization does not approve execution. Separate human approval is still required before optimizer execution, benchmark execution, prompt-source edits, active runtime apply, or default-gate promotion.

## Future execution scope

The future execution mode is candidate-only system prompt evolution. The allowed output is a candidate report and review packet under:

```text
output/phase3-system-prompt/<run-id>/
```

Expected future artifacts include:

- `baseline_system_prompt.json`
- `candidate_system_prompt.json`
- `candidate_only_report.json`
- benchmark result JSON files under `benchmarks/`

No active prompt/source apply is included in this draft.

## Benchmark gate

Benchmark commands are contract templates and have not been run against real Phase 3 artifacts. Their read-only dry-run fixture adapters are now runnable and covered by `tests/tools/test_phase3_benchmark_adapters.py`.

The benchmark gate is required before Phase 3 execution and remains blocking for:

- Phase 3 execution
- system prompt evolution acceptance
- active system prompt apply
- default-gate promotion

Draft command templates:

```bash
python -m evolution.benchmarks.run_tblite \
  --baseline-prompt output/phase3-system-prompt/<run-id>/baseline_system_prompt.json \
  --candidate-prompt output/phase3-system-prompt/<run-id>/candidate_system_prompt.json \
  --fixtures-jsonl datasets/golden/benchmarks/phase3-system-prompt/tblite_cases.jsonl \
  --output-json output/phase3-system-prompt/<run-id>/benchmarks/tblite.json \
  --dry-run
```

```bash
python -m evolution.benchmarks.run_yc_bench \
  --baseline-prompt output/phase3-system-prompt/<run-id>/baseline_system_prompt.json \
  --candidate-prompt output/phase3-system-prompt/<run-id>/candidate_system_prompt.json \
  --fixtures-jsonl datasets/golden/benchmarks/phase3-system-prompt/yc_bench_fast_test.jsonl \
  --preset fast_test \
  --output-json output/phase3-system-prompt/<run-id>/benchmarks/yc_bench.json \
  --dry-run
```

The fixed adapter contract requires `mode=dry-run-fixture`, `candidate_only=true`, `read_only=true`, `external_calls_performed=false`, and `apply_ready=false` in each output report. Real benchmark execution remains deferred until separate human approval.

## Rollback boundary

Rollback boundary requirements:

- Create a checkpoint before candidate generation.
- Snapshot the baseline system prompt before candidate generation.
- Record a checksum for the baseline prompt snapshot.
- Write candidate outputs only under `output/phase3-system-prompt/<run-id>/` before separate approval.
- Create and read back a rollback handle before any active apply.
- Require active runtime rollback coverage before any active apply.

Before separate approval, the draft explicitly prohibits writes to:

- `~/.hermes/SOUL.md`
- `~/.hermes/config.yaml`
- `~/.hermes/hermes-agent/agent/prompt_builder.py`
- `~/.hermes/skills/`
- `~/.hermes/memories/`
- `~/.hermes/profiles/`

## Human approval gate

Human approval gate status:

- Planning authorized: `rec action GO`
- Execution approved: no
- Active apply approved: no

Separate approval is required before:

1. running GEPA/DSPy optimization;
2. running TBLite/YC-Bench benchmark commands;
3. editing Hermes Agent prompt source;
4. applying an evolved prompt to active runtime;
5. promoting a candidate to a default gate.

## Next prerequisite before execution

The next safe step is not to run Phase 3. The next safe step is to run the read-only benchmark adapter fixtures in CI/local smoke, then decide whether to replace or supplement those dry-run fixture checks with real TBLite/YC-Bench result adapters before any optimizer execution.
