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

- `review/baseline_system_prompt.json`
- `review/candidate_system_prompt.json`
- `review/candidate_only_report.json`
- `review/review_packet.md`
- benchmark result JSON files under `benchmarks/`
- `phase3_preflight_report.json` under `preflight/`

No active prompt/source apply is included in this draft.

## Candidate-only scaffold

The candidate-only scaffold is now a runnable review-artifact contract:

```bash
python -m evolution.prompts.phase3_candidate_scaffold \
  --baseline-prompt output/phase3-system-prompt/<run-id>/inputs/baseline_system_prompt.json \
  --candidate-prompt output/phase3-system-prompt/<run-id>/inputs/candidate_system_prompt.json \
  --output-dir output/phase3-system-prompt/<run-id>/review/ \
  --dry-run
```

It writes only baseline/candidate snapshots, `candidate_only_report.json`, and `review_packet.md` under `output/phase3-system-prompt/`, and rejects pre-existing write targets, symlinked write targets, or input/output path overlap. The report keeps `candidate_only=true`, `execution_started=false`, `run_gepa_now=false`, `run_dspy_now=false`, `real_benchmarks_executed=false`, and `apply_ready=false`; it fails closed if any non-evolvable section changes.

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
  --baseline-prompt output/phase3-system-prompt/<run-id>/review/baseline_system_prompt.json \
  --candidate-prompt output/phase3-system-prompt/<run-id>/review/candidate_system_prompt.json \
  --fixtures-jsonl datasets/golden/benchmarks/phase3-system-prompt/tblite_cases.jsonl \
  --output-json output/phase3-system-prompt/<run-id>/benchmarks/tblite.json \
  --dry-run
```

```bash
python -m evolution.benchmarks.run_yc_bench \
  --baseline-prompt output/phase3-system-prompt/<run-id>/review/baseline_system_prompt.json \
  --candidate-prompt output/phase3-system-prompt/<run-id>/review/candidate_system_prompt.json \
  --fixtures-jsonl datasets/golden/benchmarks/phase3-system-prompt/yc_bench_fast_test.jsonl \
  --preset fast_test \
  --output-json output/phase3-system-prompt/<run-id>/benchmarks/yc_bench.json \
  --dry-run
```

The fixed adapter contract requires `mode=dry-run-fixture`, `candidate_only=true`, `read_only=true`, `external_calls_performed=false`, and `apply_ready=false` in each output report. `--output-json` is constrained to `.json` files under `output/phase3-system-prompt/` and must be fresh; resolved paths must remain under that root; pre-existing, symlinked, hardlinked, and input-overlapping output targets are rejected; and tests monkeypatch socket, `urllib.request.urlopen`, `subprocess.run`/`Popen`, and `os.system` to fail if fixture adapters attempt network or external process calls. Real benchmark execution remains deferred until separate human approval.

## Local preflight gate

The local preflight gate validates the candidate scaffold report together with dry-run TBLite/YC-Bench adapter reports:

```bash
python -m evolution.prompts.phase3_preflight_gate \
  --candidate-report output/phase3-system-prompt/<run-id>/review/candidate_only_report.json \
  --tblite-report output/phase3-system-prompt/<run-id>/benchmarks/tblite.json \
  --yc-bench-report output/phase3-system-prompt/<run-id>/benchmarks/yc_bench.json \
  --output-json output/phase3-system-prompt/<run-id>/preflight/phase3_preflight_report.json \
  --dry-run
```

A passing local preflight report means the local dry-run artifacts are coherent and prompt checksums match across reports. It still records `phase3_execution_ready=false`, `real_benchmarks_executed=false`, and `human_approval_required_before_execution=true`; real benchmark evidence and separate human approval remain blocking before optimizer execution, source edits, active apply, or default-gate promotion.

## Phase 3 real benchmark readiness manifest

The Phase 3 real benchmark readiness manifest is recorded in `reports/phase3_real_benchmark_readiness_manifest.json` and `reports/phase3_real_benchmark_readiness_manifest.md`. It keeps `real_benchmark_ready_now=false` and `active_apply_ready_now=false` while machine-recording the future inputs, environment requirements, approval boundaries, cost/runtime limits, rollback requirements, and go/no-go conditions needed before replacing dry-run fixtures with real TBLite/YC-Bench evidence.

The manifest is a readiness contract only. It does not run real benchmarks, does not approve GEPA/DSPy optimization, does not authorize nonzero benchmark/API spend, and does not permit prompt-source edits or active runtime apply.

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

The next safe step is not to run real Phase 3 optimization. The next safe step is to keep using the local preflight gate to validate candidate scaffold and dry-run benchmark artifacts, then decide whether to replace or supplement those dry-run fixture checks with real TBLite/YC-Bench result adapters before any optimizer execution.
