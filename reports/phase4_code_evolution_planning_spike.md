# Phase 4 Code Evolution Planning/Spike

Status: completed planning artifacts only.

This report records the safe Phase 4 planning/spike work authorized by `rec action GO`. It does not run Darwinian Evolver, does not install dependencies, does not mutate Hermes Agent source, and does not apply runtime changes.

## Created artifacts

- Seed: `seeds/phase4_code_evolution_planning_seed.yaml`
- Manifest: `reports/phase4_code_evolution_planning_spike.json`
- Spike README: `.planning/spikes/001-phase4-darwinian-boundary/README.md`
- This report: `reports/phase4_code_evolution_planning_spike.md`

## Current repo evidence

```text
repo: /Users/snw/.hermes/evolution/repos/hermes-agent-self-evolution
branch: hse/phase2e-closeout-phase3-prep
head: f489f5c
initial status: clean
tracked Phase 4 code file: evolution/code/__init__.py
current Phase 4 code state: placeholder only
```

Current `evolution/code/__init__.py` is still only a phase placeholder. There is no Phase 4 entrypoint, code-as-organism wrapper, fitness runner, freeze checker, or reproduction dataset runner yet.

## Phase 3 dependency state

Phase 3 publish was completed as a draft PR:

```text
PR: https://github.com/NousResearch/hermes-agent/pull/39233
state: OPEN
draft: true
mergeable: MERGEABLE
checks: no checks reported
```

Phase 3 manifest state previously checked:

```text
status: completed_with_local_active_source_apply_and_bounded_smoke_validation
review: PASS_WITH_NOTES
full_external_benchmark_executed: false
active_apply_ready_now: true
```

This is good enough for Phase 4 planning, but not enough for Phase 4 execution.

## External boundary research

External sources checked as untrusted data:

1. Hermes Darwinian Evolver docs: record upstream tool as AGPL-3.0 and require invocation only through upstream CLI/subprocess/`uv run`; no import of upstream classes into Hermes itself.
2. Imbue Darwinian Evolver README: describes an evolutionary loop that maintains organisms, mutates candidate solutions, and scores them.
3. Imbue Darwinian Evolver license: GNU Affero General Public License v3.
4. HSE PLAN: records Darwinian Evolver for code files/algorithms/tool implementations, external CLI only.

## Go/No-Go

### Phase 4 execution: NO-GO

Still blocked because:

- Phase 4 code implementation is placeholder-only.
- Darwinian Evolver is not installed or verified in the current HSE environment.
- Phase 3 PR is still open/draft and has no reported checks.
- Phase 3 full external benchmark evidence is absent.
- No reproduction-first bug target has been selected.
- No function-signature or `registry.register()` freeze checker exists yet.

### Phase 4 planning/spike: GO

The following planning/spike work is safe and now recorded:

| Spike | Purpose | Status |
|---|---|---|
| SPK-001 darwinian-external-cli-boundary | Preserve AGPL external CLI/subprocess boundary | PARTIAL: documented, not executable |
| SPK-002 isolated-dependency-smoke | Verify dependency in isolated venv only | PLANNED: needs separate install/network approval |
| SPK-003 signature-registry-freeze-checker | Block API/tool discovery drift | PLANNED |
| SPK-004 first-known-bug-reproduction | Ensure optimizer targets a concrete failing case | PLANNED |

## Required next gates before code mutation

Before any actual Phase 4 code mutation, require all of the following:

1. Phase 3 PR reviewed/merged, or CI/check evidence and explicit waiver recorded.
2. Full external benchmark gate decision recorded, or explicit waiver recorded.
3. Isolated Darwinian Evolver CLI smoke passed outside active Hermes runtime.
4. One target bug or edge case selected with a failing reproduction test.
5. Target tool file allowlist approved.
6. Baseline git status, branch, and rollback handle recorded.
7. Function-signature and `registry.register()` freeze checker implemented and tested.
8. Human approval records nonzero budget/runtime limits.

## Stop conditions

Stop and request confirmation before:

- installing external dependencies;
- running networked benchmark commands;
- running Darwinian Evolver mutation;
- editing Hermes Agent source files;
- pushing/publishing evolved-code branches or PRs;
- changing gateway, credentials, memory, skills, SOUL, model routing, or active runtime config.

## Recommended next step/action

Run **SPK-001 read-only design review** into a more detailed implementation plan for `evolution.code` scaffold and freeze checkers. If Sunwoo wants to validate dependency feasibility, request separate approval for **SPK-002 isolated dependency install smoke**.
