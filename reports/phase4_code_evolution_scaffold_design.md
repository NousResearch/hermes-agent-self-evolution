# Phase 4 Code-Evolution Scaffold Design Review

Status: SPK-001 read-only design review completed.

Created: 2026-06-04 19:50 CEST

Authorization: `rec action GO`

Repository: `hermes-agent-self-evolution`

Branch: `hse/phase2e-closeout-phase3-prep`

HEAD: `f489f5c`

## Executive conclusion

Phase 4 execution remains **NO-GO**. The safe next implementation target is a **candidate-only, read-only scaffold** under `evolution.code` that defines contracts, reports, and guardrails before any Darwinian Evolver dependency install, Darwinian mutation run, Hermes Agent source edit, benchmark spend, push, or PR.

The scaffold should mirror the proven Phase 2/3 pattern:

1. read trusted inputs;
2. write fresh review artifacts only under a phase-scoped output root;
3. keep `apply_ready=false` until separately approved downstream gates pass;
4. fail closed on output-path, symlink, hardlink, source-mutation, signature, registry, secret, and external-call boundary violations;
5. emit a machine-readable report that automation can validate before any execution stage consumes it.

## Current evidence reviewed

### HSE repository state

```text
repo: /Users/snw/.hermes/evolution/repos/hermes-agent-self-evolution
branch: hse/phase2e-closeout-phase3-prep
head: f489f5c
tracked changes: none
current Phase 4 code directory: evolution/code/__init__.py placeholder only
```

Existing Phase 4 planning artifacts are untracked and document-only:

```text
.planning/spikes/001-phase4-darwinian-boundary/README.md
reports/phase4_code_evolution_planning_spike.json
reports/phase4_code_evolution_planning_spike.md
seeds/phase4_code_evolution_planning_seed.yaml
```

This design review adds one artifact:

```text
reports/phase4_code_evolution_scaffold_design.md
```

### Relevant existing patterns

| Existing area | Pattern to preserve for Phase 4 |
|---|---|
| `evolution.tools.evolve_tool_descriptions` | Candidate-only generation writes inventory, candidates, diff, and report without active Hermes schema modification. |
| `evolution.tools.report_contract` | Lightweight stdlib contract validation is suitable for CI/local smoke checks without heavy dependencies. |
| `evolution.prompts.phase3_candidate_scaffold` | Dry-run candidate scaffold snapshots inputs, rejects output traversal/pre-existing/symlinked targets, and keeps execution/apply flags false. |
| `evolution.benchmarks.contract` | Adapter reports should be read-only, fresh-output-only, path-scoped, and explicit about external-call status. |
| `evolution.prompts.phase3_preflight_gate` | A bundle-level gate can validate upstream reports while still recording execution readiness as false until real evidence and human approval exist. |
| `tests/tools/test_phase3_candidate_scaffold.py` | Monkeypatching network/process APIs in dry-run tests is a useful pattern to prove no external calls. |
| `tests/tools/test_phase3_preflight_gate.py` | Negative contract tests should verify failed gates exit non-zero but still write inspectable reports. |
| `evolution.core.external_importers` | Secret detection and privacy filtering patterns should be reused for bug/reproduction datasets and reports. |

## Proposed Phase 4 scaffold boundary

### Non-goals for the scaffold

The initial Phase 4 scaffold must not:

- install Darwinian Evolver;
- import Darwinian Evolver as a Python dependency;
- run Darwinian mutation loops;
- edit the active Hermes Agent checkout;
- edit active Hermes runtime config, gateway, SOUL, memory, skills, or model routing;
- run paid/networked benchmarks;
- push branches or open PRs;
- mark any candidate `apply_ready=true`.

### Future module layout

Recommended future implementation files, not created by this read-only design review:

```text
evolution/code/
  __init__.py
  phase4_code_scaffold.py          # candidate-only scaffold/report writer
  target_contract.py               # target spec + allowed file validation
  darwinian_cli_boundary.py        # external CLI command construction only, no import
  freeze_checks.py                 # function signature and registry.register freeze checks
  fitness_contract.py              # pytest/benchmark/reproducer scoring contract
  report_contract.py               # stdlib report schema smoke validator

tests/code/
  test_phase4_code_scaffold.py
  test_phase4_target_contract.py
  test_phase4_darwinian_cli_boundary.py
  test_phase4_freeze_checks.py
  test_phase4_fitness_contract.py
  test_phase4_report_contract.py
```

If the project prefers keeping historical Phase 2/3 tests under `tests/tools`, these tests can live there instead, but `tests/code` better matches the existing `evolution/code` package boundary.

### Future CLI shape

Recommended future local dry-run entrypoint:

```bash
python -m evolution.code.phase4_code_scaffold \
  --target-spec seeds/phase4_code_evolution_target_example.yaml \
  --hermes-repo /path/to/trusted/hermes-agent \
  --output-dir output/phase4-code-evolution/<run-id> \
  --dry-run
```

Recommended future console script:

```toml
hse-phase4-code-scaffold = "evolution.code.phase4_code_scaffold:main"
hse-validate-phase4-code-report = "evolution.code.report_contract:main"
```

These should be added only in a later implementation PR after test coverage exists.

## Target specification contract

A Phase 4 target spec should be explicit and minimal. Suggested YAML shape:

```yaml
phase: "4"
mode: "code-evolution-target"
target_id: "file-tools-patch-edge-case-001"
hermes_base:
  repo: "/path/to/trusted/hermes-agent"
  base_ref: "<commit-or-branch>"
  require_clean_worktree: true
allowed_mutation:
  files:
    - "tools/file_tools.py"
  deny_globs:
    - "**/registry.py"
    - "**/config*.py"
    - "**/*secret*"
    - "skills/**"
    - "plugins/**"
    - "memory/**"
freeze:
  function_signatures: true
  registry_register_calls: true
  public_cli_args: true
reproduction:
  failing_case_description: "short, non-secret bug/edge-case description"
  reproducer_command: "python -m pytest tests/... -q"
fitness:
  required_commands:
    - "python -m compileall -q tools/file_tools.py"
    - "python -m pytest tests/... -q"
benchmarks:
  full_benchmark_required_before_acceptance: true
  run_benchmarks_now: false
approvals:
  darwinian_install_approved: false
  darwinian_execution_approved: false
  hermes_source_mutation_approved: false
  budget_approved_usd: 0
```

Mandatory target-spec rules:

1. exactly one initial target tool/module unless a later approval expands scope;
2. target files must resolve under the trusted Hermes checkout and match an allowlist;
3. no symlinked target files;
4. no paths under credentials, profiles, skills, plugins, memories, gateway state, or runtime config;
5. no raw secrets in target spec, reproduction text, logs, or reports;
6. no `apply_ready=true` field in target specs.

## Output and report contract

Recommended output root:

```text
output/phase4-code-evolution/
```

A scaffold run should write fresh artifacts only under:

```text
output/phase4-code-evolution/<run-id>/
  target_snapshot.json
  baseline_api_surface.json
  freeze_report.json
  reproduction_contract.json
  scaffold_report.json
  review_packet.md
```

Recommended top-level `scaffold_report.json` fields:

```json
{
  "phase": "4",
  "mode": "code-evolution-candidate-only-scaffold",
  "scaffold_version": "phase4-code-scaffold-v1",
  "dry_run": true,
  "candidate_only": true,
  "read_only_inputs": true,
  "darwinian_cli_invoked": false,
  "darwinian_imported": false,
  "external_calls_performed": false,
  "package_installed": false,
  "hermes_source_mutation_performed": false,
  "active_runtime_apply_approved": false,
  "apply_ready": false,
  "passed": true,
  "failed_checks": [],
  "target_spec": {},
  "allowed_mutation": {},
  "freeze_checks": {},
  "fitness_plan": {},
  "approval_gates": {},
  "artifacts": {},
  "write_targets": [],
  "output_constraints": {
    "allowed_root": "output/phase4-code-evolution/",
    "fresh_output_required": true,
    "symlink_output_allowed": false,
    "hardlink_output_allowed": false,
    "input_output_overlap_allowed": false,
    "hermes_source_write_allowed": false
  }
}
```

The report contract should fail if:

- required fields are absent;
- any apply payload appears (`patch`, `patches`, `write_paths`, `apply_payload`, `source_updates`);
- any execution/apply/source-mutation flag is true in scaffold mode;
- output paths escape `output/phase4-code-evolution/`;
- any write target pre-exists, is a symlink, is a hardlink to input, or overlaps an input artifact;
- a report contains raw secret patterns;
- `passed=true` while `failed_checks` is non-empty, or `passed=false` while `failed_checks` is empty.

## Darwinian Evolver external CLI boundary

The Phase 4 design should preserve an external-process boundary for Darwinian Evolver:

- HSE may construct a command line, input manifest, and output directory.
- HSE should not import Darwinian Evolver classes into Hermes Agent or active Hermes runtime.
- If an isolated smoke is later approved, run it in a disposable venv/worktree with sanitized environment variables.
- The command builder should default to `--dry-run` or equivalent no-mutation mode if available.
- The scaffold should record `darwinian_cli_invoked=false`; only a later execution-specific report may set it true.
- AGPL/license handling should remain a human/legal review item; this design only enforces a technical subprocess boundary.

Suggested future command-boundary report fields:

```json
{
  "engine": "darwinian-evolver",
  "integration_mode": "external_cli_subprocess_only",
  "python_import_allowed": false,
  "active_runtime_import_allowed": false,
  "sanitized_env_required": true,
  "network_default": "disabled_or_explicitly_approved",
  "license_review_required": true
}
```

## Sandbox and mutation model

For later execution, mutations should occur only in an isolated Hermes Agent worktree:

1. verify active Hermes checkout is not the mutation target;
2. create a temporary or named worktree from a recorded base ref;
3. create a branch such as `hse-phase4/<target-id>`;
4. give Darwinian Evolver only the isolated worktree path and allowlisted target files;
5. collect diffs, reports, and metrics back into HSE output artifacts;
6. never auto-apply to the active Hermes runtime;
7. require human review before push/PR.

The scaffold should not create or mutate this worktree yet unless a later implementation explicitly adds a read-only target snapshot stage.

## Freeze checks

Phase 4 needs stronger freeze checks than earlier text-evolution phases. At minimum:

### Function signature freeze

Compare baseline vs candidate AST for allowlisted files:

- module-level functions;
- class names;
- method names;
- positional/keyword-only/vararg/kwarg parameter names;
- default count, decorator names, and async/sync status.

Initial policy: any public signature drift is a hard failure. Private helper signature drift can be allowed only if target spec explicitly permits it.

### `registry.register()` freeze

Detect and compare calls matching:

```python
registry.register(...)
```

Freeze at least:

- registered tool name;
- toolset;
- schema parameter names/types/required fields;
- handler reference;
- description field unless the target explicitly combines Phase 2 tool-description changes, which should be disallowed initially.

### Safety/error-handling preservation

Use AST/string heuristics as a first pass:

- block deletion of exception handlers unless tests prove equivalent behavior and human review approves;
- flag removed `ValueError`, `PermissionError`, `FileNotFoundError`, `TimeoutError`, and explicit path/symlink checks;
- flag new subprocess/network calls unless target spec approves and tests sandbox them;
- flag new writes outside the allowlisted worktree.

## Fitness and gate order

Recommended gate sequence for a future execution candidate:

1. **Scaffold contract gate**: target spec and output contract valid; no source mutation yet.
2. **Baseline reproduction gate**: chosen bug/edge-case must fail on baseline or otherwise produce an objective baseline metric.
3. **Mutation sandbox gate**: Darwinian execution occurs only in isolated worktree after approval.
4. **Static candidate gate**: `git diff --check`, compile/syntax checks, secret scan, freeze checks, allowed-file diff guard.
5. **Targeted reproduction gate**: target bug fixed or metric improved.
6. **Relevant pytest gate**: target-specific and adjacent Hermes tests pass.
7. **Full Hermes pytest gate**: full suite passes before any PR-ready claim.
8. **Benchmark gate**: TBLite + YC-Bench/TerminalBench evidence or explicitly recorded waiver before acceptance.
9. **Human review gate**: every code diff reviewed; no auto-merge.
10. **Handoff gate**: branch/PR metadata recorded; active runtime apply remains separate.

A candidate can be called `review_ready=true` only after gates 1-7 pass. It can be called `acceptance_ready=true` only after gates 1-9 pass. It should not be called `apply_ready=true` until a separately approved apply path exists.

## Tests to implement before any execution

### Scaffold/report tests

- writes only fresh artifacts under `output/phase4-code-evolution/`;
- rejects output traversal;
- rejects pre-existing output targets;
- rejects symlinked output targets;
- rejects hardlink/input overlap;
- rejects reports with apply payloads;
- keeps all source-mutation and execution flags false;
- writes a human review packet.

### Target contract tests

- accepts one allowlisted Hermes tool file under a trusted repo;
- rejects target paths outside repo;
- rejects symlink targets;
- rejects multiple target files unless explicitly approved;
- rejects credentials/config/runtime/profile paths;
- rejects target specs containing secret-pattern matches.

### External boundary tests

- dry-run scaffold monkeypatches `subprocess.run`, `subprocess.Popen`, `socket.socket`, `socket.create_connection`, `urllib.request.urlopen`, and `os.system` to prove no external calls;
- CLI-boundary builder returns command vectors but does not execute them in scaffold mode;
- missing Darwinian dependency is recorded as a blocker, not as a Python import failure in normal scaffold mode.

### Freeze-check tests

- detects public function signature changes;
- detects handler/schema/tool-name drift in `registry.register()` calls;
- allows comments/docstring-only changes;
- flags removed exception/path/symlink checks;
- fails closed on unparsable Python.

### Fitness-contract tests

- validates required command list shape;
- records baseline reproduction status;
- refuses to mark improvement without a failing baseline or objective metric delta;
- keeps benchmark spend at zero unless explicitly approved.

## Implementation sequencing

Recommended safe sequence after this design review:

1. Implement `target_contract.py` and `report_contract.py` with tests only.
2. Implement `phase4_code_scaffold.py` as a dry-run report writer with no Darwinian dependency.
3. Add `freeze_checks.py` with AST-based signature and registry freeze tests.
4. Add `darwinian_cli_boundary.py` command construction only; no execution.
5. Add `fitness_contract.py` for command/report schema, not command execution.
6. Run HSE targeted tests for the new modules.
7. Only then request separate approval for SPK-002 isolated dependency smoke.

## Required approvals still pending

Separate approval is still required before any of the following:

- installing Darwinian Evolver or any new dependency;
- importing or executing Darwinian Evolver;
- creating mutation worktrees that write Hermes Agent source;
- running networked or paid benchmarks;
- pushing a branch or opening a PR;
- modifying active Hermes runtime, gateway, memory, skills, SOUL, profiles, or model routing.

## Decision

SPK-001 read-only design review is complete.

Recommended next step/action: implement the **dry-run-only Phase 4 scaffold/report contract** (`target_contract.py`, `report_contract.py`, and `phase4_code_scaffold.py`) with tests, still without installing or running Darwinian Evolver.
