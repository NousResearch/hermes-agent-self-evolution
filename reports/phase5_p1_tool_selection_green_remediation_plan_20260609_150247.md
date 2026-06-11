# Phase 5 P1 Tool-selection GREEN Remediation Plan

Status: `P1_TOOL_SELECTION_GREEN_REMEDIATION_PLAN_CLEANED_FOR_LOCAL_COMMIT`
Publication: `LOCAL_PATH_LIMITED_COMMIT_NO_PUSH`
Created: `2026-06-09T15:02:47+02:00`
Cleanup: `2026-06-11T22:55:58+02:00`

## Summary

At original plan creation time, the Phase 5 P1 tool-selection evidence was still RED, and the likely GREEN path was narrow: refresh stale local candidate-only evidence from the preserved local inventory with the deterministic cue-enrichment generator.

This 2026-06-11 cleanup made the remaining draft artifacts commit-safe: the regression test now generates deterministic candidate-only evidence under pytest `tmp_path` instead of reading ignored `output/` artifacts, and stale publication/current-state wording is labeled as historical. This cleanup did **not** regenerate repo-local candidate reports, modify production source, resume cron, run an optimizer, update a PR, or mutate active Hermes runtime/tool schemas.

## Historical State Checked at Plan Creation

- Repo branch: `hse/phase5-continuous-loop-prep`
- Repo HEAD: `18182eda72929377610fe76e91002a7ac43bd2d8`
- HEAD subject: `Add Phase 5 P1 tool selection RED plan`
- Then-current candidate report: `output/tool-description/phase2e-heldout-review/run/candidate_only_report.json`
- Original local RED test: `tests/monitor/test_phase5_tool_selection_triage_regression.py`
- Existing RED plan: `reports/phase5_p1_tool_selection_triage_red_plan_20260608_234216.json`

Important worktree notes:

- At plan creation, `tests/monitor/test_phase5_tool_selection_triage_regression.py` was an untracked local RED test file; this cleanup stages it as a CI-robust `tmp_path` regression.
- `output/` is ignored by `.gitignore`; the current candidate-only report is local evidence, not a tracked PR artifact.
- The existing RED plan artifact is tracked and records `P1_TOOL_SELECTION_TRIAGE_RED_PLAN_READY_NOT_EXECUTED`.

## RED Evidence Reproduced at Plan Creation

Focused command shape:

```bash
PY=<hse-venv-python>
$PY -m py_compile tests/monitor/test_phase5_tool_selection_triage_regression.py
$PY -m pytest tests/monitor/test_phase5_tool_selection_triage_regression.py -q
```

Observed result:

- `py_compile`: exit `0`
- focused pytest: exit `1`
- result: `2 failed`
- then-current `selection_accuracy=0.8889 < 0.9000`
- `apply_ready=false`
- `metrics.candidate_only=true`
- `case_count=45`

## Failing Rows

| Row | Expected | Selected | Margin | Cue coverage | Class |
|---|---:|---:|---:|---:|---|
| `tool-selection-002` | `search_files` | `search_files` | `0.0000` | `1.0` | insufficient discrimination margin |
| `tool-selection-003` | `terminal` | `terminal` | `0.0000` | `1.0` | insufficient discrimination margin |
| `tool-selection-004` | `patch` | `write_file` | `-0.1375` | `1.0` | wrong tool selected |
| `tool-selection-016` | `read_file` | `search_files` | `-0.0050` | `1.0` | wrong tool selected |
| `tool-selection-028` | `terminal` | `execute_code` | `-0.0445` | `0.6` | wrong tool selected + cue coverage gap |

## Diagnosis

The evaluator uses a deterministic overlap score:

```text
0.65 * description_overlap + 0.2 * baseline_overlap + 0.15 * name_overlap
```

A row only passes when the expected tool score is strictly greater than every confusing score. Ties therefore fail.

The then-current local report appeared stale relative to the generator:

- current report candidate descriptions lack the expanded privacy-safe request-signal cue variants;
- generic tokens and `Prefer over ...` terms can count as positive overlap for confusing tools;
- in-memory regeneration from the same local inventory with current generator code produced `selection_accuracy=1.0` and zero failing rows, without writing files or mutating active Hermes schemas.

The historical all-pass report is useful context but is not post-cleanup P1 production proof.

## Cleanup Delta — 2026-06-11 22:55:58 CEST +0200

Performed in this cleanup:

- refactored `tests/monitor/test_phase5_tool_selection_triage_regression.py` away from ignored `output/tool-description/.../candidate_only_report.json`;
- changed the regression to create deterministic candidate-only evidence under pytest `tmp_path`;
- cleaned stale publication/current-state wording in this JSON/Markdown plan artifact for local commit readiness.

Reviewer-blocker resolution:

- stale “currently untracked” wording is explicitly historical;
- RED preflight failure expectation is labeled as plan-creation history, not current post-cleanup behavior;
- current committed-regression expectation is `2 passed` under pytest `tmp_path`.

Still not performed:

- repo-local candidate report regeneration under ignored `output/`;
- source-code fix;
- cron resume or remaining-fire execution;
- optimizer execution;
- production-loop enablement;
- automatic PR loop or external PR push/update;
- active Hermes runtime/schema mutation;
- credential, budget, or network mutation.

## GREEN Remediation Strategy

Primary path:

1. Keep the RED test thresholds intact.
2. Regenerate local candidate-only evidence from the existing local inventory using the current deterministic generator.
3. Rerun the focused RED test and require it to turn GREEN.
4. Run focused tool-description safety tests and compile checks.
5. Only after separate approval, rerun the read-only Phase 5 monitor pipeline.

Why this is the minimal path:

- current generator already passes in memory from the same inventory;
- no active Hermes schema/source apply is needed;
- failure is local evidence freshness/discrimination, not a reason to force `apply_ready=true` or weaken thresholds.

## Future Execution Plan After Separate Approval

### 1. Historical RED preflight before 2026-06-11 `tmp_path` cleanup

```bash
PY=<hse-venv-python>
$PY -m py_compile tests/monitor/test_phase5_tool_selection_triage_regression.py
$PY -m pytest tests/monitor/test_phase5_tool_selection_triage_regression.py -q
```

Historical expected result at plan creation: compile passed; pytest failed with the known five-row gap. Post-cleanup current expectation for the committed regression is `2 passed`, because the test generates candidate-only evidence under pytest `tmp_path` instead of reading ignored `output/` artifacts.

### 2. Regenerate candidate-only evidence locally

```bash
PY=<hse-venv-python>
$PY -m evolution.tools.evolve_tool_descriptions \
  --inventory-json output/tool-description/phase2e-heldout-review/run/inventory.json \
  --output-dir output/tool-description/phase2e-heldout-review/run
```

Expected: refreshed local `candidate_descriptions.json`, `candidate_only_report.json`, and `candidate.diff` under ignored `output/`, preserving `apply_ready=false` and candidate-only semantics.

### 3. Post-cleanup GREEN focused regression

```bash
$PY -m pytest tests/monitor/test_phase5_tool_selection_triage_regression.py -q
```

Expected: `2 passed`; `selection_accuracy >= 0.9000`; all five critical rows select the expected tool with `score_margin > 0.0200`; row 028 cue coverage is at least `0.9000`, using `tmp_path`-generated candidate-only evidence.

### 4. Focused safety tests

```bash
$PY -m pytest \
  tests/tools/test_evolve_tool_descriptions.py \
  tests/tools/test_tool_description_eval.py \
  tests/monitor/test_phase5_tool_selection_triage_regression.py \
  -q
$PY -m compileall -q evolution/tools tests/tools tests/monitor
```

Expected: candidate cue enrichment remains privacy-safe and report/evaluator contracts remain green.

### 5. Read-only monitor rerun only after separate approval

Use the regenerated report in the existing read-only sequence:

```text
provenance_dataset -> performance_snapshot -> auto_triage -> scheduler_dry_run
```

Expected: `tool_selection_accuracy` no longer appears as a weak metric; scheduler dry-run remains side-effect-free.

## Acceptance Criteria

- Current RED regression turns GREEN without weakening thresholds or assertions.
- Regenerated candidate-only report has `apply_ready=false`, `metrics.candidate_only=true`, `case_count=45`, and `selection_accuracy>=0.9000`.
- Rows 002, 003, 004, 016, and 028 select expected tools and have `score_margin>0.0200`; row 028 has `cue_coverage>=0.9000`.
- No active Hermes tool schema, runtime config, source apply, cron resume, optimizer execution, automatic PR loop, network/API spend, or credential mutation occurs.
- Reviewer-facing artifacts contain no private absolute paths and no high-risk credential assignments.
- The paused P1 finite cron soak remains paused until a later explicit bounded resume approval.

## Explicit Non-actions

At original plan creation time this plan did not perform:

- source-code fix;
- test-code fix or commit at plan creation time; this 2026-06-11 cleanup later performed test/report cleanup only;
- candidate report regeneration;
- provenance or monitor pipeline rerun;
- cron resume or remaining fire execution;
- optimizer execution;
- automatic PR loop;
- PR update or push;
- production-loop enablement;
- active Hermes runtime/schema mutation;
- budget/network spend;
- credential change.

## Paused Cron State

The HSE P1 finite cron soak remains paused:

- job id: `b24aca09f168`
- name: `HSE Phase 5 P1 finite cron soak 20260608-164455`
- state: `paused`
- enabled: `false`
- repeat: `1/3`
- profile: `default`

No resume was performed.

## Recommended Next Step/action

After this cleanup passes verification and is locally committed, continue with Phase 3 current-Hermes reconcile in a separate safe-mode task. Keep the P1 cron paused and do not publish, resume, optimize, or enable production automation without separate approval.
