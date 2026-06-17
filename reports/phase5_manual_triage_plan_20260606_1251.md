---
title: "HSE Phase 5 Manual Triage Plan"
created: "2026-06-06 12:51 CEST"
source: Hermes EvAH
channel: Discord
type: response
status: final
tags:
  - eva/response
  - hse/phase5
  - hse/triage
  - verification
related_skills:
  - hse-phase-gate-completion
  - writing-plans
  - obsidian-md-response
pdf_export: false
---

# HSE Phase 5 Manual Triage Plan

## 결론

`rec action GO`에 따라 직전 권장 action을 실행했습니다.

**이번 산출물은 Phase 5 formal completion이 아니라, read-only dry-run이 지목한 두 약점에 대한 manual triage plan입니다.**

대상:

1. `tool_selection_accuracy`
2. `prompt_contract_warning_rate`

Source evidence:

```text
dry_run_root=output/phase5-continuous-loop/unattended-dry-run-20260606-124310
performance_status=NEEDS_TRIAGE
auto_triage_status=REVIEW_REQUIRED
scheduler_status=DRY_RUN_REVIEW_REQUIRED
scheduler_dry_run_action_count=2
scheduler_side_effect_count=0
```

## Safety Boundary

이번 plan 작성에서 하지 않은 것:

- Hermes cron job 생성 없음
- benchmark cron enablement 없음
- GEPA/DSPy/Darwinian optimizer 실행 없음
- GitHub PR 자동 생성/수정 없음
- 외부 네트워크/API 호출 없음
- active Hermes runtime/source/prompt/skill/memory/config mutation 없음

## Target 1 — `tool_selection_accuracy`

| Field | Value |
|---|---:|
| component | `tool_descriptions` |
| value | `0.86` |
| threshold | `0.90` |
| baseline | `0.88` |
| sample_count | `90` |
| severity | `0.04` |
| priority_score | `3.6` |

### Initial hypotheses

- Tool descriptions may overlap semantically, causing ambiguous tool routing in held-out examples.
- The metric may be fixture-derived rather than a live benchmark regression; first step is evidence classification, not mutation.
- Failures may cluster around file/search/terminal/web boundaries or mandatory skill-loading behavior.

### Manual triage tasks

1. Locate the metric source and fixture/eval file that produced `tool_selection_accuracy=0.86`.
2. Extract a sanitized miss table: expected tool, selected tool, task text class, and failure class.
3. Classify misses as:
   - description ambiguity;
   - eval-label issue;
   - task ambiguity;
   - true routing regression.
4. Draft candidate tool-description edits only for true description ambiguity.
5. Define a future reviewed fix-slice threshold before any mutation.

### Definition of done

- A sanitized miss-classification table exists.
- Top 3 failure clusters are named with evidence counts.
- Any proposed edit links to a specific failure cluster.
- No scheduler, optimizer, API, or external PR mutation occurred.

## Target 2 — `prompt_contract_warning_rate`

| Field | Value |
|---|---:|
| component | `system_prompts` |
| value | `0.07` |
| threshold | `0.05` |
| baseline | `0.06` |
| sample_count | `20` |
| severity | `0.02` |
| priority_score | `0.4` |

### Initial hypotheses

- Warnings may reflect prompt-contract drift between Phase 3 guidance and current tests/fixtures.
- Warning rate may be sensitive to small sample size; each warning needs class-level inspection before prompt change.
- Some warnings may be documentation/fixture warnings rather than behavioral prompt regressions.

### Manual triage tasks

1. Locate the report or fixture that produced `prompt_contract_warning_rate=0.07`.
2. Build a sanitized warning table: warning class, affected prompt contract area, count, severity.
3. Separate behavior-affecting warnings from harmless documentation/fixture warnings.
4. Draft candidate Phase 3 prompt-contract adjustments only for behavior-affecting warning classes.
5. Define focused tests needed before any prompt change is applied to the Hermes operating checkout.

### Definition of done

- A sanitized warning-class table exists.
- Each warning class is marked behavior-affecting, fixture-only, or inconclusive.
- Future fixes include explicit focused tests before prompt mutation.
- No active runtime prompt/config mutation occurred in this triage step.

## Execution sequence

1. Freeze the existing dry-run evidence pack and do not overwrite it.
2. Trace metric provenance in repo-local fixtures/reports/tests.
3. Produce sanitized classification tables for both targets.
4. Write a follow-up triage findings artifact with `PASS / WARN / INCONCLUSIVE` labels.
5. Only after human review, open a separate fix slice for candidate edits and tests.

## Future verification commands

```bash
cd ~/.hermes/evolution/repos/hermes-agent-self-evolution
PY=~/.hermes/evolution/venvs/self-evolution/bin/python
$PY -m pytest tests/monitor/test_phase5_performance_monitor.py tests/monitor/test_phase5_auto_triage.py tests/monitor/test_phase5_scheduler_dry_run.py tests/monitor/test_phase5_continuous_loop_readiness_manifest.py -q
$PY -m pytest tests -q
git diff --check
```

## Blocked actions without separate approval

- create or enable Hermes cron jobs;
- start GEPA/DSPy/Darwinian optimizer execution;
- spend benchmark/API budget;
- mutate active Hermes runtime prompt/config/skills/memory;
- create or update external GitHub PRs automatically.

## Artifacts

Repo-local HSE artifacts:

```text
reports/phase5_manual_triage_plan_20260606_1251.json
reports/phase5_manual_triage_plan_20260606_1251.md
```

Obsidian response artifact:

```text
SnwEvAH/Response/2026-06-06-1251-hse-phase5-manual-triage-plan.md
```

## Recommended next step/action

**Run read-only metric provenance tracing** and create sanitized miss/warning classification tables for the two targets. Keep scheduler/optimizer/API/GitHub mutation OFF.
