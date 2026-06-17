---
title: HSE Phase 5 Tool-selection Triage Completion
created: 2026-06-06 15:38 CEST
source: Hermes/EvAH
---

# HSE Phase 5 Tool-selection Triage Completion

## 결론

**PASS.** `rec action GO` 범위에서 `tool_selection_accuracy` fail-row 원인 중 deterministic scoring의 cue-token mismatch를 최소 수정했고, provenance-backed Phase 5 read-only dry-run을 다시 1회 실행했습니다.

- `tool_selection_accuracy`: `0.9111` / threshold `0.9` → **PASS**
- `prompt_contract_warning_rate`: `0.0` / threshold `0.05` → **PASS**
- performance snapshot: **PASS**
- auto triage: **NO_ACTION**
- scheduler dry-run: **DRY_RUN_NOOP**
- scheduler dry-run actions: `0`
- scheduler side effects: `0`

중요: 이것은 **Phase 5 read-only dry-run PASS** 입니다. 실제 cron/scheduler/optimizer enablement 또는 external PR update는 수행하지 않았고, formal Phase 5 completion으로 주장하지 않습니다.

## 변경 사항

- `evolution/tools/tool_description_eval.py`
  - deterministic tool-selection scoring 전 common plural/verb cue variants를 normalize/expand.
  - 예: `files → file`, `mentioning → mention`, `replacement → replace`, `lines → line`, `directories → directory` 등.
- `tests/tools/test_evolve_tool_descriptions.py`
  - Phase 5 provenance inventory 기반 candidate generation이 `tool_selection_accuracy >= 0.90` 를 만족해야 하는 regression test 추가.

Diff stat:

```text
evolution/tools/tool_description_eval.py     | 38 +++++++++++++++++++++++++++-
 tests/tools/test_evolve_tool_descriptions.py | 30 ++++++++++++++++++++++
 2 files changed, 67 insertions(+), 1 deletion(-)
```

## RED → GREEN

- RED test: `tests/tools/test_evolve_tool_descriptions.py::test_phase5_provenance_inventory_candidate_generation_clears_tool_selection_threshold`
- RED 상태: normalization patch 전 provenance candidate `selection_accuracy`가 `0.90` threshold 미만이라 실패.
- GREEN 상태: 수정 후 동일 test 포함 focused suite 통과.

## 새 tool-selection generation 산출물

- generation root: `output/tool-description/phase5-tool-selection-triage-20260606-133625`
- candidate report: `output/tool-description/phase5-tool-selection-triage-20260606-133625/run/candidate_only_report.json`
- candidate descriptions: `output/tool-description/phase5-tool-selection-triage-20260606-133625/run/candidate_descriptions.json`
- heldout review: `output/tool-description/phase5-tool-selection-triage-20260606-133625/heldout_review.json`

Metrics:

- candidate `selection_accuracy`: `0.9111`
- candidate `wrong_tool_avoidance`: `0.9111`
- heldout passed: `True`
- heldout `selection_accuracy`: `0.8889`

## Phase 5 read-only dry-run 재실행

- output root: `output/phase5-continuous-loop/tool-selection-triaged-dry-run-20260606-133625`
- summary: `output/phase5-continuous-loop/tool-selection-triaged-dry-run-20260606-133625/phase5_tool_selection_triaged_dry_run_summary.json`
- source kind: `provenance_backed_sanitized_dataset`

Row evidence:

- tool-selection rows: `45`
- tool-selection pass/fail: `41 pass / 4 fail`
- prompt-contract rows: `20`
- prompt-contract warnings: `0`

## 남은 non-passing rows

Threshold는 통과했지만, sanitized provenance rows 기준 non-passing row는 `4`개 남아 있습니다.

- `tool-selection-001`: expected `read_file`, selected `read_file`, classification `insufficient_discrimination_margin`, margin `0.0` — Show the first 40 lines of README.md without using a shell pager.
- `tool-selection-002`: expected `search_files`, selected `terminal`, classification `wrong_tool_selected`, margin `-0.0889` — Find Python files mentioning browser_navigate in the tools directory.
- `tool-selection-004`: expected `patch`, selected `write_file`, classification `wrong_tool_selected`, margin `-0.0833` — Make a targeted replacement in one Python file and preserve surrounding content.
- `tool-selection-016`: expected `read_file`, selected `search_files`, classification `wrong_tool_selected`, margin `-0.07` — Read only lines 120 through 180 of gateway/run.py.


## 검증 결과

- focused tests: `52 passed in 0.61s`
- full HSE tests: `357 passed, 11 warnings in 5.48s`
- compileall: `PASS`
- `git diff --check`: `PASS`
- JSON semantics: `PASS`
- secret/private scan on emitted sanitized rows: `PASS`
- Obsidian/cache byte-identical: `PASS` after this report copy

## Safety invariants

- cron jobs created: `False`
- benchmark cron enabled: `False`
- optimizer execution started: `False`
- automated PR created/updated: `False`
- active runtime mutation: `False`
- external calls performed: `False`
- network calls performed: `False`

## Current git status note

```text
## hse/phase5-continuous-loop-prep
 M evolution/monitor/performance_snapshot.py
 M evolution/tools/tool_description_eval.py
 M tests/tools/test_evolve_tool_descriptions.py
?? evolution/monitor/provenance_dataset.py
?? reports/phase5_manual_triage_plan_20260606_1251.json
?? reports/phase5_manual_triage_plan_20260606_1251.md
?? reports/phase5_metric_provenance_classification_20260606_1318.json
?? reports/phase5_metric_provenance_classification_20260606_1318.md
?? reports/phase5_provenance_backed_dry_run_20260606_1428.json
?? reports/phase5_provenance_backed_dry_run_20260606_1428.md
?? tests/monitor/test_phase5_provenance_dataset.py
```

## Recommended next step/action

**scheduler/cron/optimizer는 아직 켜지지 않은 상태를 유지**하는 것이 맞습니다. 추가 정밀도를 원하면 남은 4개 non-passing row까지 계속 triage할 수 있지만, Phase 5 read-only monitor 관점에서는 현재 `tool_selection_accuracy`가 threshold를 넘어 weak target이 사라진 상태입니다.
