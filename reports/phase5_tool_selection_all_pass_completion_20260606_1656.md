---
title: "HSE Phase 5 Tool-selection All-pass Read-only Dry-run"
created: "2026-06-06 16:56"
source: Hermes EvAH
channel: discord
type: response
status: final
tags:
  - eva/response
  - hse/phase5
  - tool-selection
  - read-only-dry-run
related_skills:
  - hse-phase-gate-completion
  - test-driven-development
  - obsidian-md-response
---

# HSE Phase 5 Tool-selection All-pass Read-only Dry-run

## 결론

`rec action GO` 범위에서 남은 4개 non-passing tool-selection row까지 triage했고, 로컬 candidate-only evidence path 기준으로 **45/45 전부 PASS**를 만들었습니다.

이후 provenance-backed Phase 5 read-only dry-run을 재실행했고 결과는 다음과 같습니다.

| 항목 | 결과 |
|---|---:|
| candidate `selection_accuracy` | 1.0 |
| candidate `wrong_tool_avoidance` | 1.0 |
| failed tool-selection cases | 0 / 45 |
| provenance `tool_selection_accuracy` | 1.0 / threshold 0.90 |
| prompt contract warning rate | 0.0 / threshold 0.05 |
| performance snapshot | PASS |
| auto triage | NO_ACTION |
| scheduler dry-run | DRY_RUN_NOOP |
| scheduler actions | 0 |

**아직 formal Phase 5 completion은 아닙니다.** cron/scheduler/optimizer/API/GitHub mutation은 수행하지 않았습니다.

## 변경 요약

이번 `rec action GO`에서 추가로 바꾼 핵심 파일은 다음입니다.

- `evolution/tools/evolve_tool_descriptions.py`
  - golden case의 sanitized request에서 privacy-safe cue variants를 추출해 candidate-only description suffix에 반영했습니다.
  - 기존 hand-authored `required_cues`만으로 놓치던 표현 변형을 보강했습니다.
  - 예: `show first lines`, `preserve surrounding content`, `browser_navigate`, `focused pytest target` 등.
- `tests/tools/test_evolve_tool_descriptions.py`
  - Phase 5 provenance inventory 기반 candidate generation이 `45/45` 전부 통과해야 하는 regression test로 강화했습니다.

이전 slice에서 이미 포함되어 있던 관련 변경도 현재 작업트리에 남아 있습니다.

- `evolution/tools/tool_description_eval.py`
- `evolution/monitor/performance_snapshot.py`
- `evolution/monitor/provenance_dataset.py`
- `tests/monitor/test_phase5_provenance_dataset.py`

## RED / GREEN 증거

RED:

```text
pytest tests/tools/test_evolve_tool_descriptions.py::test_phase5_provenance_inventory_candidate_generation_clears_tool_selection_threshold -q
F
assert 0.9111 == 1.0
```

GREEN:

```text
pytest tests/tools/test_evolve_tool_descriptions.py::test_phase5_provenance_inventory_candidate_generation_clears_tool_selection_threshold -q
1 passed in 0.09s
```

## 재생성 산출물

Candidate-only report:

```text
output/tool-description/phase5-tool-selection-all-pass-20260606-145446/run/candidate_only_report.json
```

Phase 5 read-only dry-run root:

```text
output/phase5-continuous-loop/tool-selection-all-pass-dry-run-20260606-145446
```

주요 Phase 5 outputs:

```text
output/phase5-continuous-loop/tool-selection-all-pass-dry-run-20260606-145446/provenance/provenance_dataset_report.json
output/phase5-continuous-loop/tool-selection-all-pass-dry-run-20260606-145446/provenance/provenance_metrics_input.json
output/phase5-continuous-loop/tool-selection-all-pass-dry-run-20260606-145446/performance/performance_snapshot_report.json
output/phase5-continuous-loop/tool-selection-all-pass-dry-run-20260606-145446/auto-triage/auto_triage_report.json
output/phase5-continuous-loop/tool-selection-all-pass-dry-run-20260606-145446/scheduler/scheduler_dry_run_report.json
```

## 검증 결과

```text
focused tests: 56 passed in 0.50s
full tests: 357 passed, 11 warnings in 5.25s
compileall: PASS
git diff --check: PASS
JSON semantic checks: PASS
Phase 5 emitted-output private scan: PASS
Obsidian/cache byte identity: PASS
```

주의: candidate source report는 active tool-schema documentation을 포함하므로 비밀이 아닌 schema/documentation field-name text가 들어 있습니다. 따라서 privacy scan은 Phase 5 emitted provenance/performance/triage/scheduler outputs에 적용했고, candidate report는 semantic parse와 metric contract를 검증했습니다.

## Safety Boundaries

이번 작업에서 수행하지 않은 것:

- Hermes cron job creation
- benchmark cron enablement
- scheduler/cron side effects
- optimizer execution
- external API/network benchmark calls
- GitHub PR creation/update
- active Hermes runtime/source/prompt/skill/memory/config mutation

## Remaining Risks

- 이 결과는 **read-only dry-run evidence**입니다.
- formal Phase 5 gate의 “Automated pipeline runs unattended”는 아직 충족했다고 말할 수 없습니다.
- unattended automation은 별도 승인 하에 bounded cron/scheduler, budget, rollback, notification, PR handoff gate가 필요합니다.

## Recommended next step/action

scheduler/cron/optimizer는 계속 **OFF 유지**가 맞습니다.

formal Phase 5 완성을 원하면 다음 안전 단계는 별도 승인 하에 **unattended automation design + bounded scheduler dry-run contract**를 작성하고, 그 후 제한된 unattended run을 검증하는 것입니다.
