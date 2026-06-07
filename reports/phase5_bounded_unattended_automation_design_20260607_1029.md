# HSE Phase 5 — Bounded Unattended Automation Design + Dry-run Contract

## 결론

이 문서는 **formal Phase 5 완성으로 가기 위한 bounded unattended automation 설계와 dry-run contract**입니다.

중요한 경계는 명확합니다.

- 이번 승인으로 수행한 것은 **문서/계약 작성**입니다.
- 이번 단계에서는 **cron/scheduler enablement, optimizer 실행, 외부 PR 자동화, active Hermes runtime 변경을 하지 않았습니다.**
- 따라서 이 문서는 **Phase 5 formal completion claim 자체가 아니라**, formal completion을 안전하게 닫기 위한 **L0 계약 산출물**입니다.

현재 read-only evidence chain은 강합니다.

- `tool_selection_accuracy = 1.0`
- `prompt_contract_warning_rate = 0.0`
- performance snapshot: `PASS`
- auto-triage: `NO_ACTION`
- scheduler dry-run: `DRY_RUN_NOOP`
- scheduler actions: `0`

하지만 `PLAN.md`의 Phase 5 gate는 **“Automated pipeline runs unattended”**입니다. 그러므로 다음에는 이 계약에 따라 **정확히 1회의 bounded unattended read-only dry-run**을 실행해 unattended proof를 만들어야 합니다.

## 승인 범위

승인 문구:

> Sunwoo rec action GO — formal Phase 5 완성하자; bounded unattended automation design + dry-run contract 작성 승인.

이번 승인에 포함된 것:

1. reviewer-readable Markdown 설계 문서 작성
2. machine-readable JSON dry-run contract 작성
3. Obsidian Response artifact 작성 및 Discord 첨부용 staged copy 생성
4. JSON/경로/첨부 동일성/고위험 credential material 미포함 검증

이번 승인에 **포함되지 않은 것**:

- Hermes cron job 생성/활성화
- GEPA/DSPy/Darwinian optimizer 실행
- 유료 또는 networked benchmark/API 호출
- GitHub PR 자동 생성/갱신/merge
- active Hermes Agent source, skill, prompt, memory, config, runtime 상태 변경
- candidate를 EvAH/Hermes 운영 checkout에 apply

## 현재 상태 요약

| 항목 | 현재 값 |
|---|---|
| HSE repo | `hermes-agent-self-evolution` |
| branch | `hse/phase5-continuous-loop-prep` |
| HEAD | `74210b6c7dc8f45b7de39a2c085e9018074018b8` |
| worktree | 기존 Phase 5 provenance/tool-selection 관련 modified/untracked artifacts 존재 |
| formal Phase 5 claim | `false` |
| continuous loop enabled | `false` |
| cron jobs created | `false` |
| optimizer execution | `false` |

주의: worktree에는 기존 Phase 5 후속 산출물이 남아 있습니다. 이 계약 작성은 해당 변경을 commit/push/apply/schedule하지 않습니다.

## Phase 5 gate 해석

`PLAN.md`의 Phase 5는 다음을 요구합니다.

> Automated pipeline runs unattended

엄격히 보면 Phase 5 formal completion은 다음 중 최소한의 증거가 필요합니다.

1. monitor/benchmark 또는 equivalent dry-run pipeline이 unattended로 시작되어 인간 입력 없이 끝난다.
2. auto-triage가 underperforming target을 찾거나 valid metrics에서 `NO_ACTION`을 명시한다.
3. 승인된 budget/safety gate 안에서 최소 1회의 bounded optimization 또는 no-op decision cycle이 end-to-end로 완료된다.
4. 생성된 PR/후보는 사람이 review/merge한다. auto-merge/deploy는 금지된다.

현재 상태는 read-only chain이 좋지만 unattended scheduler/optimizer/PR handoff cycle이 아직 실행되지 않았으므로 **formal completion은 아직 주장하지 않습니다.**

## Automation levels

### L0 — Contract only

현재 단계입니다.

허용 side effect:

- `reports/phase5_bounded_unattended_automation_design_20260607_1029.json`
- `reports/phase5_bounded_unattended_automation_design_20260607_1029.md`
- `SnwEvAH/Response/2026-06-07-1029-hse-phase5-bounded-unattended-automation-contract.md`
- Discord 첨부용 byte-identical cache copy

허용 claim:

> Phase 5 bounded automation contract drafted; not formal completion.

### L1 — One-shot unattended read-only dry-run

별도 승인 후 다음 안전 단계입니다.

Bounds:

- repeat count: max `1`
- max total runtime: `1800s`
- network calls: `false`
- optimizer execution: `false`
- external PR update: `false`
- active runtime mutation: `false`
- allowed write roots:
  - `output/phase5-continuous-loop/bounded-unattended-dry-run-<stamp>/`
  - `reports/phase5_bounded_unattended_run_<stamp>.json`
  - `reports/phase5_bounded_unattended_run_<stamp>.md`

허용 claim:

> One unattended dry-run proof passed.

단, 이것만으로 full formal Phase 5 completion은 아닙니다. optimizer/handoff gate가 별도로 충족되거나 명시적으로 waive되어야 합니다.

### L2 — Budgeted unattended optimization cycle

별도 budget/optimizer 승인 후 가능한 formal completion 후보 단계입니다.

Bounds:

- repeat count: max `1`
- candidate apply: `false`
- human review: `required`
- auto merge/deploy: `false`
- benchmark/API/network budget cap: required
- optimizer/model routing: explicitly approved

허용 claim:

> Formal Phase 5 candidate completion after one detected target is optimized into a candidate artifact and review packet without manual intervention, with human review preserved.

## L1 dry-run contract

Contract name:

```text
phase5-bounded-unattended-readonly-dry-run
```

Mode:

```text
read_only_unattended_dry_run
```

Pipeline:

```text
preflight_git_and_environment_snapshot
→ provenance_dataset
→ performance_snapshot
→ auto_triage
→ scheduler_dry_run
→ semantic_safety_verifier
→ run_summary_report
```

Required inputs:

```text
tool_selection_report_json = output/tool-description/phase5-tool-selection-all-pass-20260606-145446/run/candidate_only_report.json
heldout_review_json       = output/tool-description/phase2e-heldout-review/heldout_review.json
prompt_source             = prompt_builder=<HERMES_AGENT_CHECKOUT>/agent/prompt_builder.py
hse_python                = <HSE_VENV>/bin/python
```

Command template:

```bash
cd <HSE_REPO>
PY=<HSE_VENV>/bin/python
STAMP=$(date -u +%Y%m%d-%H%M%S)
RUN_ROOT="output/phase5-continuous-loop/bounded-unattended-dry-run-${STAMP}"

$PY -m evolution.monitor.provenance_dataset   --tool-selection-report-json output/tool-description/phase5-tool-selection-all-pass-20260606-145446/run/candidate_only_report.json   --heldout-review-json output/tool-description/phase2e-heldout-review/heldout_review.json   --prompt-source prompt_builder=<HERMES_AGENT_CHECKOUT>/agent/prompt_builder.py   --output-dir "${RUN_ROOT}/provenance"   --window-start <YYYY-MM-DD>   --window-end <YYYY-MM-DD>

$PY -m evolution.monitor.performance_snapshot   --metrics-json "${RUN_ROOT}/provenance/provenance_metrics_input.json"   --output-dir "${RUN_ROOT}/performance"

$PY -m evolution.monitor.auto_triage   --performance-report-json "${RUN_ROOT}/performance/performance_snapshot_report.json"   --output-dir "${RUN_ROOT}/auto-triage"

$PY -m evolution.monitor.scheduler_dry_run   --auto-triage-report-json "${RUN_ROOT}/auto-triage/auto_triage_report.json"   --output-dir "${RUN_ROOT}/scheduler"

$PY scripts_or_inline_verifier.py --run-root "${RUN_ROOT}"
```

Verifier required checks:

- every JSON report parses
- output root is under `output/phase5-continuous-loop/`
- all expected files exist and are non-empty
- `read_only=true`
- all side-effect flags remain false:
  - `cron_jobs_created`
  - `benchmark_cron_enabled`
  - `scheduler_or_cron_side_effects_performed`
  - `notifications_sent`
  - `optimizer_execution_started`
  - `automated_pr_created_or_updated`
  - `active_runtime_mutation`
  - `external_calls_performed`
  - `network_calls_performed`
  - `raw_private_session_data_committed`
  - `raw_credentials_recorded`
- scheduler actions, if any, remain dry-run/manual-review only
- no source/runtime/config/memory/skill mutation
- no GitHub/network/benchmark/optimizer side effects

Allowed terminal statuses:

| Status | Meaning |
|---|---|
| `PASS_NO_ACTION` | Valid metrics, no weak target, scheduler dry-run noop |
| `PASS_REVIEW_REQUIRED` | Valid metrics, weak target found, dry-run manual review action emitted, no side effects |
| `FAIL_BLOCKED` | Contract violation or execution failure; fail closed and report only |

Failure policy:

> Fail closed and report only. Do not retry blindly. Do not enable scheduler or optimizer after failure.

## Acceptance criteria

| ID | Name | Required for | Pass condition |
|---|---|---|---|
| P5-BUA-01 | unattended execution proof | L1 dry-run | Dry-run starts from a single command or one-shot scheduled invocation, completes without interactive input, and emits a terminal summary with exit status. |
| P5-BUA-02 | side-effect zero | L1 dry-run | Every emitted safety invariant remains read-only and all side-effect flags are false. |
| P5-BUA-03 | bounded runtime and scope | L1 dry-run | Runtime, repeat count, write roots, input files, and output files stay within declared bounds. |
| P5-BUA-04 | reviewable artifact packet | L1 dry-run | JSON and Markdown summaries exist, parse/read back, and include source inputs, statuses, safety flags, and recommended next action. |
| P5-BUA-05 | formal completion gate not weakened | formal Phase 5 claim | Report distinguishes read-only dry-run success from full formal completion unless a separately approved optimizer/handoff cycle also passes or is explicitly waived. |

## Human review gates

Before L1 one-shot scheduler invocation:

1. exact invocation method 승인: manual background command vs Hermes cron one-shot
2. delivery target 승인: local only vs current Discord thread
3. runtime timeout and repeat count 승인

Before L2 optimizer cycle:

1. benchmark/API/network budget 승인
2. optimizer family and model routing 승인
3. candidate output root and PR handoff policy 승인
4. no auto-merge/deploy 재확인

## Machine-readable contract

Repo JSON artifact:

```text
reports/phase5_bounded_unattended_automation_design_20260607_1029.json
```

Repo Markdown artifact:

```text
reports/phase5_bounded_unattended_automation_design_20260607_1029.md
```

## Recommended next step/action

다음 안전 조치는 **L1 one-shot unattended read-only dry-run을 정확히 1회 실행**하는 것입니다.

실행 전 별도 확인이 필요한 선택지는 세 가지입니다.

1. invocation method: manual background command 또는 Hermes cron one-shot
2. delivery target: local only 또는 현재 Discord thread
3. timeout/repeat: 기본 `repeat=1`, `max_total_runtime=1800s`

그 전까지 scheduler/cron/optimizer/PR automation은 계속 OFF가 맞습니다.
