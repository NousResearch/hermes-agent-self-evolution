# HSE Phase 5 P1 Manual Triage Review Packet — 20260608_230139

- Status: `P1_MANUAL_TRIAGE_REVIEW_PACKET_READY_NOT_EXECUTED`
- Scope: document and publish the `manual_triage_review` target only.
- Source run: `20260608-164455`
- Cron job: `b24aca09f168` remains paused after `1/3` fires.
- Run status: `P1_SOAK_REVIEW_REQUIRED_NO_ACTION`
- Scheduler status: `DRY_RUN_REVIEW_REQUIRED`
- Auto-triage status: `REVIEW_REQUIRED`

## Target

| Field | Value |
|---|---:|
| Metric | `tool_selection_accuracy` |
| Component | `tool_descriptions` |
| Value | `0.8889` |
| Threshold | `0.9` |
| Baseline | `0.1111` |
| Sample count | `45` |
| Pass / fail rows | `40` / `5` |
| Priority score | `0.4995` |

## Failed sanitized rows

- `tool-selection-002` — `insufficient_discrimination_margin`; expected `search_files`, selected `search_files`, margin `0.0`; excerpt: Find Python files mentioning browser_navigate in the tools directory.
- `tool-selection-003` — `insufficient_discrimination_margin`; expected `terminal`, selected `terminal`, margin `0.0`; excerpt: Run the focused pytest target for the new tool evaluation tests.
- `tool-selection-004` — `wrong_tool_selected`; expected `patch`, selected `write_file`, margin `-0.1375`; excerpt: Make a targeted replacement in one Python file and preserve surrounding content.
- `tool-selection-016` — `wrong_tool_selected`; expected `read_file`, selected `search_files`, margin `-0.005`; excerpt: Read only lines 120 through 180 of gateway/run.py.
- `tool-selection-028` — `wrong_tool_selected`; expected `terminal`, selected `execute_code`, margin `-0.0445`; excerpt: Install the project dependencies and run the npm build script.

## Manual triage tasks

1. Review the five sanitized non-passing rows and confirm classification.
2. Separate tie/near-tie discrimination issues from true wrong-tool-selection issues.
3. Identify reusable cue distinctions, especially `patch` vs `write_file`, without one-off row hacks.
4. Define a future RED regression target before implementation changes.
5. Keep all fixes, reruns, cron resume, optimizer, and production enablement blocked pending separate approval.

## Preserved OFF boundaries

Production continuous loop, threshold optimizer execution, automatic PR loop, auto-merge/deploy, network/model/API budget spend, credential modification, active runtime/source/config mutation, cron resume, and remaining P1 fires all remain OFF.
