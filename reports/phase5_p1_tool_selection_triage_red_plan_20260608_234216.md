# HSE Phase 5 P1 Tool-selection Triage / RED Regression Plan — 20260608_234216

- Status: `P1_TOOL_SELECTION_TRIAGE_RED_PLAN_READY_NOT_EXECUTED`
- Scope: read-only/manual triage and future RED-test planning only.
- Publication: `LOCAL_ONLY_NOT_COMMITTED_OR_PUSHED`
- PR head at plan time: `35e113a7d6da0610d123d3ef1ec7671c8df1f45c`
- Checks caveat: `MISSING_NOT_CI_PASS`

## Current metric state

| Field | Value |
|---|---:|
| Metric | `tool_selection_accuracy` |
| Value | `0.8889` |
| Threshold | `0.9` |
| Baseline | `0.1111` |
| Sample count | `45` |
| Pass / fail rows | `40` / `5` |
| Performance snapshot | `NEEDS_TRIAGE` |
| Auto-triage | `REVIEW_REQUIRED` |
| Scheduler dry-run | `DRY_RUN_REVIEW_REQUIRED` |

## Failed row triage

### `tool-selection-002` — `insufficient_discrimination_margin`

- Excerpt: Find Python files mentioning browser_navigate in the tools directory.
- Expected / selected: `search_files` / `search_files`
- Score margin: `0.0`; cue coverage: `1.0`
- Finding: Correct tool is selected, but search_files and terminal tie at the score boundary. Treat as insufficient discrimination, not a wrong-tool case.
- Hypothesis: File-discovery/content-search cues such as find files, mentioning, and directory need enough search_files margin over terminal shell execution.
- Future RED assertion: For this row, selected_tool remains search_files and expected_score - top_confusing_score > 0.02.
- Safe fix direction: Improve reusable search_files cue weighting/description for file discovery and content search without shelling out; avoid row-specific hacks.

### `tool-selection-003` — `insufficient_discrimination_margin`

- Excerpt: Run the focused pytest target for the new tool evaluation tests.
- Expected / selected: `terminal` / `terminal`
- Score margin: `0.0`; cue coverage: `1.0`
- Finding: Correct tool is selected, but terminal ties with execute_code. This is an insufficient discrimination margin for pytest execution.
- Hypothesis: Shell command/test-run cues such as run pytest target should distinguish terminal from Python snippet execution.
- Future RED assertion: For this row, selected_tool remains terminal and expected_score - top_confusing_score > 0.02.
- Safe fix direction: Strengthen generic terminal cues for command execution, package managers, builds, and test runners; avoid weakening execute_code for script-processing tasks.

### `tool-selection-004` — `wrong_tool_selected`

- Excerpt: Make a targeted replacement in one Python file and preserve surrounding content.
- Expected / selected: `patch` / `write_file`
- Score margin: `-0.1375`; cue coverage: `1.0`
- Finding: Wrong tool selected: write_file outranks patch for a targeted replacement that should preserve surrounding content.
- Hypothesis: Targeted replacement and preserve-surrounding-content cues should select patch; complete overwrite/new-file cues should select write_file.
- Future RED assertion: For this row, selected_tool becomes patch and expected_score - top_confusing_score > 0.02.
- Safe fix direction: Add reusable patch-vs-write_file discrimination around targeted replacement, unique old_string, preserve context, and whole-file overwrite boundaries.

### `tool-selection-016` — `wrong_tool_selected`

- Excerpt: Read only lines 120 through 180 of gateway/run.py.
- Expected / selected: `read_file` / `search_files`
- Score margin: `-0.005`; cue coverage: `1.0`
- Finding: Wrong tool selected: search_files narrowly outranks read_file for a line-range read request.
- Hypothesis: Line-range and read-only excerpt cues should prefer read_file over search_files.
- Future RED assertion: For this row, selected_tool becomes read_file and expected_score - top_confusing_score > 0.02.
- Safe fix direction: Strengthen read_file cues for explicit line ranges, first/last lines, and file excerpt reading; keep search_files for discovery/pattern search.

### `tool-selection-028` — `wrong_tool_selected`

- Excerpt: Install the project dependencies and run the npm build script.
- Expected / selected: `terminal` / `execute_code`
- Score margin: `-0.0445`; cue coverage: `0.6`
- Finding: Wrong tool selected and weaker cue coverage: execute_code outranks terminal for dependency installation and npm build execution.
- Hypothesis: Package-manager install/build commands are shell/process actions and should route to terminal, while execute_code remains for multi-call Hermes-tool scripting and data reduction.
- Future RED assertion: For this row, selected_tool becomes terminal, cue_coverage >= 0.9, and expected_score - top_confusing_score > 0.02.
- Safe fix direction: Strengthen terminal cues for dependency install, npm/build scripts, subprocess/system state, and package managers without broadening to arbitrary Python processing.

## Future RED regression plan

1. `test_phase5_tool_selection_current_candidate_report_clears_minimum_threshold` — should fail now because current `selection_accuracy=0.8889 < 0.90`.
2. `test_phase5_tool_selection_critical_rows_have_correct_tool_and_margin` — should fail now on the five current weak rows.
3. `test_phase5_tool_selection_closeout_candidate_reaches_all_pass_before_soak_resume` — optional stricter closeout before any P1 soak resume.

## Explicitly not performed

No source code was modified. No test code was written. No RED test was executed. No candidate report was regenerated. No provenance/performance/auto-triage/scheduler pipeline was rerun as new evidence. Cron job `b24aca09f168` was not resumed and remaining fires were not run. Optimizer, automatic PR loop, production loop, auto-merge/deploy, and network/model/API budget spend remain OFF.

## Recommended next step

Path-limited publish this local triage/RED-plan artifact to PR #108 if Sunwoo approves. After that, a separate approval can authorize actual RED-test authoring without source fix yet.
