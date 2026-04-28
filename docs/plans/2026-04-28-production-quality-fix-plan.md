# Production Quality Fix Plan

> For Hermes: use this plan to harden hermes-content and the self-evolution loop against the recurring failures found in session review.

Generated: 2026-04-28T14:51:16Z
Owner profile: hermes-content / video-production
Primary repo: /tmp/hermes-agent-self-evolution-readonly
Protected files: do not modify SOUL.md, USER.md, MEMORY.md, profile config, credentials, or upstream remotes without explicit approval.

Goal: Make content/video work production-grade by replacing claim-based completion with verifiable proof packages, enforcing Mac/runtime preflight, and converting historical failures into self-evolution eval pressure.

Architecture:
- Skills define the operating contract: proof package, runtime preflight, lifecycle completion gates.
- Self-evolution repo stores historical failure traces as executable eval examples.
- Every content package must pass structural evidence, runtime evidence, visual/human critique, and safety gates before completion.

Tech stack:
- Hermes skills: video-editing-cost-safe-stack, video-project-lifecycle.
- Self-evolution: JSONL attempt traces, SQLite/control-plane loop, rubric gates.
- Verification: pytest, compileall, git diff --check, skill readback.

---

## STOP-gate definition

A content/video task is not complete unless all are true:

1. Required package artifacts exist and paths are literal.
2. proof-assets/ contains inspectable visual evidence.
3. runtime_check.json proves the correct host/runtime boundary.
4. timeline_manifest.json or edit_spec_v1.json anchors audio/timing truth.
5. qa_verdict.json includes a human-facing quality verdict, not only file-count checks.
6. manifest.json records run_id, stage, generated_at, owner_profile, artifact paths, and hashes.
7. Any claimed visual technology has at least one proof artifact or is explicitly marked not used.

Failure of any item = HOLD.

---

## Task list

### Task 1: Patch video stack production contract
Objective: Make proof-assets and runtime preflight mandatory in video-editing-cost-safe-stack.
Files:
- Modify skill: /root/.hermes/profiles/content/skills/media/video-editing-cost-safe-stack/SKILL.md
Steps:
1. Add a Production Quality Gate section.
2. Define required package files.
3. Define runtime_check.json fields.
4. Define visual-tech claim rule.
5. Verify with skill_view.
Acceptance:
- Skill explicitly says empty proof-assets/ means HOLD.
- Skill explicitly says no render/export completion without runtime_check.json and Mac evidence.

### Task 2: Patch lifecycle completion gate
Objective: Stop project state from reaching complete without proof and QA verdict.
Files:
- Modify skill: /root/.hermes/profiles/content/skills/media/video-project-lifecycle/SKILL.md
Steps:
1. Add proof-assets and QA fields to project.json schema.
2. Update phase transition rules so rendering -> complete requires render + manifest + proof assets + qa_verdict pass.
3. Add pitfalls for feature-card decks and mechanical QA.
4. Verify with skill_view.
Acceptance:
- Lifecycle says proof assets and qa verdict block completion.

### Task 3: Add historical production-failure eval traces
Objective: Feed the self-evolution loop with real hermes-content failure patterns.
Files:
- Create: examples/content-production/failures.jsonl
- Create: tests/core/test_content_production_evals.py
Steps:
1. Write failing test first for expected dataset shape and categories.
2. Run targeted test; expect missing file failure.
3. Add JSONL traces for historical failures:
   - proof-layer failure
   - Mac/runtime boundary failure
   - claimed visual stack not used
   - skipped human critique
   - path/host confusion
4. Run targeted test; expect pass.
Acceptance:
- Dataset is valid JSONL.
- All traces status=failure.
- Secret scanner passes.
- Required categories are present.

### Task 4: Verify full repo quality after dataset addition
Objective: Ensure self-evolution repo remains green.
Commands:
- .venv/bin/python -m pytest tests -q
- .venv/bin/python -m compileall -q evolution tests
- git diff --check
- git status --short
Acceptance:
- Full tests pass.
- Compile passes.
- Diff check passes.

### Task 5: Commit repo changes
Objective: Preserve the eval/plan hardening work as a local git milestone.
Commands:
- git add docs/plans/2026-04-28-production-quality-fix-plan.md examples/content-production/failures.jsonl tests/core/test_content_production_evals.py
- git commit -m "test: add content production failure evals"
Acceptance:
- Clean working tree after commit.

### Task 6: Next execution wave after this checkpoint
Objective: Move from documented gates to executable package validators.
Planned files/classes:
- A profile-local content-package validator skill or script.
- Optional self-evolution target skill eval dataset import run.
- Runtime preflight script for Mac media host.
Acceptance:
- A command can validate a content package and emit qa_verdict.json.
- Real failed sessions can be imported into hermes-evolve loop once without manual JSON rewriting.

---

## Production-quality target state

The system reaches production quality when:

- content packages cannot pass on file existence alone;
- Mac/VPS/runtime boundary is checked before production;
- visual proof is mandatory for human-facing claims;
- self-evolution has real failure evals from our own bad sessions;
- safe review bundles remain the only promotion path;
- no upstream push/merge happens without explicit user approval.
