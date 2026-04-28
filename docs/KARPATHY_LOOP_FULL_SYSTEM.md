# Hermes Agent Self-Evolution: Karpathy Loop Full System

Status: production blueprint
Repo reviewed: NousResearch/hermes-agent-self-evolution
Local review path: /tmp/hermes-agent-self-evolution-readonly
Review date: 28/04/2026

## Executive verdict

The repo has the right idea but is still a Phase-1 prototype.

It currently contains:
- a skill evolution CLI wrapper
- synthetic/golden/session-derived dataset helpers
- a DSPy skill wrapper
- shallow fitness and constraint validators
- output artifacts for baseline/evolved/metrics

It does not yet contain the full self-improvement system promised by the README/PLAN:
- no durable run database
- no content-addressed artifact lineage
- no actual Hermes Agent batch_runner execution loop
- no per-candidate hard gates inside optimization
- no wired pytest/benchmark gates
- no PR/deployment builder
- no resumable jobs
- tool/prompt/code/continuous phases are placeholders

The correct system is an eval-driven improvement flywheel:

attempt tasks -> record traces -> detect failures -> crystallize evals -> propose targeted changes -> run baseline vs candidate -> gate brutally -> ship only versioned, reversible improvements -> monitor -> repeat

That is the Karpathy-style loop applied to Hermes. Not mystical self-awareness. Just compounding eval discipline with a nice lab coat.

## The loop

### 1. Attempt

Hermes performs a real user task with the current production configuration.

Inputs:
- user request
- system/developer/profile prompts
- loaded skills
- available tools
- memory retrievals
- model/router config
- repo/workdir state

Outputs:
- final response
- tool calls and tool results
- errors
- cost/latency
- files changed
- user correction/satisfaction signal

### 2. Record trajectory

Every task attempt creates a replayable trace.

Minimum trace fields:
- trace_id
- session_id
- task_input
- final_output
- model/provider
- prompt_version
- skill_versions
- tool_schema_versions
- memory_snapshot_refs
- tool_calls
- file diffs/checksums
- exit status
- user feedback
- cost_usd
- latency_ms

### 3. Detect failure

Failure signals:
- explicit user correction
- failed tests
- missing required artifact
- hallucinated file/tool result
- bad tool choice
- schema/format violation
- unsafe side-effect attempt
- no evidence for completion claim
- high cost/latency spike
- benchmark regression

Failure categories:
- prompt failure
- skill missing or stale
- tool description/tool routing failure
- tool implementation bug
- memory policy failure
- planning failure
- evaluator/rubric failure
- safety/permission failure
- model routing failure

### 4. Crystallize eval

No behavior change without an eval.

Each meaningful failure becomes an eval with:
- source_trace_id
- target component
- task prompt
- environment assumptions
- required behavior
- forbidden behavior
- deterministic checks where possible
- LLM judge rubric where needed
- severity
- split assignment: train/val/holdout/regression

### 5. Propose minimal improvement

Patch the smallest thing likely to fix the eval.

Targets:
- SKILL.md body
- tool description text
- prompt section
- tool implementation code
- memory policy
- model/router config
- evaluator rubric

Rule: do not bundle unrelated changes. If the candidate changes prompts, skills, tools, and memory all at once, it is not evolution; it is a garage explosion with a changelog.

### 6. Evaluate baseline vs candidate

Run both baseline and candidate against:
- new failure evals
- target-specific evals
- global regression evals
- safety evals
- tool-use evals
- memory behavior evals
- hidden holdout evals for final promotion

The candidate must be compared against the pinned current version, not vibes.

### 7. Gate

Hard gate failures reject the candidate.

Core hard gates:
- syntax/structure valid
- no secret leakage
- semantic preservation where required
- pytest pass when code touched
- benchmark non-regression
- critical/safety evals 100% pass
- holdout no regression
- target eval improvement above threshold
- cost/latency within budget
- human review for high-risk changes

### 8. Ship through PR only

The system may propose changes.
The eval system may measure them.
Gates may approve them.
Humans merge them.
Production only receives versioned, tested, reversible artifacts.

### 9. Monitor and rollback

After promotion:
- watch user correction rate
- tool error rate
- cost/latency
- regression evals
- safety events
- memory contamination

Rollback triggers:
- critical eval failure
- safety event
- high correction spike
- tool side-effect failure
- cost spike without quality gain

## System architecture

### Runtime plane

The thing being improved.

Components:
- Hermes Agent core loop
- prompt registry
- skill registry
- tool registry
- memory system
- model/router layer
- gateway/platform adapters

### Telemetry plane

The evidence source.

Components:
- trajectory recorder
- outcome logger
- file/diff manifest collector
- tool-call ledger
- cost/latency tracker
- user feedback collector

### Eval plane

The engine room. Everything else is garnish.

Components:
- eval registry
- dataset store
- deterministic graders
- LLM-as-judge graders
- batch_runner adapter
- holdout manager
- scorecard renderer

### Failure mining plane

Turns messy sessions into clean evals.

Components:
- failure classifier
- failure clusterer
- severity scorer
- eval crystallizer
- duplicate detector
- privacy/secret scrubber

### Improvement plane

Creates candidates, never ships directly.

Components:
- target scanner
- improvement planner
- candidate generator
- GEPA engine
- MIPRO fallback engine
- Darwinian code engine behind external CLI boundary
- candidate artifact writer

### Gating plane

Prevents clever nonsense from reaching production.

Components:
- constraint gate
- semantic drift gate
- pytest gate
- benchmark gate
- safety gate
- cost/latency gate
- holdout gate
- human review gate

### Release plane

Moves approved changes safely.

Components:
- version registry
- PR builder
- changelog generator
- canary manager
- rollback manager
- evolution ledger

## Repository module layout

Recommended package structure:

```text
evolution/
  cli.py
  config.py
  logging.py

  db/
    __init__.py
    models.py
    session.py
    migrations/
    repositories.py
    runs.py

  artifacts/
    __init__.py
    store.py
    manifest.py
    hashing.py

  repos/
    __init__.py
    discovery.py
    git.py
    snapshots.py
    targets.py

  targets/
    __init__.py
    base.py
    skills.py
    tools.py
    prompts.py
    code.py

  datasets/
    __init__.py
    schema.py
    synthetic.py
    golden.py
    session_importers.py
    redaction.py
    validators.py

  engines/
    __init__.py
    base.py
    gepa_engine.py
    mipro_engine.py
    darwinian_engine.py

  evaluators/
    __init__.py
    runner.py
    batch_runner_adapter.py
    judge.py
    metrics.py
    significance.py

  gates/
    __init__.py
    constraints.py
    semantic.py
    pytest_gate.py
    benchmark_gate.py
    security.py
    cost_latency.py

  orchestrator/
    __init__.py
    state_machine.py
    run_manager.py
    candidate_manager.py
    selector.py
    resume.py

  reports/
    __init__.py
    renderer.py
    pr_builder.py

  skills/
    skill_module.py
    evolve_skill.py       # compatibility wrapper, not the god-script
```

Refactor current files:
- evolution/core/config.py -> keep or expand into config.py
- evolution/core/dataset_builder.py -> datasets/synthetic.py + schema.py
- evolution/core/external_importers.py -> datasets/session_importers.py + redaction.py
- evolution/core/fitness.py -> evaluators/judge.py + evaluators/metrics.py
- evolution/core/constraints.py -> gates/constraints.py
- evolution/skills/evolve_skill.py -> orchestrator/run_manager.py + thin CLI wrapper

## Persistent data model

MVP: SQLite.
Production later: Postgres.
Blobs live in content-addressed storage; DB stores metadata and hashes.

### repositories

Tracks target repos.

Fields:
- id
- name
- url
- local_path
- default_branch
- created_at
- updated_at

### repo_snapshots

Pins exact source state.

Fields:
- id
- repository_id
- git_sha
- branch
- dirty
- diff_sha256
- created_at

### targets

Anything evolvable.

Fields:
- id
- repository_id
- target_type: skill | tool_description | prompt_section | code_file
- name
- file_path
- selector
- baseline_artifact_id
- metadata_json
- created_at

### artifacts

Immutable files/blobs.

Fields:
- id
- target_id
- kind: baseline | candidate | dataset | prompt | patch | report | trace | benchmark_result
- content_sha256
- storage_uri
- size_bytes
- mime_type
- parent_artifact_id
- metadata_json
- created_at

### datasets

Versioned eval sets.

Fields:
- id
- target_id
- source: synthetic | sessiondb | golden | mixed
- version
- artifact_id
- split_spec_json
- pii_scan_status
- secret_scan_status
- example_count
- created_at

### eval_examples

Individual eval rows.

Fields:
- id
- dataset_id
- split: train | val | holdout | regression
- source
- task_input
- expected_behavior
- difficulty
- category
- source_ref_hash
- metadata_json
- created_at

### runs

One optimization run.

Fields:
- id
- target_id
- repository_snapshot_id
- baseline_artifact_id
- dataset_id
- engine: gepa | mipro | darwinian
- status: pending | running | evaluating | gated | succeeded | failed | cancelled
- config_json
- seed
- self_evolution_git_sha
- started_at
- completed_at
- cost_usd
- error

### candidates

Each proposed artifact.

Fields:
- id
- run_id
- artifact_id
- generation
- parent_candidate_id
- optimizer_trace_artifact_id
- mutation_summary
- status: generated | evaluated | rejected | gated | selected
- created_at

### evaluations

Scores for baseline/candidates.

Fields:
- id
- run_id
- candidate_id
- artifact_id
- split
- evaluator: heuristic | llm_judge | batch_runner | deterministic
- score
- metric_json
- cost_usd
- latency_ms
- trace_artifact_id
- created_at

### constraint_results

Gate details.

Fields:
- id
- run_id
- candidate_id
- constraint_name
- severity: hard | soft
- passed
- message
- details_artifact_id
- created_at

### benchmark_results

Regression evidence.

Fields:
- id
- run_id
- candidate_id
- benchmark_name
- baseline_score
- candidate_score
- regression_pct
- passed
- details_artifact_id
- created_at

### pr_submissions

Release trail.

Fields:
- id
- run_id
- candidate_id
- repository_id
- branch_name
- commit_sha
- pr_url
- status: created | open | merged | closed
- created_at
- updated_at

### approvals

Human gate.

Fields:
- id
- run_id
- approver
- decision: approve | reject
- comment
- created_at

## Artifact layout

```text
.evolution/
  config.yaml
  evolution.db

  blobs/
    sha256/
      ab/
        abc123...md
        abc123...json
        abc123...jsonl
        abc123...patch

  datasets/
    skills/
      github-code-review/
        dataset_<uuid>/
          manifest.json
          train.jsonl
          val.jsonl
          holdout.jsonl
          scan_report.json

  runs/
    run_<uuid>/
      manifest.json
      config.resolved.yaml
      repo_snapshot.json

      baseline/
        artifact.md
        metrics.json

      candidates/
        cand_<uuid>/
          artifact.md
          mutation_summary.txt
          optimizer_trace.jsonl
          eval_train.json
          eval_val.json
          eval_holdout.json
          constraints.json
          benchmark_tblite_fast.json
          diff.patch

      selected/
        artifact.md
        diff.patch
        report.md
        pr_body.md

      logs/
        orchestrator.log
        lm_calls.jsonl
        costs.json

  reports/
    run_<uuid>.md
    run_<uuid>.html

  cache/
    lm/
    batch_runner/
```

Invariant: every artifact is immutable and hash-addressed. No magic latest file as source of truth. Latest is a DB query, not a filesystem superstition.

## CLI surface

Add production entrypoint:

```toml
[project.scripts]
hermes-evolve = "evolution.cli:main"
```

Commands:

```bash
hermes-evolve init \
  --db sqlite:///.evolution/evolution.db \
  --artifact-root .evolution

hermes-evolve repo add hermes-agent \
  --path /opt/hermes-agent

hermes-evolve repo snapshot hermes-agent

hermes-evolve targets scan \
  --repo hermes-agent \
  --type skill

hermes-evolve dataset build \
  --target skill:github-code-review \
  --source synthetic \
  --size 60

hermes-evolve dataset build \
  --target skill:github-code-review \
  --source sessiondb \
  --sources claude-code,copilot,hermes

hermes-evolve dataset build \
  --target skill:github-code-review \
  --source golden \
  --path datasets/skills/github-code-review

hermes-evolve run skill \
  --target github-code-review \
  --dataset dataset_<uuid> \
  --engine gepa \
  --iterations 10 \
  --optimizer-model openai/gpt-4.1 \
  --eval-model openai/gpt-4.1-mini \
  --gates size,growth,semantic,pytest \
  --budget-usd 10

hermes-evolve eval \
  --run run_<uuid> \
  --candidate cand_<uuid> \
  --split holdout

hermes-evolve gate \
  --run run_<uuid> \
  --candidate cand_<uuid> \
  --pytest \
  --benchmark tblite-fast

hermes-evolve report \
  --run run_<uuid> \
  --format md

hermes-evolve pr create \
  --run run_<uuid> \
  --candidate best \
  --base main

hermes-evolve runs list
hermes-evolve runs show run_<uuid>
hermes-evolve runs resume run_<uuid>
hermes-evolve runs cancel run_<uuid>
```

## Mandatory gates

### Eval-first gate

Behavior-changing patch must link to at least one eval.

Exception:
- docs-only
- typo-only
- explicitly approved emergency patch

### Reproducibility gate

A run must be reproducible from:
- repo snapshot
- target artifact hash
- dataset hash
- run config
- model IDs
- random seed
- candidate artifact hash

### Secret/PII gate

Block if candidate or dataset contains secrets or inappropriate private data.

### Structure gate

For skills:
- full SKILL.md has YAML frontmatter
- name and description exist
- body is non-empty
- body remains within size/growth limits

Important current bug: the repo validates skill["body"] as artifact_type="skill", but the validator expects frontmatter. Fix by validating full reassembled SKILL.md for structure and body for size/growth.

### Semantic preservation gate

Candidate must preserve target purpose.

Use:
- embedding similarity for cheap screen
- LLM entailment/rubric for final gate
- human review for critical skills/prompts

### Evaluation gate

Promotion requires:
- target eval improvement above threshold
- no holdout regression
- no global regression
- critical evals 100% pass
- safety evals 100% pass

### Pytest gate

If code or production skill packages are touched:
- run pytest in the target hermes-agent repo or isolated worktree
- do not make this opt-in for release path
- allow --skip-tests only for local experiments, never PR promotion

### Benchmark gate

Use TBLite/YC-Bench/TerminalBench-style suites as budget allows.

Rules:
- expensive benchmarks can be opt-in during experiments
- benchmark gate is mandatory before high-risk prompt/tool/code promotion
- threshold defaults: max 2% regression for non-critical, 0% for safety/permission behavior

### Cost/latency gate

Reject candidates that increase cost/latency without quality gain.

Track:
- total cost per eval run
- avg cost per task
- p50/p95 latency
- tool-call count
- expensive-model routing ratio

### Human review gate

Required for:
- prompt sections
- memory policy
- tool schemas for side-effect tools
- code changes
- anything touching safety/permissions
- any run with weak/LLM-only evidence

## Corrected skill evolution flow

The current skill evolution path must be repaired before adding more phases.

Required behavior:

1. Load full SKILL.md.
2. Split frontmatter and body.
3. Persist baseline artifact.
4. Build/reuse dataset.
5. Generate candidate skill body variants.
6. Reassemble full SKILL.md for each candidate.
7. Run body constraints and full-file structure constraints.
8. Evaluate baseline vs candidate using actual Hermes Agent where possible.
9. Use LLM-as-judge or deterministic grader, not keyword overlap alone.
10. Run holdout comparison.
11. Persist all scores/gates/artifacts.
12. Select best valid candidate.
13. Create report and PR branch only after gates pass.

## Fix list for current repo

Priority 0: truth in docs
- README must say Phase 1 prototype, not full self-evolution, until gates/PR/benchmarks exist.

Priority 1: fix broken/weak Phase 1
- fix skill body vs full SKILL.md constraint mismatch
- wire --run-tests or replace with --skip-tests for release path
- use sys.executable for pytest runner
- use LLMJudge or real rubric metric, not keyword overlap only
- persist optimizer/candidate traces
- label MIPRO fallback honestly

Priority 2: make runs durable
- add SQLite DB
- add content-addressed artifact store
- add run/candidate/evaluation/constraint records
- add reports from DB, not loose files

Priority 3: run real Hermes evals
- add batch_runner adapter
- execute with target skill in temp checkout/worktree
- capture actual tool trajectories
- store replayable traces

Priority 4: release path
- add PR builder
- generate branch/commit/PR body
- require human review
- add rollback metadata

Priority 5: expand target types
- tool descriptions
- prompt sections
- code files behind Darwinian external CLI
- continuous monitor

## MVP milestones

### M0: Stabilize current prototype

Acceptance:
- dry-run works
- synthetic dataset build works
- skill structure/body constraint bug fixed
- compileall passes
- existing tests pass in environment with pytest

### M1: Persistence foundation

Acceptance:
- `.evolution/evolution.db` created
- artifact store writes SHA256-addressed blobs
- repositories, snapshots, targets, datasets, runs, candidates, evaluations, constraints are recorded
- every run has a manifest

### M2: Dataset productionization

Acceptance:
- synthetic/golden/session datasets share one schema
- secret scan runs before persistence
- dataset manifests include source, count, hashes, model, timestamp
- datasets can be reused by ID

### M3: Resumable skill evolution

Acceptance:
- run id exists before optimizer starts
- candidates persist incrementally
- interrupted runs can resume or fail cleanly
- baseline/candidate holdout comparison is recorded
- report includes diff, metrics, cost, gates, dataset id, repo SHA

### M4: PR workflow

Acceptance:
- selected candidate can be applied to an isolated branch
- commit created with evolution metadata
- PR body includes eval/gate evidence
- no direct commits to main

### M5: Real eval harness

Acceptance:
- batch_runner adapter runs actual Hermes tasks
- tool traces captured
- deterministic and LLM graders supported
- holdout protected from candidate generator

### M6: Expand beyond skills

Acceptance:
- tool description target parser/reassembler
- prompt section parser/reassembler
- code patch engine in isolated worktree
- same run/gate/report pipeline handles all target types

### M7: Continuous loop

Acceptance:
- recent failures become candidate evals
- budget controls exist
- scheduled jobs can propose runs
- release monitor can trigger rollback

## First sprint tasks

1. Add artifact store.
2. Add SQLite run DB.
3. Add CLI init/repo/targets/runs commands.
4. Fix skill constraint bug.
5. Refactor evolve_skill.py into RunManager.
6. Persist baseline/dataset/run/candidate/eval/constraint artifacts.
7. Replace keyword-only fitness with rubric judge option.
8. Add report renderer.
9. Add PR builder.
10. Only then start tools/prompts/code phases.

## Strategic rule

Do not let the system auto-edit production Hermes.

The safe version is:
- Hermes proposes
- evals measure
- gates decide
- humans merge
- git rolls back

Anything else is not a self-evolution system. It is an expensive way to create a haunted toaster.
