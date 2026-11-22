# KAFO-EVO-001 — Standalone Adaptive Hermes: Research Sprint

## Objective

Demonstrate a reproducible, headless learning loop that selects between
strategies based on task context and independently scored outcomes.
Preserve the existing Nous Research GEPA skill evolution pipeline.

## Architecture and terminology

- **Monitor:** capture task outcome, cost, time, external verifier result.
- **Analyse:** identify quality, constraints and failure type.
- **Plan:** contextual epsilon-greedy policy (first baseline); fixed, random
  and LinUCB policies behind the same interface in `evolution/adaptive/policies.py`.
- **Legitimise:** fixed reward calculation and critical-violation handling.
- **Execute:** `evolution/adaptive/executor.py` — `Executor` protocol,
  `HermesExecutor` (documented Hermes Python API, opt-in) and deterministic
  `FakeExecutor`. See "Hermes executor (gate 1)" below.
- **Knowledge:** SQLite outcomes, context/action counts and estimated reward.
- **Slow evolution:** upstream GEPA, later connected through candidate validation.

This is a MAPE-K-inspired implementation with an explicit adaptation gate,
not a new RL training framework. The initial bandit is **not LinUCB** and
does not update any LLM model weights.

## Delivered in initial branch

- Pure Python contextual epsilon-greedy strategy selector.
- Explicit action-selection propensity for future policy evaluation.
- Utility = verified quality - 0.10 * min(relative cost, 2) - 0.05 * min(relative latency, 2).
- Critical observed violation yields a disqualifying -1 reward, strictly below any valid outcome.
- Every strategy tried once per context before epsilon-greedy selection.
- NaN, +/-inf, booleans and non-numeric rewards rejected before estimates change.
- Local SQLite experience records with policy version, skill version and UTC timestamp;
  older databases migrated in place (legacy rows keep NULL provenance).
- Dependency-free, seeded synthetic demonstration.
- Unit tests for policy, reward, invalid actions and persistence.

## Reward range

| Outcome | Reward |
|---|---|
| Valid (no violation) | `quality - 0.10*min(cost_ratio, 2) - 0.05*min(time_ratio, 2)`, in **[-0.30, 1.0]** |
| Critical violation | **-1.0** (`VIOLATION_REWARD`), always below every valid outcome |

`RATIO_CAP = 2.0`. Cost or latency beyond twice the reference adds no further
penalty, so an expensive success can never score as low as a violation.
`score_outcome` rejects non-finite or boolean fields, and rejects weights for
which `(cost_weight + time_weight) * RATIO_CAP >= 1` (a valid outcome could
then tie a violation).

## Run a synthetic smoke test

From repository root:

```bash
python -m evolution.adaptive.cli --episodes 300 --seed 7
python -m pytest tests/test_adaptive_loop.py -q
```

The simulated reward distributions in the CLI are **hypothetical**.
Any observed learning in this simulator is NOT evidence that Hermes
improves in real code-review tasks.

## Validation evidence (2026-11-18, synthetic only)

| Check | Result |
|---|---|
| Targeted `tests/test_adaptive_loop.py`, initial branch (5 tests) | 5 passed (Python 3.14, pytest 9.0.3) |
| Initial CLI, `--episodes 300 --seed 7` | average utility 0.771 |
| Targeted suite after corrections (40 tests) | 40 passed (Python 3.12.13, isolated venv) |
| Full repository suite, `pytest -q` | 185 passed (Python 3.12.13, dspy 3.5.0, isolated venv) |
| CLI after corrections, `--episodes 300 --seed 7` | average utility 0.760, exit 0 |

The 0.771 -> 0.760 change comes from the per-context initial sweep altering the
random sequence; it is not a quality regression measurement. In the `complex`
context the post-correction run settled on `tool_assisted` (n=75) after a poor
single early sample of `structured` (n=3), although `structured` has the higher
simulated success rate. This is a known weakness of plain epsilon-greedy and is
what the pending LinUCB/random/fixed comparison must measure.

**These are results from a hypothetical simulator. They are not evidence that
Hermes improves on any real task.**

## Remaining Sprint 01 acceptance gates

- [x] Verify upstream dependencies and run the complete repository test suite
      (185 passed, Python 3.12 + dspy 3.5.0; Python 3.14 has no dspy install here).
- [x] Implement a Hermes executor adapter using an actually supported API
      (`AIAgent.run_conversation`, contract-checked against hermes-agent source;
      plus a 4-case live smoke run on 2026-11-22, see "Hermes executor (gate 1)").
- [x] Build labelled Python-review benchmark with ground-truth defect IDs.
      (`datasets/python_review/v1`: 48 hand-written synthetic cases, 50 defects,
      8 clean controls; train 28 / validation 10 / sealed test 10; SHA-256
      manifest pinned in `evolution/adaptive/benchmark.py`. Labels are
      single-author and not yet independently audited.)
- [x] Build independent verifier with reproducible precision/recall scoring
      (`evolution/adaptive/verifier.py`, stdlib-only, no LLM judge; 61 unit
      tests on small synthetic inline fixtures, full suite 246 passed on
      GitHub Actions. Run on the gate 2 benchmark via `evolution/adaptive/evaluation.py`:
      a ground-truth review scores 1.0 and an empty review 0.0 (micro P/R/F1) on
      train and validation; synthetic data, not a Hermes measurement).
- [x] Compare fixed, random, epsilon-greedy and LinUCB policies.
      Comparison implemented and run (PR #5). Result: no improvement over the
      best fixed strategy on validation (CI includes 0); adaptive policies beat
      random. Synthetic data only. (Owner decision, 2026-11-22; see
      "Policy comparison (gate 4)".)
- [x] Preserve train / validation / sealed test separation.
      The adaptive loop (`core.choose_strategy`), its CLI (`cli.main`) and the
      gate 4 harness (`compare.compare`) run inside `benchmark.optimisation_phase()`,
      which blocks loading or opening the sealed split in-process (audit hook; also
      through symlinks and dataset copies). `verifier.load_cases` refuses sealed
      cases; only `evaluation.evaluate_sealed_test(final_evaluation=True)` unseals.
      Tests: a static scan of `evolution/` (no benchmark path literals or loader
      internals, no other unsealing, all three entry points guarded) and runtime
      tests showing the loop and `compare()` fail if they touch the sealed split.
      Limits: covers this Python process only, not subprocesses; directory listing
      of `sealed/` is not blocked.
- [x] Connect GEPA skill-candidate generation to the strategy registry
      (`evolution/adaptive/registry.py`, schema `candidate-registry/1`;
      `evolution/adaptive/gepa_adapter.py` registers `evolve_skill` output as
      `proposed` only, never admitted or deployed; only admitted candidates become
      strategies, with toolsets inherited from a fixed base strategy. Tested on a
      **stubbed GEPA output directory only; no live GEPA/LLM run**. Admission is
      the next gate).
- [x] Require candidate-vs-baseline validation before candidate admission.
      `evolution/adaptive/admission.py` (synthetic tests only, FakeExecutor):
      candidate and baseline run on the same validation cases, with the same
      executor factory and seeds (0, 1, 2), inside `optimisation_phase()`. The
      verifier scores them. Admission needs a mean utility gain >= 0.05, a 95%
      paired bootstrap CI above 0, and zero violations; otherwise it rejects and
      logs every failed rule. Thresholds are a frozen `ADMISSION_POLICY`, not
      arguments of `admit()`; its SHA-256 is pinned in tests and recorded in every
      decision. `review_registered` reviews a proposed gate 5 candidate against its
      parent, using the evaluation-only `registry.evaluation_strategy` (`eval:` IDs,
      tools from base, no status change), and records the decision via
      `registry.transition` (GEPA provenance is never read). Required by:
      (1) `CandidateRegistry.transition(..., "admitted")` accepts only evidence with
      `admitted=true`, the same `candidate_id`, and a `policy_sha256` registered by
      `admission` at import (`register_admission_policy`). With nothing registered,
      nothing can be admitted, and the registry does not import scoring code.
      (2) Static scans: only `review_registered` calls `transition`, and only
      `admission` registers policies. (3) `admitted_catalogue` re-checks the full
      decision (`decided_by`, margin, CI, violations). Limit: these checks validate
      the evidence, not who wrote it, so a person with direct registry or database
      access could still copy a genuine decision.
- [x] Implement policy/skill checkpoint and rollback.
      (`evolution/adaptive/checkpoint.py`, PR #9; 45 offline tests, full suite
      475 passed on GitHub Actions. See "Checkpoint and rollback (gate 7)".
      Wired into the real admission path: `admission.review_with_rollback`
      checkpoints before `review_registered` admits, re-validates the admitted
      candidate's catalogue strategy, and on a drop beyond `ROLLBACK_MARGIN` or a
      critical violation retires it in the registry and rolls the checkpoint
      store back, logging both (`tests/test_admission_rollback.py`, real registry
      and admission, fake executor). Skill rollback = retire (terminal). Synthetic
      data only; no live run of this path yet.)
- [x] Publish quality, utility, cost, latency, uncertainty and safety report.
      (`reports/kafo-evo-001/report.md` and `report.json`, generated by
      `python -m evolution.adaptive.report` from merged gates 2-7; seeded,
      offline, run inside `optimisation_phase()`; a test checks the committed
      files equal fresh output. **Synthetic**, apart from the 4-case live smoke
      run, which is labelled a plumbing check. Headline, derived from the
      verdicts: no adaptive policy beats the best fixed strategy on validation.
      The one-time final test evaluation **ran once on 2026-11-22**, after owner
      approval, as the pre-registered paired run
      (`final_evaluation.run_paired_final_evaluation`). The run-once ledger is
      `reports/kafo-evo-001/final_test_evaluation.json`, and the scores and paired
      CIs are in the report. Primary (strict micro-F1, candidate minus baseline):
      0.302 [0.157, 0.462]. That meets the pre-registered rule on **synthetic**
      hand-written cases only; it is not evidence of a real Hermes improvement.
      See "Pre-registered analysis plan" below.)

## Hermes executor (gate 1)

### Supported Hermes interfaces (verified, not invented)

Checked in NousResearch/hermes-agent at `caafe5389a` (2026-11-19):

| Interface | Where it is defined / documented | Used? |
|---|---|---|
| `run_agent.AIAgent(...)` + `run_conversation(user_message, task_id)` -> dict | `run_agent.py` (`class AIAgent`), `agent/turn_facade.py` (`TurnFacadeMixin.run_conversation`), result keys built in `agent/turn_finalizer.py`; guide `website/docs/guides/python-library.md` | **Yes** |
| `hermes -z PROMPT --usage-file PATH` | `website/docs/reference/cli-commands.md`, `hermes_cli/oneshot.py` | No: loads all tools and auto-bypasses approvals, routes through the gateway daemon |
| ACP, TUI gateway JSON-RPC, OpenAI-compatible API server | `website/docs/developer-guide/programmatic-integration.md` | No: long-lived servers, more surface than one turn needs |

### How a run works

`HermesExecutor.execute(task, strategy)` writes a JSON request to a temp dir
and runs `evolution/adaptive/hermes_bridge.py` as
`<hermes python> -c <bridge> request.json result.json` with the hermes-agent
checkout as working directory (as the Python-library guide requires). The
bridge calls `AIAgent(**kwargs).run_conversation(...)`, then `close()`, and
writes a JSON summary. The project itself gains no dependency.

Fixed by the bridge, not settable by a strategy:
`quiet_mode=True`, `skip_memory=True`, `skip_context_files=True`,
`skip_background_review=True`, `save_trajectories=False`,
`checkpoints_enabled=False`. Tools: `enabled_toolsets` is the strategy's
allowlisted toolsets (default `[]`, which Hermes treats as *no tools*; only
`None` means all). Only `todo` is allowlisted. Terminal, code execution, file,
browser, skills, memory, delegation, cron, kanban, session search and
connections are always passed as `disabled_toolsets`.

Budgets (`Budget`: iterations, run seconds, max tokens, max cost) belong to
the executor, so every compared strategy gets the same limits. The subprocess
is killed at `timeout_s + startup_grace_s`.

### Outcome record

`ExecutionOutcome`: output, completed, latency (agent turn), wall time,
`cost_usd` (Hermes `estimated_cost_usd`; **`None` when `cost_status` is
`unknown`**, never silently 0), tokens, API calls, tool calls, error, and
`violation_reasons`. Violations: a tool call outside the strategy allowlist,
a Hermes guardrail `block`/`halt`, or known cost above `max_cost_usd`.
`to_reward(outcome, quality=..., reference_cost_usd=..., reference_latency_s=...)`
builds `core.Reward`; quality must come from the independent verifier, and
an unknown cost is refused. `loop_executor(...)` plugs an executor into
`choose_strategy`. Reward logic in `core.py` is unchanged.

### Limits and what is still pending

- `HermesExecutor` refuses to run unless constructed with `allow_live=True`;
  a live run spends LLM credit from the local Hermes configuration.
- Offline evidence: unit tests, a real subprocess run of the bridge against a
  stub `run_agent`, and an AST contract test against the local hermes-agent
  source. The contract test is skipped on CI, which has no checkout; it passed
  locally against `caafe5389a` on 2026-11-22.
- `model` must be passed explicitly. `AIAgent(model="")` does not resolve the
  configured model: the provider failed with `model is required`. A live
  `HermesExecutor` with no model now raises `ValueError` before any call.
- `HERMES_PYTHON` must be Hermes' own PM interpreter. On the test machine that
  was `~/.hermes/tools/python-3.14.9+…/bin/python3`, shown by
  `hermes --print-runtime-command`. Set `HERMES_DISABLE_LAZY_INSTALLS=1` so
  Hermes' launch path never syncs or installs dependencies.

### Live smoke run (2026-11-22, owner-approved, cap USD 5)

**Four cases. This is a plumbing check, not evidence that Hermes improves or
performs well.** Model `claude-sonnet-5-5[1m]`, provider
`claude-subscription-directsdk-experimental` (the local Hermes default: the
`claude` CLI with no built-in tools, no user settings or MCP, one turn). No
toolsets. Budget: 2 iterations, 240 s, 2000 max tokens, USD 1 per case.
Validation-split cases only, scored by `evolution.adaptive.verifier`.

| Case | Difficulty | Defects | Tokens out / total | Est. cost USD | Latency s | P / R / F1 |
|---|---|---|---|---|---|---|
| pyrev-007 | simple | 1 | 321 / 1647 | 0.0041 | 4.3 | 0.33 / 1.00 / 0.50 |
| pyrev-008 | moderate | 1 | 320 / 1687 | 0.0045 | 3.7 | 0.50 / 1.00 / 0.67 |
| pyrev-031 | moderate | 0 (clean) | 4 / 1388 | 0.0014 | 3.7 | 1.00 / 1.00 / 1.00 |
| pyrev-043 | complex | 2 | 474 / 1890 | 0.0063 | 5.2 | 0.67 / 1.00 / 0.80 |

- No errors, tool calls or violations. The Hermes checkout was unchanged.
- Estimated spend across all attempts: about USD 0.025. This includes an
  earlier single-case run; a first attempt failed before any model call.
- `cost_status` was `estimated`, which Hermes prices at API rates. The
  provider bills the Claude subscription, so metered spend may be zero.
- Most tokens are cache read/write: `input_tokens` was 373 uncached on every
  case. The bridge now carries `cache_read_tokens` and `cache_write_tokens`.
- One run, n=4, no confidence intervals.
- Cost covers the main agent loop only; Hermes reports auxiliary calls
  (title generation, compression, …) separately and they are not included.
- `python` must be an interpreter that can import hermes-agent
  (`HERMES_PYTHON`); Hermes' launcher picks a PM-managed environment, so this
  is set by the operator.

## Policy comparison (gate 4)

### Policies (`evolution/adaptive/policies.py`)

All implement `select(context) -> (strategy, propensity)`, `update(context,
strategy, reward)` and `version`. A `Context` holds a discrete `key` and a
numeric `features` vector, built only from task metadata visible before the
decision.

| Policy | Version string | Propensity |
|---|---|---|
| `FixedPolicy` | `fixed/1:<strategy>` | 1 |
| `RandomPolicy` | `random/1` | 1/K |
| `EpsilonGreedyPolicy` (wraps `core.ContextualBandit`, uses `key`) | `epsilon-greedy/1:eps=<e>` | as before |
| `LinUCBPolicy` (disjoint LinUCB, Li et al. 2010, uses `features`) | `linucb/1:alpha=<a>,ridge=<r>` | 1, or 1/ties on exact ties |

LinUCB is pure Python: `A^-1` is kept up to date with Sherman-Morrison, so
no matrix library is needed. Being deterministic, it mostly logs propensity 1,
which limits later off-policy evaluation of its logs.

### Harness (`evolution/adaptive/compare.py`)

- Same conditions for all policies: per seed, one task sequence (drawn from
  train with a seeded RNG), the same executor and budget, and the same episode
  count. Each (task, strategy) pair is executed and scored once and shared by
  every policy, so all policies see identical outcomes.
- Two phases: online learning on `train`, then one pass over `validation` with
  learning off.
- Metrics: utility (`core.score_outcome`, unchanged), quality, cost, latency,
  violation rate, and regret against the per-task best strategy in hindsight.
  Task-level noise is not predictable from the context, so regret cannot reach 0.
- 95% percentile bootstrap CIs over seed-level means (seeded, 2000 resamples).
  Each adaptive policy is compared with the best fixed policy (chosen by train
  utility) and with random, using paired per-seed differences. Verdicts:
  `improvement` (CI > 0), `worse` (CI < 0), `no improvement detected`.
- Only `train` and `validation` are accepted. `test`, the sealed split and
  unknown splits raise `SealedSplitRefused`. `CaseBenchmark` refuses to
  construct if any case is outside those splits.
- Every decision can be written to `ExperienceStore`, with its propensity and
  policy version.
- Results are labelled SYNTHETIC unless the benchmark is real and the executor
  backend is `hermes`.

Pluggable inputs, once the other gates merge:

```python
from evolution.adaptive.benchmark import load_split          # gate 2
from evolution.adaptive.verifier import quality              # gate 3 (PR #2)
from evolution.adaptive.executor import FakeExecutor, Strategy  # gate 1 (PR #3)
bench = CaseBenchmark([*load_split("train"), *load_split("validation")])
quality_fn = lambda item, outcome: quality(item.ground_truth, parse(outcome.output))
```

`parse` (model output to findings) is not defined yet. Tests for these three
plug-ins are in `tests/test_policy_compare.py`. All three (executor #3,
verifier #2, gate 2 benchmark #4) now run on the sprint branch and pass.

### Synthetic result (2026-11-22)

`python -m evolution.adaptive.compare --seeds 20 --train-episodes 300`
(`synthetic-python-review/1`: 120 train / 60 validation tasks, scripted
executor, invented success rates, costs and latencies in
`evolution/adaptive/synthetic.py`). Mean utility, 95% CI:

| Policy | Train utility | Validation utility |
|---|---|---|
| fixed:direct | 0.503 [0.492, 0.514] | 0.517 |
| fixed:structured (best fixed) | 0.565 [0.554, 0.575] | 0.584 |
| fixed:tool_assisted | 0.398 [0.384, 0.411] | 0.483 |
| random | 0.489 [0.478, 0.500] | 0.538 [0.518, 0.560] |
| epsilon_greedy | 0.590 [0.576, 0.604] | 0.579 [0.559, 0.598] |
| linucb | 0.606 [0.592, 0.619] | 0.596 [0.576, 0.617] |

Paired utility difference against the best fixed policy (`fixed:structured`):

| Policy | Train | Validation |
|---|---|---|
| epsilon_greedy | +0.025 [+0.007, +0.043] improvement | -0.006 [-0.024, +0.014] **no improvement detected** |
| linucb | +0.041 [+0.026, +0.056] improvement | +0.012 [-0.009, +0.033] **no improvement detected** |

Both adaptive policies beat random in both phases. **On held-out validation,
neither beats the best single fixed strategy.** The train-phase gain is
measured against a baseline picked in hindsight on the same train data, so it
does not show generalisation. Fixed-policy validation intervals have zero width
because those runs are deterministic. **This is a hypothetical simulator: none
of it is evidence about Hermes.**

## Checkpoint and rollback (gate 7)

`evolution/adaptive/checkpoint.py`. Checkpoints live in the SQLite knowledge
store next to `ExperienceStore` (new tables `policy_checkpoints`,
`checkpoint_log`, `checkpoint_active`; the experience schema and its
`user_version` are unchanged). Pass an `ExperienceStore` to share its file.

### What a checkpoint holds

Sequence number, policy kind, policy version, active skill version, label,
UTC creation time, the validation result it was saved with, and the policy
state: epsilon-greedy counts and means, LinUCB `A^-1`, `b` and counts, and the
RNG state for every seeded policy (fixed, random, epsilon-greedy, LinUCB,
plain `ContextualBandit`). SHA-256 is taken over canonical JSON of the whole
record. The duplicated metadata columns must match the checksummed payload.

### Safety

- **Whitelisted fields only.** Every field is type- and range-checked on
  restore, and configuration (strategies, epsilon, LinUCB dim/alpha/ridge,
  fixed action, policy version) must equal the live policy. A checkpoint
  restores learned estimates, counts and RNG state only. It cannot change
  hyperparameters, reward weights, budgets, permissions or evaluation data,
  even with a recomputed checksum (tested).
- **Verify before change.** Corruption (checksum, truncated or non-canonical
  JSON, unknown kind, edited columns, bad state fields, bad RNG state) raises
  `CheckpointCorrupted`. The live policy and the store stay unchanged and a
  `corruption_detected` event is logged.
- **Atomic, logged rollback.** Status change, active pointer and the log row
  are written in one SQLite transaction. The in-memory policy changes only
  after commit. Tests inject a failure at the log write and at the pointer
  update and check that nothing changed.
- **Policy lines.** One store can hold several policies. Each checkpoint
  belongs to the line `(kind, policy_version)`; the active pointer and "last
  good" are per line, so a policy never rolls back onto another policy's
  checkpoint (tested). Two live policies with the same kind and version need
  separate stores.
- **Last good.** Rollback targets the newest `good` checkpoint on the same
  line that is older than the rejected one. A corrupt target is marked `corrupt` and skipped. If none
  verifies, rollback fails closed (`NoGoodCheckpoint`). A rolled-back
  checkpoint is never a target again.
- **Idempotent restore.** Restoring twice gives the same state, and the next
  decisions match the policy at save time (RNG restored). Tested for all five
  policy kinds and after reopening the database.

### Rollback trigger

`admit_with_rollback(store, policy, skill_version=..., admit=..., validate=...)`:
validate and save a `good` pre-admission checkpoint, call `admit(policy)`
(returns the new skill version), validate again and save a `candidate`. Roll
back if post-admission validation utility is strictly below the pre-admission
checkpoint's recorded utility minus `ROLLBACK_MARGIN` (fixed at 0.02 utility,
not a parameter), or on any critical violation even if utility rose.
Otherwise promote the candidate. If `admit` or validation raises, the
pre-admission checkpoint is restored before the error propagates. Validation
gets a deep copy, so it never changes the live policy.

`validation_utility(...)` is the standard `validate`: it loads only the
`validation` split through `benchmark.load_benchmark` inside
`optimisation_phase()` and runs the policy with learning off. Both entry
points are in the guarded list in `tests/test_benchmark_integration.py`.

### Wiring into admission (gates 5 and 6)

`admission.review_with_rollback(registry, candidate_id, base=..., make_executor=...,
store=..., policy=...)` is the checkpointed admission path:

1. Measure the parent (the active admitted skill) with
   `admission.catalogue_validation`: the catalogue strategy
   (`registry.candidate_strategy`) on the validation split, with the same
   cases, seeds, references and scoring as `admit` (`ADMISSION_POLICY`).
2. Save a `good` pre-admission checkpoint of the live selection policy whose
   skill version is the parent's registry candidate ID.
3. `review_registered(..., apply=True)` makes the gate 6 decision. A rejection
   logs `admission_rejected` and nothing is rolled back.
4. If admitted, measure the candidate's catalogue strategy the same way, save
   a `candidate` checkpoint (skill version = the candidate ID) and apply
   `rollback_decision` against the pre-admission checkpoint's recorded
   utility. Pass: the checkpoint is promoted. Fail (drop beyond
   `ROLLBACK_MARGIN` = 0.02, or any critical violation):
   - the candidate is moved `admitted -> retired` in the registry, decided by
     `admission-rollback:<policy sha>` with the pre/post utility, margin,
     violation flag and checkpoint IDs as evidence; it leaves the catalogue;
   - the checkpoint store rolls back atomically to the pre-admission
     checkpoint, and the same details are written in its `rollback` row.
5. If the post-admission check raises (for example unknown cost), the
   candidate is retired and the pre-admission checkpoint restored before the
   error propagates.

`ROLLBACK_MARGIN` (0.02) is separate from the admission `min_margin` (0.05);
`ADMISSION_POLICY` and its pinned SHA are unchanged. Only
`review_registered` and `_retire_after_rollback` (to `retired` only) may
call `registry.transition`; the static test pins both.

Cost: the path adds a pre and a post validation pass (3 seeds x 10 cases =
30 executor calls each) on top of admission's 60, so a live run of it costs
about twice an admission review.

### Limits

- Skill rollback is **retire**, which is terminal in the gate 5 registry: the
  same content cannot be proposed again. The parent stays admitted and is the
  active skill again. Nothing deploys, rewrites or deletes skill files.
- Registry and checkpoint store are separate SQLite databases, so the two
  rollback writes are not one transaction. The order is fail-safe: retire
  first, then roll the checkpoint back. A crash between the admission commit
  and the post-admission check leaves the candidate admitted without that
  check.
- Admission does not change the selection policy, so restoring the policy
  from the pre-admission checkpoint returns the same state. A policy rebuilt
  over the enlarged catalogue is a different policy line and cannot be
  restored from a pre-admission checkpoint.
- With a deterministic fake executor the post-admission check reproduces the
  admission result unless the catalogue run differs. The tests script that
  difference; the trigger matters for live, nondeterministic runs.
- Checkpoint status is lifecycle metadata outside the checksum. The checksum
  detects corruption; it is not a signature.
- All results in the tests are **synthetic** (hand-written benchmark, fake
  executor). No live Hermes run.

## Sealed test evaluation (owner decision, 2026-11-22)

The verifier CLI (`python -m evolution.adaptive.verifier`) cannot score the
sealed split: `load_cases` refuses sealed cases read from files. This is
accepted. The final sealed evaluation will be a deliberate, logged,
in-process step built for the report gate, through
`evolution.adaptive.evaluation.evaluate_sealed_test(reviews, final_evaluation=True)`.
That call loads the split with the integrity-checked loader, logs the
unsealing, and refuses inside an optimisation phase. Its results must not feed
back into optimisation or selection.

### Pre-registered analysis plan for the one-time final evaluation

Registered 2026-11-22, before any call on the sealed split (owner decision,
option 1). Plan ID `kafo-evo-001-final-eval-plan/1`. The same plan is the
`final_evaluation.ANALYSIS_PLAN` constant, and its SHA-256 is recorded in the
ledger. It is not changed after the run.

- **Design: paired.** Two skills review the same 10 sealed test cases, once
  each (sample 0):
  - the baseline skill `github-code-review@11ad06e6d49b`;
  - the admitted, kept GEPA candidate `github-code-review@28dcec5ab528`.

  Both use the same model, provider, budget, review prompt and parser, and
  the same live Hermes reviewer. Calls interleave per case. In `case_id`
  order, the baseline goes first on the 1st, 3rd, … case and the candidate
  on the 2nd, 4th, ….
- **Primary metric:** candidate minus baseline micro-F1 under **STRICT**
  scoring. STRICT is the gate 3 verifier, unchanged, with exact normalised
  categories.
- **Secondary metric:** the same difference under **LENIENT** scoring. LENIENT
  uses the strict line rules, one-to-one matching and counting, and also
  accepts the alternate valid categories recorded by the independent label
  audit.
  - The alternates are a scoring-side table,
    `evolution/adaptive/category_alternates_v1.json`, pinned by SHA-256 in
    `lenient.ALTERNATES_SHA256`. It covers 15 defects across all splits.
  - The dataset, its labels and its manifest are not modified.
- **Descriptive only, no inference:**
  - per-skill micro precision, recall and F1, and macro F1, each with a CI;
  - the paired macro-F1, micro-precision and micro-recall differences;
  - cost, completions, parse failures and violations.
- **CI method:** a 95% paired percentile bootstrap over cases.
  - 10000 resamples of case indices, `random.Random(0)`, with cases sorted by
    `case_id`.
  - Both skills use the same resamples.
  - Quantiles are linearly interpolated.
- **Missing data:** the spend-ledger wrapper automatically re-sends a call
  that errors or reports no cost, once, as in the full run.
  - A call that still fails after that, is skipped by the cost cap, or can't
    be parsed is scored as an empty review for that skill and case.
  - Nothing else is re-run.
- **Interpretation:** the candidate counts as better on the sealed test only if
  the primary CI's lower bound is above 0. The secondary result cannot
  override the primary.
  - The cases are synthetic and hand-written, so no result here is a claim of
    a real Hermes improvement.
  - With 10 cases the intervals are expected to be wide.
- **Spend and run-once:** the run has a shared USD 2 cap across both skills.
  Spend is also recorded in the full run's spend ledger, under the USD 20
  run cap.
  - It goes through `final_evaluation.run_paired_final_evaluation` and the
    run-once ledger `reports/kafo-evo-001/final_test_evaluation.json`.
    Exclusive-create means a failed run also counts as the one run.
  - The committed ledger stores, per skill and case:
    - case ID;
    - TP, FP and FN counts;
    - completion and parse flags;
    - cost, cost status and latency;
    - model;
    - error and exception types, and a violation count.

    It never stores sealed code, labels, findings, or free-text error or
    violation messages. Those stay in the run's own spend ledger, outside the
    repo. The report and PRs show aggregates only.

## Scientific acceptance criteria

1. Compared policies receive equivalent task distributions and resource budgets.
2. Task context features are observable before the decision.
3. Record action probabilities and policy/skill versions.
4. No access to sealed test cases during optimisation.
5. Identify if improvement is absent; never infer generalisation from synthetic data.
6. Evaluate policy adaptation separately from GEPA skill evolution.

## Safety constraints

Keep CLI demonstrations offline and read-only by default. Do not permit
unrestricted code execution or let the agent modify reward logic, evaluation
data, permissions or budgets. Treat violations as hard disqualifiers for
candidate promotion. Do not automatically deploy evolved skills.

## Integration note

Existing `python -m evolution.skills.evolve_skill` is a separate GEPA
pipeline. It is **not yet invoked** by `evolution.adaptive.cli`.
The initial PR is the bandit/MAPE-K foundation, not a completed
Hermes + RL + GEPA end-to-end claim.
