# Architecture

PLAN.md is the plan. This document is the implementation: what the code in
`evolution/` actually does, what calls what, and where each decision is
enforced. Every claim here is meant to be checkable against a named module,
class, or function. Where the implementation departs from PLAN.md, the
departure is stated with its reason.

---

## 1. Orientation

### 1.1 What operates on what

This repo never runs inside hermes-agent. It reads a separate hermes-agent
checkout, computes candidate rewrites, validates them, and either writes them
back into that checkout or leaves it untouched. Nothing is ever merged. A write
is deployed as a git branch plus a rendered PR body, both local; pushing that
branch and opening the pull request each need their own explicit flag and are
off by default.

```
  hermes-agent-self-evolution                  hermes-agent checkout
  (this repo)                                  (HERMES_AGENT_REPO)

  evolution/core/artifact_io.py   -- reads -->  tools/*.py  (schema dicts)
                                  -- reads -->  agent/prompt_builder.py
  evolution/skills/*              -- reads -->  skills/**/SKILL.md
  evolution/prompts/behavioral_eval.py
                                  -- runs  -->  batch_runner.py (subprocess)
  evolution/core/gates.py         -- runs  -->  tests/  (pytest, subprocess)
                                  -- runs  -->  environments/benchmarks/* (absent today)
  evolution/code/organism.py      -- git   -->  a new branch, one file per commit
  evolution/core/pr_builder.py    -- git   -->  evolve/<target>-<timestamp>,
                                                one commit holding the written files
                                  -- net   -->  git push        (only with --push)
                                  -- net   -->  gh pr create    (only with --open-pr)

  writes back:  tools/*.py descriptions (Phase 2, --write)
                agent/prompt_builder.py sections (Phase 3, --write)
                a git branch and a diff (Phase 4, always; never merged)

  and, when a write happened:
                a branch carrying the written files, plus PULL_REQUEST.md in
                the run's output directory (Phases 2 and 3 build the branch
                through pr_builder; Phase 4 reuses the branch CodeOrganism
                already made and renders only the body)
```

Every phase resolves the checkout through
`evolution.core.config.resolve_hermes_agent_path`, so an explicit
`--hermes-repo` always wins over discovery.

### 1.2 Directory layout

```
evolution/
  cli.py                      hermes-evolve entry point; lazy per-subcommand imports plus `status`
  core/
    config.py                 EvolutionConfig dataclass; hermes-agent repo discovery
    artifact_io.py            AST discovery and span-exact rewriting of schemas and prompt constants
    stats.py                  paired significance tests, confidence intervals, power, OLS trends
    gates.py                  GateStatus / GateResult / GateChain; pytest and benchmark gates
    dataset_builder.py        EvalExample / EvalDataset; synthetic generation; golden loading
    constraints.py            char budgets, growth ceiling, non-empty, skill frontmatter, pytest runner
    fitness.py                LLM-as-judge FitnessScore and the fast skill_fitness_metric
    external_importers.py     mine Claude Code / Copilot / Hermes session files into an EvalDataset
    cost.py                   UsageTracker / CostReport; what a run spent, read out of dspy's history
    pr_builder.py             branch, commit, and render the PR body PLAN.md constraint 5 asks for
  skills/                     Phase 1
    skill_module.py           SKILL.md load/split/reassemble; SkillModule (dspy)
    evolve_skill.py           CLI and run body
  tools/                      Phase 2
    tool_catalog.py           ToolCatalog, budgets, bundles, span-safe write-back
    selection_eval.py         tool-selection dataset, ToolSelector module, scoring
    accuracy.py               factual-accuracy checks of description text against the frozen schema
    cross_tool.py             per-tool rates, confusion matrix, the paired accept/reject guard
    evolve_tool_descriptions.py  CLI and run body
  prompts/                    Phase 3
    sections.py               EvolvableSection, SectionInventory, caching checks, writes
    behavioral_eval.py        scenario bank, harnesses, judges, the dspy module under optimization
    evolve_prompt_section.py  CLI and run body
  code/                       Phase 4
    organism.py               CodeOrganism: one file, one git branch, one commit per candidate
    safety.py                 AST guardrails (signatures, registry, error handling, guards)
    fitness_code.py           composite fitness with pytest as a hard gate
    evolve_tool_code.py       CLI and run body; drives Darwinian Evolver as a subprocess
  monitor/                    Phase 5
    metrics.py                append-only JSONL metric history, aggregates, trends
    triage.py                 ranking: potential improvement x usage frequency, plus adjustments
    loop.py                   one cycle: check, triage, dispatch, record

datasets/                     generated eval datasets (gitignored: .gitkeep is tracked, *.jsonl and *.json are not)
reports/                      the Phase 1 validation PDF
generate_report.py            standalone reportlab script that builds that PDF; not imported by the package
docs/ARCHITECTURE.md          this document
tests/                        the test suite; every test runs offline, with no API key and no network
.github/workflows/tests.yml   pytest on 3.10 / 3.11 / 3.12 with no secrets available, plus a --help smoke test per phase
```

`evolution/tools/__init__.py`, `prompts/__init__.py`, and `monitor/__init__.py`
still carry the docstring "Phase placeholder". That is stale text, not a
statement about the code: all three packages are implemented.

---

## 2. The shared core, in dependency order

### 2.1 `core/config.py`

`EvolutionConfig` is a plain dataclass holding every tunable a run needs:
model names (`optimizer_model`, `eval_model`, `judge_model`), the size budgets
(`max_skill_size` 15000, `max_tool_desc_size` 500, `max_param_desc_size` 200,
`max_prompt_growth` 0.2), dataset split ratios (0.5 / 0.25 / 0.25), gate
settings (`run_pytest` True, `run_tblite` False,
`tblite_regression_threshold` 0.02), and output settings (`output_dir`
`./output`, `create_pr` True). `create_pr` was a dead field for the whole of
Phases 1 through 5: it defaulted to True with nothing reading it. It is now the
default for Phase 2's `--create-pr/--no-create-pr`. `run_pytest` is still dead,
set by two phases and read by none; see section 6.

Repo discovery has two entry points and one deliberate non-raising wrapper.

- `get_hermes_agent_path()` is the strict one. Precedence:
  1. `HERMES_AGENT_REPO`, expanded, used only if the path exists,
  2. `~/.hermes/hermes-agent`,
  3. `Path(__file__).parent.parent.parent / "hermes-agent"`. The docstring calls
     this "`../hermes-agent` (sibling directory)", but from
     `evolution/core/config.py` that resolves to
     `<self-evolution repo root>/hermes-agent`: a checkout nested inside this
     repo, a sibling of the `evolution/` package rather than of the repo.
  If none resolve it raises `FileNotFoundError` naming the env var.
- `resolve_hermes_agent_path(hermes_repo)` is what every CLI calls. An explicit
  path is expanded and used as-is without an existence check, so a caller can
  point at a repo in an unusual location even when `~/.hermes/hermes-agent` is
  absent. With no override it falls through to `get_hermes_agent_path()`.
- `_discover_hermes_agent_path()` swallows the `FileNotFoundError` and returns
  `None`. It is the default factory for `EvolutionConfig.hermes_agent_path`, so
  constructing a config never crashes in a unit test or on a machine with no
  checkout.

### 2.2 `core/artifact_io.py`

This is the only module that rewrites hermes-agent source. Phases 2 and 3 both
route through it.

**Discovery.** `discover_tool_schemas(repo)` walks `<repo>/tools/*.py`, parses
each file with `ast.parse`, and iterates module-level assignments
(`_iter_module_assignments`, which handles both `Assign` and annotated
`AnnAssign`). A dict-valued constant qualifies as a tool schema when it has a
literal `"name"` and a `"description"` entry. A description that is computed
rather than written as a string literal is skipped entirely: there is no source
span to rewrite safely. Parameters come from
`parameters -> properties -> <name> -> description`. Results are sorted by
`(tool_name, constant)` so runs are reproducible.
`discover_prompt_sections(repo)` does the same for the four string constants in
`agent/prompt_builder.py` named by `EVOLVABLE_PROMPT_SECTIONS`.

**Span-exact replacement.** The AST reports `lineno` and `col_offset` for each
node, and `col_offset` is a UTF-8 *byte* column, not a character index.
`_line_offsets(source)` builds a table of character offsets for each 1-indexed
line; `_offset_of` then re-encodes the line prefix to bytes, truncates at the
byte column, and decodes back, so the returned offset is a character index into
the string that `read_text` produced. This matters because hermes-agent's
prompt text contains em dashes and emoji, and a naive `start + col` would be
wrong by one position per multi-byte character on that line.

```
  source string  ......'Read a file from disk...'..............
                       ^                        ^
                       span.start               span.end
  replace_span   ......<render_string_literal(new_text)>.......
```

`_span_of` packages the pair as a frozen `SourceSpan`, and `replace_span`
does the substitution by slicing. Nothing else in the file is touched, so no
reformatting, reordering, or re-quoting can happen as a side effect.

**Why `render_string_literal` uses `repr`.** The replacement text has to be
valid Python source that evaluates back to exactly the string the optimizer
produced. `repr` handles quote selection, backslashes, embedded quotes, and
non-printables in one step and round-trips exactly. Hand-rolled quoting would
have to re-derive all of that and would eventually get one case wrong inside a
file that then fails to import. For multi-line values the function emits an
implicitly concatenated parenthesised block, one `repr` per line, keeping the
terminating newline attached to the line it ends so reassembly is byte
identical. The indent argument matches the nesting depth of the target:
`apply_tool_description` passes 8, `apply_param_description` passes 12,
`apply_prompt_section` passes 4.

**Why structure verification exists and how it works.** PLAN.md says the schema
structure is frozen and only text evolves. `schema_skeleton(source)` makes that
checkable: it re-parses the source and builds, for every schema constant, the
tool name, the per-parameter shape (every inner key *except* `description`,
literal-evaluated where possible), the `required` list, and the `parameters`
type. Description strings are deliberately excluded, so the skeleton is exactly
the part that must not move. `verify_structure_unchanged(before, after)`
computes both skeletons and raises `StructureViolation` if the constant sets
differ or any entry differs. It also raises when the rewritten source no longer
parses. `_rewrite_checked` runs it after every individual write, and
`tool_catalog.write_bundle` runs it once more over the accumulated result of
all edits to a file before anything reaches disk.

**Why spans are invalidated by a write.** A span is a pair of offsets into one
exact string. Replacing a span with text of a different length shifts every
offset after it, so any other span captured from the original text now points
at the wrong characters. The two writers handle this the same way, by staging
and re-discovering:

- `tool_catalog._descriptors_for_source` writes the current in-memory source to
  a throwaway `tools/<module>.py` in a `TemporaryDirectory` and calls
  `discover_tool_schemas` on it, so the next edit in a multi-edit sequence uses
  fresh spans. This is also what lets `dry_run` exercise the full rewrite path
  without touching the real repo.
- `sections.write_sections` stages `agent/prompt_builder.py` in a temp
  directory and re-runs `discover_prompt_sections` before each section update.

`apply_prompt_section` additionally refuses any constant not in
`EVOLVABLE_PROMPT_SECTIONS`, so a typo cannot rewrite an unrelated module
constant, and it re-parses the result to confirm the file still compiles.

### 2.3 `core/stats.py`

Every comparison this pipeline makes is paired: the baseline and the candidate
are run over the identical example list, in the same order, so example *i*
yields a matched pair of outcomes. `stats.py` exists because the rest of the
code used to throw that pairing away and model no uncertainty at all.

**Why paired.** Under pairing, the variance of the difference depends only on
the examples where the two versions disagreed. Examples both got right, and
examples both got wrong, carry no information about which version is better.
Treating the two runs as independent samples discards that and loses power;
comparing point estimates against a fixed tolerance models no noise whatsoever.
With 10 examples a rate of 0.8 has a standard error of 0.126, so a 12 point
swing is one standard error of nothing happening.

**McNemar's exact conditional test.** `mcnemar_exact(b, c, alternative)` takes
only the two discordant counts: `b` is baseline-right and candidate-wrong, `c`
is the reverse. Conditional on `b + c` discordant pairs, the null hypothesis
says each discordant pair is equally likely to fall either way, so the count is
Binomial(b + c, 0.5) and the p-value is exact, computed by `binomial_sf` and
`binomial_cdf` with `math.comb`. Concordant pairs are excluded because they are
uninformative under that conditional, not because they are inconvenient: adding
them back would inflate the denominator with observations that cannot
discriminate between the two versions. `alternative="worse"` is the one-sided
test a safety gate wants, `P(at least b losses)`.

`compare_paired_binary(baseline, candidate)` builds a `PairedBinary` from two
aligned boolean sequences and raises `ValueError` on a length mismatch, which is
the only mechanical protection against silently unpaired input.
`PairedBinary.delta_interval()` is built **conditionally on the discordant
pairs**, not from the Wald variance. The Wald form for correlated proportions,
`(b + c - (c - b)^2 / n) / n^2`, collapses to exactly zero whenever every
disagreement points the same way, so a candidate that flipped ten of ten
examples in its favour reported a zero-width 95 percent interval around +100
percent. That is the same degeneracy Wald is rejected for two paragraphs below,
and it is worse here because it shows up precisely on the strongest results.

Conditioning fixes it. With `m = b + c` disagreements the count that went the
candidate's way is Binomial(m, pi), so a Wilson interval on `c / m` maps back to
the rate scale through `(2*pi - 1) * m / n` and is never zero-width for `m > 0`.
With no disagreements at all there is nothing to condition on, so the discordant
*rate* is bounded by a Wilson interval on `0 / n` and the difference inherits
that bound symmetrically: no disagreement is not proof of no difference.
`Interval.estimable` is False when a sample genuinely cannot support an interval,
so a zero-width result is reported as "not estimable from this sample" rather
than printed as certainty.

**Wilson versus Wald.** `wilson_interval(successes, n, confidence)` is used for
single-rate intervals. The Wald interval `p +/- z*sqrt(p(1-p)/n)` collapses at
the boundaries: at 10 successes out of 10 it reports `[1.0, 1.0]`, asserting
certainty from ten observations. Wilson solves the score equation instead and
reports `[0.72, 1.0]` for the same data. Eval sets here are small and rates sit
near 1.0, which is exactly where Wald misbehaves.

**`min_detectable_paired_shift(n, alpha)` and why it matters here.** The most
favourable possible outcome for detection is that every disagreement points the
same way. If `k` of `n` examples flip against the candidate and none flip for
it, the one-sided exact p-value is `0.5**k`, so significance requires
`k >= ceil(-log2(alpha))`. The function returns `k / n`. At alpha = 0.05 that is
5 flips: at n = 10 the smallest detectable regression is 50 percentage points,
at n = 40 it is 12.5. A gate enforcing a 5 percent per-tool tolerance on 40
examples per tool was therefore never enforcing it by evidence, and
`PairedBinary.underpowered_for(tolerance)` is the predicate that says so
out loud.

**Continuous outcomes.** `compare_paired_continuous` returns a
`PairedContinuous` carrying the two means, the paired delta, a bootstrap
interval, a Wilcoxon signed-rank p-value, and paired Cohen's d.
`paired_bootstrap_ci` resamples *pairs* rather than individual observations,
which is what preserves the pairing, and takes a fixed default seed
(`20260731`) so a borderline gate verdict is reproducible and auditable rather
than able to flip on a rerun.

`wilcoxon_signed_rank` drops zero differences and assigns average ranks to ties.
Its p-value is **exact** up to `_EXACT_WILCOXON_MAX_N` (50) non-zero
differences, falling back to the tie-corrected normal approximation only above
that. Exactness is not a refinement here, it is the difference between deploying
and not: the approximation is badly anti-conservative in the small-sample regime
these eval sets live in, reporting p = 0.046 for four pairs all moving the same
way where the exact answer is 0.125. The exact null is the distribution of
subset sums of the observed ranks over all `2**n` sign assignments, counted by
dynamic programming over a doubled-rank lattice so the half-integer ranks that
ties produce stay on integers.

`PairedContinuous` takes its **direction from the ranks and not from the mean**.
The two can disagree: eleven scenarios improving while one collapses drags the
mean negative while the ranks clearly favour the candidate, and taking the
p-value from one and the sign from the other labelled that a significant
*regression*. `signed_rank_direction` returns the sign of `w_plus - w_minus`,
`PairedContinuous.direction_conflict` is True when it contradicts the mean, and
neither `significant_improvement` nor `significant_regression` can be True while
it is. A result that hinges on one outlier is reported as inconclusive rather
than resolved in either direction.

**Trends.** `ols_trend(xs, ys)` fits `y = a + b*x`, computes the residual
standard error, the t statistic on the slope, a two-sided p-value from
`student_t_sf` (Student-t tail via a directly implemented regularized
incomplete beta, since the standard library has none), R squared, and a
confidence interval on the slope whose critical t is found by bisecting the
survival function. `OLSTrend.significant` is `n >= 3 and p_value < alpha`, a
real test rather than a magnitude threshold. Fewer than three points returns a
zero-slope non-significant result: with two points the line passes through both
exactly, the residual variance is zero, and there is no error term left to test
against.

**Degenerate fits fall back to Mann-Kendall.** "No residual scatter" has to be
judged relative to the data rather than against exact zero: three readings on a
straight line leave residuals of order 1e-16, which is enough for the standard
error to underflow and the t statistic to explode, manufacturing `p = 0.000` out
of rounding error. `ols_trend` treats `sse <= sst * 1e-12` as a perfect fit, and
this is not an exotic case - success rates measured over two sessions are
quantized to {0, 0.5, 1.0}, and 1.0/0.5/0.0 is exactly collinear.

The t-test being undefined does not make the question unanswerable, so
`mann_kendall(ys)` answers it instead. It counts concordant minus discordant
pairs, needs no estimate of sigma, and is exact up to 20 points with no repeated
values by counting permutations with each inversion number (the Mahonian
distribution). It gets the intuition right in both directions where a blanket
"degenerate means no evidence" rule would not: a perfectly straight six-point
decline is one ordering in 720, `p = 0.003`, and is significant; the same shape
over three points is one in six, `p = 0.333`, and is not. `OLSTrend.method`
records which test produced the p-value, and `OLSTrend.degenerate` flags the
fallback.

**Intersection-union, and why no multiplicity correction.** Accepting a Phase 2
candidate means asserting a conjunction: "no individual tool regressed". Each
per-tool claim is tested at alpha, and under the intersection-union principle a
conjunction of claims each tested at level alpha is itself valid at level
alpha, because the null of the conjunction is rejected only when every
component null is rejected. No Bonferroni or Benjamini-Hochberg adjustment is
applied or needed. Applying one would be actively harmful: Bonferroni divides
alpha by the number of tools, making each per-tool regression harder to call
significant, and since the gate rejects the candidate when *any* tool's
regression is significant, that makes the gate more permissive the more tools
there are. A safety gate should not get easier to pass as the surface area it
protects grows. This is stated in the `stats.py` module docstring and again at
the guard in `cross_tool.py`.

**And where correction is required.** The exemption is about the direction of
the quantifier, not about multiplicity being harmless. `--all-sections` measures
one baseline holdout and then tests each section against it, deploying every
section that clears alpha. That is a **disjunction** - "any of these worked" -
and selecting the best of k inflates the family-wise error: four sections at
alpha = 0.05 give `1 - 0.95**4` = 18.5 percent, not 5. Those p-values are
Holm-adjusted by `holm_adjust` in `evolve_prompt_section.evolve`, and a section
that survives alone but not the correction is dropped with its adjusted p-value
named in the reason. Holm rather than plain Bonferroni because it is uniformly
more powerful and equally valid without assuming independence.

The rule to carry away: correct a disjunction, never a conjunction.

`stats.py` uses the standard library only: no numpy, no scipy. The incomplete
beta function is implemented directly because a Student-t tail is needed and the
standard library has none. Its tests in `tests/core/test_stats.py` check the
distribution primitives against published t-tables, hand-computed binomial
sums, and Anscombe quartet I.

Five call sites use it, and each is described in its own section below:

| caller | what it tests | section |
|---|---|---|
| `tools/cross_tool.py` | per-tool selection, paired binary, exact McNemar | 3.2.1 |
| `prompts/evolve_prompt_section.py` | holdout judge scores, paired continuous, Wilcoxon | 3.3.1 |
| `code/fitness_code.py` | per-test pytest outcomes paired by node id; Wilson interval on a repro fix rate | 3.4 |
| `monitor/metrics.py` | OLS trend with a t-test on the slope | 3.5.1 |
| `monitor/triage.py` | `_rate_interval`, a Wilson interval on a triaged rate reconstructed from its mean and sample count | 3.5 |

### 2.4 `core/gates.py`

A gate answers "did this candidate break something else?", as distinct from
fitness, which answers "is it better at its job?". `GateStatus` has four
values and the distinction between two of them is the point of the module:

| status | meaning | blocking? |
|---|---|---|
| `PASSED` | the gate ran and the candidate cleared it | no |
| `FAILED` | the gate ran and the candidate did not clear it | always |
| `UNAVAILABLE` | the gate could not run at all | only under `strict` |
| `SKIPPED` | the gate was deliberately not run | no |

`UNAVAILABLE` is not `PASSED` because scoring an absent benchmark as a pass
would let an unvalidated variant ship while the run log says everything was
checked. `GateResult.blocking` is `status is FAILED`; `GateChain._is_blocking`
adds `strict and status is UNAVAILABLE`.

`run_pytest_gate(repo, subset, timeout, python)` shells out to
`python -m pytest <subset or tests/> -q --tb=short` with `cwd` set to the
hermes-agent repo. A missing repo or a missing `tests/` directory is
`UNAVAILABLE`; a timeout or a non-zero exit is `FAILED` with the last 25 lines
of output attached. `subset` is what makes the gate affordable per candidate
rather than only on finalists.

`find_benchmark(repo, name)` resolves a benchmark directory from
`KNOWN_BENCHMARKS`, which lists every plausible location for `tblite`,
`terminalbench2`, and `yc_bench`. An explicit `HERMES_BENCH_<NAME>` env var
overrides discovery. `run_benchmark_gate` returns `UNAVAILABLE` when nothing
resolves, `FAILED` when the runner produced output no score could be parsed
from (`_parse_benchmark_score` accepts a trailing JSON object with
`score`/`pass_rate`/`accuracy`, an `N/M` fraction, or a percentage), and
otherwise compares against `baseline` at `regression_threshold`, failing when
`delta < -abs(threshold)`. With `baseline=None` it passes and reports the raw
score.

`GateChain.run(*gates)` accepts callables or ready-made results, appends each
result, and **breaks out of the loop at the first blocking result**. That
short-circuit is what keeps a cheap gate in front of an expensive one: in
Phase 4 the safety guardrails run before pytest, and in
`fitness_code.CodeFitnessEvaluator.evaluate` a failed pytest means the
benchmarks never run. `passed` is "no result in the list is blocking", so a
chain that short-circuited is not passed.

### 2.5 `core/cost.py`

PLAN.md requires the cost of the optimization run in every PR body, and puts a
figure on the expectation ("GEPA optimization: ~$2-10 per run"). Nothing
measured cost at all until this module existed.

**Why the number is read rather than computed.** dspy records every model call
in `dspy.clients.base_lm.GLOBAL_HISTORY`, each entry carrying a `usage` dict and
a `cost`. `cost.py` reads that log. It does not carry a price table of its own,
because a local table goes stale the week a provider changes rates and a stale
price presented as a cost is worse than no cost at all. `_history()` imports
`GLOBAL_HISTORY` inside a `try` and returns an empty list if dspy moves it, so
a dspy upgrade degrades the cost line rather than breaking the run.

**Three honesty rules, built in rather than left to the caller.**

- *An unpriced call is excluded, never zeroed.* `LMCall.cost` is `None` for any
  model dspy has no pricing for. `CostReport.known_cost` sums only the calls
  where `priced` is True, and `unpriced_calls` counts the rest. Summing an
  unknown as zero would produce a total that is quietly too low, which is the
  one failure mode a cost report must not have.
- *A truncated history is flagged.* dspy caps the global log, so a long run can
  lose its early entries. `UsageTracker` records `len(history)` on entry; if the
  log is *shorter* on exit than it was on entry, entries were evicted, so the
  tracker takes everything still there and sets `CostReport.truncated`. That is
  a deliberate over-count of scope in exchange for never reporting a negative or
  a silently partial total.
- *A cached call is counted separately.* `LMCall.cached` is True when the entry
  says so, or when it carries no usage and no cost, which is what a
  cache-served call looks like. `cached_calls` is reported on its own line so a
  cheap rerun is not mistaken for a cheap pipeline.

`CostReport.complete` is `not truncated and unpriced_calls == 0`, and
`describe()` prefixes the total with **"at least "** whenever it is False. Every
phase prints `describe()` verbatim for exactly that reason. `to_dict()` carries
the same fields into `metrics.json` under `cost`, including `complete` and
`truncated`, so a reader of the artifact can tell a measured total from a floor.

**Reading an entry defensively.** The shape of a dspy history entry is not part
of dspy's public API, so `LMCall.from_entry` treats every field as optional: a
non-dict entry becomes an empty `LMCall`, a non-dict `usage` is discarded, token
counts are read from `prompt_tokens`/`input_tokens` and
`completion_tokens`/`output_tokens` in that order, and a non-numeric cost
becomes `None`. A malformed entry degrades to an unpriced zero-token call
instead of raising in the middle of someone's optimization run.

`UsageTracker` is a context manager and also exposes `stop()`, which is what
lets Phase 3 hold it open across five numbered stages through an `ExitStack`
rather than indenting all of them. `read_history(entries)` builds a report from
raw entries and is what the tests use, since it needs no dspy at all.

The scope of a measurement is "whatever entered dspy's global history inside
this block", which means it is process-global rather than per-caller, and it
cannot see a model call made by a subprocess. Phase 4 says so explicitly; see
`EVOLVER_COST_NOTE` in 3.4.

### 2.6 `core/pr_builder.py`

PLAN.md constraint 5 is "Deployment via PR (Never Direct Commit)": an evolved
change reaches hermes-agent as `evolve/<target>-<timestamp>` plus a pull request
whose body carries the before/after scores per split, the full diff, the cost of
the run, and every constraint violation caught and rejected on the way. Until
this module existed the pipeline stopped one step short of that, writing evolved
text and a `metrics.json` into an output directory and leaving the reviewer to
assemble the rest by hand. `EvolutionConfig.create_pr` had defaulted to `True`
the whole time **with nothing reading it**; it is now the default for Phase 2's
`--create-pr/--no-create-pr`.

**Local by default, network only on request.** Building a branch, staging files,
committing them and rendering a body are local operations, and
`build_pull_request` does all four. Pushing and opening the PR are separate
methods on the returned plan, `PullRequestPlan.push(remote="origin")` and
`PullRequestPlan.open(base="main")`, and every phase gates them behind its own
`--push` and `--open-pr`, both defaulting to off. An optimization run that
phoned out to GitHub because a config field defaulted to True would be a bad
surprise, and "never direct commit" is a rule about review, not a licence to
publish automatically. `open()` checks `shutil.which("gh")` first and raises
`GitError` naming the branch and body as usable by hand, rather than failing
opaquely.

**What it does, in order.** `build_pull_request(repo=, target=, phase=,
timestamp=, files=, ...)` raises `GitError` immediately when `repo/.git` does
not exist, so a caller never believes in a branch that was never made. It then
reads the current ref through `_current_ref` (branch name, or the HEAD sha when
detached), creates `evolve/<target>-<timestamp>`, and, when `commit=True` and
`files` is non-empty, runs `git add -- <files>`, captures `git diff --cached`
for the body, and commits. The timestamp is a parameter rather than a clock
read, so a run is reproducible and a test can assert on the branch name. The
working tree is expected to already hold the evolved content: this stages what
it is given and never decides what changed.

**The body.** `render_body` emits PLAN.md's sections in PLAN.md's order:
a one-line header, a Scores table built from `ScoreLine(split, baseline,
evolved, detail)` rows, an Evidence block holding the phase's own statistics
string, the Gates list, "Rejected along the way" built from
`RejectedCandidate(label, reason)`, a Run block naming the optimizer, the
iterations, the eval dataset and `cost.describe()` (or `not measured`), and
finally the diff in a fenced block. The diff is clipped at `max_diff_lines`
(400) with a note saying how many lines were dropped and that the branch has all
of them, because a body that silently truncates a diff is worse than one that
admits it. `RejectedCandidate` exists because a PR showing only the winner hides
how hard the gates were working, and a reviewer who cannot see what was refused
cannot tell a careful run from a lucky one.

**The commit message** is assembled separately and follows PLAN.md's template:
`evolve: <target> - <last split> X to Y`, then the optimizer and iteration
count, the eval dataset, one `split: before -> after (delta)` line per score,
and the cost. `build_pull_request` headlines the *last* score row, which is why
each phase orders holdout last.

**Restoring the checkout.** `PullRequestPlan.restore()` checks the repo back out
onto `original_ref`, and does so only when `created_branch` is True, so a plan
bound to a branch somebody else owns cannot fight them for it. Every caller
invokes it from a `finally`. `to_dict()` records the branch, title, files,
`created_branch`, `original_ref` and the body path, and each phase stores that
under `pull_request` in its `metrics.json`.

`render_body` is exported separately from `build_pull_request` for Phase 4,
which already has a branch and a commit from `CodeOrganism` and needs the body
without a second branch mechanism racing over the same checkout. See 3.4.

### 2.7 What Phase 1 built that the later phases reuse

`core/dataset_builder.py` defines `EvalExample` (`task_input`,
`expected_behavior`, `difficulty`, `category`, `source`) and `EvalDataset`
(train / val / holdout, JSONL persistence, `to_dspy_examples`).
`SyntheticDatasetBuilder.generate` prompts the judge model for test cases,
tolerates prose around the JSON array, and splits with the config ratios.
`GoldenDatasetLoader.load` reads pre-split JSONL, or auto-splits a single
`golden.jsonl`. Phase 2 does not use `EvalDataset` as its working type, but
`ToolSelectionExample.to_eval_example` and
`ToolSelectionDataset.to_eval_dataset` bridge into it so shared reporting code
keeps working.

`core/constraints.py` provides `ConstraintValidator.validate_all(text, kind,
baseline_text)`, which runs `_check_size` against the per-kind budget
(`skill` 15000, `tool_description` 500, `param_description` 200),
`_check_growth` against `max_prompt_growth` when a baseline is supplied,
`_check_non_empty`, and `_check_skill_structure` for skills (frontmatter with a
`name:` and a `description:` inside the first 500 chars). Phase 2 calls it per
description. It also carries `run_test_suite(repo)`, which predates
`core/gates.py` and has no caller anywhere in the package; every phase that
gates on tests calls `run_pytest_gate` instead.

`core/fitness.py` provides `FitnessScore` (a weighted composite of correctness
0.5, procedure following 0.3, conciseness 0.2, minus a length penalty that
ramps from 0 at 90 percent of the size budget to 0.3 at 100 percent),
`LLMJudge` around a dspy `ChainOfThought` signature, and
`skill_fitness_metric`, a fast keyword-overlap proxy used inside the
optimization loop where a judge model call per example per iteration would
dominate the cost. Phase 3 replaces this with its own behavioural metric;
Phase 2 replaces it with selection scoring.

---

## 3. The phases

### 3.1 Phase 1 - skill files

**Target.** `<hermes_repo>/skills/**/SKILL.md`. `skill_module.find_skill`
matches on the containing directory name first, then falls back to scanning the
first 500 characters of each file for `name: <skill>`. `load_skill` splits the
file on `---` into `frontmatter` and `body` and pulls `name` and `description`
out of the frontmatter by line prefix. Only the body is evolved;
`reassemble_skill(frontmatter, evolved_body)` puts the original frontmatter
back verbatim.

**Wrapping.** `SkillModule` is a `dspy.Module` whose `TaskWithSkill` signature
takes `skill_instructions` and `task_input` as *input fields*. The skill text is
held on the instance as `self.skill_text` and passed in on every forward pass.

**Dataset.** `--eval-source synthetic` calls `SyntheticDatasetBuilder`;
`golden` calls `GoldenDatasetLoader`; `sessiondb` calls
`external_importers.build_dataset_from_external` over Claude Code, Copilot, and
Hermes session files. Synthetic datasets are saved to
`datasets/skills/<skill>/`.

**Scoring.** `skill_fitness_metric`, the keyword-overlap proxy.

**Constraints.** `ConstraintValidator.validate_all(body, "skill")` on the
baseline (violations are reported and the run continues) and again on the
evolved body with `baseline_text=` set, which adds the growth check. A failure
here stops the run and writes the rejected variant to
`output/<skill>/evolved_FAILED.md` for inspection.

**Decision.** Phase 1 compares `avg_evolved - avg_baseline` over the holdout
split and prints an improvement line. It applies no significance test and no
gate ladder, and it never writes into the hermes-agent checkout: the artifacts
are `output/<skill>/<timestamp>/{evolved_skill.md, baseline_skill.md,
metrics.json}`. A human copies the file across.

**Invocation and printed flow.**

```bash
python -m evolution.skills.evolve_skill \
    --skill github-code-review --iterations 10 --eval-source synthetic
# or: hermes-evolve skill --skill github-code-review
```

Phase 1's stages are numbered in source comments but the console output is
unnumbered: a load line, "Building evaluation dataset", "Validating baseline
constraints", "Configuring optimizer", "Running GEPA optimization",
"Validating evolved skill", "Evaluating on holdout set", the results table, and
the output path. `--dry-run` stops after the load line.

**Known behaviour.** The `dspy.GEPA(metric=..., max_steps=iterations)` call in
`evolve_skill.evolve` raises `TypeError` on dspy 3.2.1 - `max_steps` is not a
GEPA parameter in that version - and the surrounding `except Exception` prints
"GEPA not available" and falls back to `dspy.MIPROv2(auto="light")`. Phase 1 is
therefore MIPROv2 in practice on the pinned dspy. Phases 2 and 3 pass
`max_full_evals` and `max_metric_calls` respectively, both of which exist.

### 3.2 Phase 2 - tool descriptions

**Target.** Literal `description` strings inside the schema dicts in
`<hermes_repo>/tools/*.py`, both the tool-level description and each
parameter's. In the reference checkout that is four tools in
`tools/file_tools.py`: `read_file` (539 chars), `patch` (483),
`search_files` (435), `write_file` (387, with a 302-char `cross_profile`
parameter).

**Catalogue.** `tool_catalog.load_catalog(repo, config)` combines
`artifact_io.discover_tool_schemas` with `discover_toolsets`, which reads
`toolset=` off every `registry.register(...)` call in the AST rather than
importing the modules (importing would drag in the whole agent). A tool with no
readable registration gets `infer_toolset(module)` and
`toolset_source="inferred"`, never presented as fact. `param_types` and
`required` come from `schema_skeleton`, so the frozen half of the schema can be
rendered into prompts by `ToolEntry.signature()` without the optimizer being
able to invent arguments. `ToolEntry.budget_findings()` reports over-budget
descriptions; the loader never raises on them, because two of the real ones are
already over budget before evolution starts.

**Dataset.** `selection_eval.ToolSelectionDatasetBuilder.generate` asks the
judge model for three classes of case, which is PLAN.md's recipe:
`clear` (one tool is obviously right), `confusable` (two could work, one is
better), and `no_tool` (the right move is to answer directly). The `no_tool`
class is scored as a first-class outcome with its own confusion-matrix row,
because a description rewritten to sound maximally eager wins on the first two
classes while turning the agent into something that reaches for a tool when it
should just answer. Generated cases are validated against the real catalogue:
unknown tool names are dropped and arguments that do not exist in the schema
are stripped, both recorded in `builder.rejected`. `split_examples` stratifies
by correct tool before splitting, because the cross-tool guard compares
per-tool rates and a random split that puts every `patch` example in train makes
that comparison meaningless. Datasets persist to
`datasets/tools/<toolset or "all">/`.

**Wrapping.** `ToolSelector` renders the whole description bundle with
`render_tool_catalog` and installs it as the predictor's *instructions*
(`SelectTool.with_instructions(rendered)`) as well as passing it as an input
field. The instructions copy is what an instruction-mutating optimizer edits,
which is how evolved description text gets back out.
`extract_bundle(module, baseline)` recovers it: the module's own `bundle`
attribute if it moved, otherwise `parse_tool_catalog` over the rewritten
instructions, otherwise the baseline unchanged. `parse_tool_catalog` accepts
only tools and parameters that already exist in the baseline and falls back to
baseline text for any block it cannot read, so a reformatted candidate degrades
to "no change" instead of to garbage in a source file.

**Scoring.** `SelectionOutcome.score` is `0.7 * tool_correct + 0.3 *
param_score`, with `param_score` zero when the tool is wrong: a wrong tool with
perfect arguments is still a wrong tool. `parameter_correctness` scores only the
arguments the example pins down and does not penalise extras, because the schema
is full of optional parameters with defaults. `SelectionOutcome.feedback()`
writes the sentence GEPA reflects on, naming the tool that was picked instead.
`gepa_selection_metric` returns a bare score for evaluation and a
`dspy.Prediction(score=, feedback=)` when GEPA passes `pred_name`.

**Constraints, in order.** `freeze_unselected(candidate, baseline, allowed)`
reverts every tool the run was not asked to touch, so the comparison measures
the requested change and nothing else. Then `enforce_constraints` validates each
*changed* description against the 500/200 budgets, the growth limit, and the
factual-accuracy checker described below, and reverts that single description to
baseline on any failure rather than discarding the whole candidate. Unchanged
text is never re-validated, because `read_file`'s real description is already
539 chars and failing the run on a pre-existing violation would make the tool
unusable against the real repo. Finally the candidate's parameter dict is
filtered down to the baseline's parameter names, so invented parameters are
dropped.

**Factual accuracy.** PLAN.md's last Phase 2 constraint is that a description
"must remain factually accurate (can't claim a tool does something it
doesn't)". `evolution/tools/accuracy.py` enforces it in two tiers, and the
deterministic tier always runs. `FactualAccuracyChecker` compares text against
`ToolSchemaFacts`, built from the frozen half of the schema by
`facts_from_catalog` (parameter names, types, `required`, and `param_enums`,
which `tool_catalog` now extracts). Four structural findings:
`unknown_parameter` (a quoted token that reads as a parameter reference and is
not one), `enum_contradiction` (a value claimed for an enum parameter that the
enum does not allow, matched either as `mode='x'` or inside a window after the
parameter's name that stops at the next sentence or the next parameter),
`requiredness` (a required parameter described as optional, or "takes no
arguments" when the schema requires some), and `unsupported_capability` (text
promising the caller can set a timeout, encoding, retry count, or concurrency
level with no parameter to carry it). Behaviour claims are deliberately not
adjudicated: "searches recursively" is a fact about the handler, not about the
parameter list, and the schema cannot settle it. The second tier is a single
`DescriptionEntailment` model call asking whether the rewrite claims anything
the baseline description and the schema do not support; it is optional, skipped
with a stated reason when no predictor or no LM is configured, and never raises.
`build_accuracy_checker`, which lives in `evolve_tool_descriptions.py` rather
than in `accuracy.py`, constructs the checker and decides whether the entailment
tier gets a predictor at all; `--dry-run` and offline runs get the structural
tier alone.

**Cross-tool decision.** `CrossToolReport.from_outcomes` builds per-tool
`ToolRate` records and a `ConfusionMatrix` in which `NO_TOOL` is a row and a
column like any other tool. The rate table is seeded with every catalogue name
so a tool with zero examples still appears with zero opportunities, because a
silently absent row is how a regression hides. `CrossToolGuard.compare(baseline,
candidate)` then decides. The details of that decision, including the paired
statistics now applied, are in section 3.2.1 below.

**Gate ladder.** `GateChain(strict=strict_gates).run(...)` over, in order: the
cross-tool verdict as a `GateResult`, `run_pytest_gate(repo)` when
`--run-tests` was passed, and `run_benchmark_gate(repo, "tblite", fast=True)`.
The benchmark reports `UNAVAILABLE` on today's hermes-agent.

**Holdout.** The same baseline and candidate modules are run over
`dataset.holdout` and compared with the same guard, reported but not used as a
write precondition.

**Write-back.** `may_write = write and verdict.accepted and chain.passed and
bool(changes)`. `write_bundle` is called either way, with `dry_run=not
may_write`, so a run that reports a clean write really did execute the whole
rewrite and verification path. `--no-write` is the default.

**Cost.** Everything that can reach a language model, from dataset generation
through the baseline, the optimizer and every evaluation, happens inside a
single `with UsageTracker() as usage:` block that opens before stage 2 and
closes after stage 8. `cost = usage.report` is printed as a row of the results
table via `describe()`, stored under `cost` in `metrics.json`, and handed to the
PR body. A figure that skipped the optimizer would be the wrong figure, which is
why the block starts where it does.

**Deployment.** Stage 12 is the only stage that can build a branch, and it runs
only when `metrics["written"]` is true and `write_report.files_written` is
non-empty. No write means no branch: a run that changed nothing has nothing to
deploy, and an empty branch in the review queue is noise. A dry run returns at
stage 1 and never reaches it, printing instead what it *would* build, push and
open. With `--no-create-pr` the files are written in place and left uncommitted,
which the run says out loud.

Otherwise `build_pull_request` is called with the target slug from
`pr_target_slug(selected.names, toolset)`, the repo-relative paths of the files
`write_bundle` actually wrote, `score_lines(...)` (a `val` row always, a
`holdout` row when the holdout was measured, and no `train` row because this
phase never scores train and an invented number is worse than a missing one),
the `CostReport`, `collect_rejections(...)`, the gate summary, the dataset
split, the optimizer that actually ran, `verdict.summary()` as the evidence
block, and a notes list carrying the two model names, the description char
delta, the wall clock and the artifact directory. `plan.write_body(output_dir)`
drops `PULL_REQUEST.md` beside the other artifacts and `plan.to_dict()` goes
into `metrics.json` under `pull_request`.

`collect_rejections` is what makes the "constraint violations caught and
rejected" half of PLAN.md's PR body real: every reverted description with its
reason, every regressed tool from both the val and holdout verdicts, and a
whole-candidate entry when a verdict refused without naming a single tool.
Factual reverts appear on the same footing as budget failures, because that is
how `enforce_constraints` treats them.

`--push` and `--open-pr` are the only two steps that leave the machine, and both
are off by default; `--pr-base` (default `main`) is the base branch. A `GitError`
from any of them is reported and the run continues, because the evolved
descriptions are already in the artifacts. A `finally` calls `_restore_checkout`,
which prefers `plan.restore()` and falls back to the ref read before the call,
since `build_pull_request` creates the branch before it can fail and has no plan
to hand back when it does. Leaving an operator parked on an `evolve/` branch they
did not ask for is exactly the surprise this stage exists to avoid.

**Invocation and printed stages.**

```bash
python -m evolution.tools.evolve_tool_descriptions --toolset file --iterations 8
# or: hermes-evolve tools --tool read_file --tool search_files --write
python -m evolution.tools.evolve_tool_descriptions --write --push --open-pr
```

Phase 2 prints twelve numbered banners via `_banner`: 1 tool catalogue,
2 tool-selection dataset, 3 baseline measurement, 4 GEPA optimization,
5 constraint validation, 6 cross-tool regression check, 7 gate ladder,
8 holdout evaluation, 9 results, 10 write-back, 11 artifacts, 12 deployment.
`--dry-run` stops after stage 1. Stage 5 also reports whether the entailment
tier ran. Stage 6 prints `_rates_table`, which carries three columns beyond the
rates themselves: the 95 percent CI on the change, `p(worse)`, and a power cell
reading `no pairing`, `needs N%` when underpowered, `significant` in either
direction, or a bare tick. Artifacts land in
`output/tools/<timestamp>/{baseline_descriptions.json,
evolved_descriptions.json, cross_tool_report.json, gates.json, changes.json,
metrics.json}`, joined by `PULL_REQUEST.md` when stage 12 built one, and
`metrics.json` records `per_tool`, `underpowered_tools`, `unpaired_tools`,
`significant_regressions`, the two accuracy intervals, `chance_accuracy`,
`factual_reverts`, `entailment_ran`, `cost`, and `pull_request` alongside the
older fields. `docs/example-run/` holds one of these directories verbatim, so
the format can be read without executing a run first; its own README says what
the numbers in it are and are not worth.

#### 3.2.1 How the cross-tool guard decides

The guard used to compare two per-tool rates against a fixed tolerance. Two
rates cannot distinguish a candidate that flipped four answers against you and
four in your favour from a candidate that changed nothing, and a tolerance
compared against a point estimate models no noise at all. The guard now keeps
the pairing and tests it.

**The pairing is carried, not reconstructed.** `ToolRate` holds `outcomes`, the
per-example correctness vector for that tool in dataset order, and
`example_keys`, produced by `_example_keys` from the task text with `#2`, `#3`
suffixes for repeated tasks. Baseline and candidate are evaluated over the same
`dataset.val` list, so the two vectors line up example for example.
`align_outcomes(baseline, candidate, tool)` returns the aligned pair: identical
key tuples are returned directly, differing ones are joined through a key index
so a reordered report still pairs correctly, and vectors with no keys at all
(supplied directly by a caller asserting the order) are checked for equal length
only. It returns `None` only when one side never measured the tool or the two
share no example. `paired_for_tool` then calls
`stats.compare_paired_binary(..., alpha, confidence)` and gets a `PairedBinary`.

**The rejection rule is an OR, on purpose.** `_PairedEvidence` is the mixin
behind `ToolComparison`, `ToolRegression`, and `ToolImprovement`. It exposes
`breaches_tolerance` (the point estimate fell further than the tolerance) and
`significant_regression` (McNemar's one-sided `p_worse < alpha`), and
`regressed` is `breaches_tolerance or significant_regression`. The point
estimate catches a large drop the sample was too small to prove; the test
catches a small drop that is real. Requiring both would have made the gate more
permissive than it was before the statistics arrived, which is the wrong
direction for a safety gate. With `DEFAULT_TOLERANCE = 0.0` the point-estimate
arm already fires on any drop, so on the default setting the test only ever adds
rejections.

**Power is surfaced, not hidden.** `_PairedEvidence.underpowered` is
`paired.underpowered_for(tolerance)`, and it is suppressed when the tolerance is
zero, because a zero tolerance is already maximally strict and needs no warning.
`CrossToolVerdict.underpowered` lists the affected tools, `power_note()` turns
that into a sentence, and `summary()` and `to_gate_result()` both append it, so
a pass never reads as more coverage than it had. `unpaired` lists tools for
which no pairing could be built at all, reported rather than treated as a wash.

**Every measured tool gets a record.** `comparisons` holds one `ToolComparison`
per measured tool including the ones that held steady, because a tool that held
steady on four examples and one that held steady on four hundred are not the
same finding. Tools below `min_opportunities` land in `ignored`.

**The interval and the baseline travel with the accuracy.**
`CrossToolReport.accuracy_interval` is a Wilson interval on overall accuracy,
`chance_accuracy` is `1 / num_options` counting `NO_TOOL` as an option, and
`describe_accuracy` prints all three together, because a bare accuracy is
uninterpretable: 40 percent is poor against two options and excellent against
thirty.

**No multiplicity correction.** The comment sits directly above the rejection
branch in `CrossToolGuard.compare`. Accepting a candidate asserts the
conjunction "no tool regressed", which is an intersection-union test and valid
at alpha without adjustment; Bonferroni would raise each tool's bar as the
catalogue grows and make the gate looser the more tools it protects.

`ToolRate.to_dict` serialises the outcome vector as 0/1 integers, so every
p-value in `cross_tool_report.json` can be recomputed from the artifact without
rerunning the evaluation.

### 3.3 Phase 3 - system prompt sections

**Target.** Four module-level string constants in
`<hermes_repo>/agent/prompt_builder.py`: `DEFAULT_AGENT_IDENTITY` (513 chars in
the reference checkout), `MEMORY_GUIDANCE` (1426), `SESSION_SEARCH_GUIDANCE`
(186), `SKILLS_GUIDANCE` (1007). PLAN.md also lists `PLATFORM_HINTS`. In the
real file that is a dict of twenty-two per-platform strings (`whatsapp` through
`webui`, 13,250 chars in total), not a string, so
`sections._discover_structured` reports it as a `StructuredSection` whose
`reason` names the key count, and it is excluded from the allowlist. Rewriting
one platform's hint has its own accuracy rules ("do not tell Telegram to use
ANSI codes") and is a different operation; reporting it beats crashing discovery
or silently dropping it. `prompt_builder.py` carries a dozen other module-level
string constants as well (`KANBAN_GUIDANCE`, `TOOL_USE_ENFORCEMENT_GUIDANCE`,
`OPENAI_MODEL_EXECUTION_GUIDANCE` and the rest). None is on
`EVOLVABLE_PROMPT_SECTIONS`, so discovery never returns them and
`apply_prompt_section` refuses them by name.

**Inventory.** `load_sections(repo, names, max_growth, cache_budget_tokens)`
returns a `SectionInventory` holding `EvolvableSection` objects, the structured
constants, and any allowlisted name that was not found (`missing`). It refuses
unknown names up front through `validate_section_names`.

**Wrapping.** `behavioral_eval.SectionBehaviorModule` puts the section text in
the signature's *instructions*, wrapped in `<<<SECTION>>>` markers, not in an
input field. GEPA mutates instructions, so instruction mutation is section
mutation, and `SectionBehaviorModule.section_text` reads the evolved text back
out with `extract_section_text`. If the optimizer destroys the markers the
extractor falls back to the whole instruction block, and the growth and identity
constraints still get to judge the result.

**Evaluation.** A prompt section cannot be scored by reading it, so the unit is
a `BehavioralScenario`: a user message, the section it targets, a rubric, and
explicit `expected_tools` / `forbidden_tools`. `SEED_SCENARIOS` is a
hand-written bank of 60, twelve in each of five categories, each category
carrying at least two negative cases where the graded behaviour is restraint.
The bank is hand written because generated scenarios drift toward restating the
section they came from, which makes a prompt look good for agreeing with itself.

Two harnesses, chosen by `select_harness`:

- `BatchRunnerHarness` when `<repo>/batch_runner.py` exists. It writes a JSONL
  dataset, invokes batch_runner as a *subprocess* with
  `--ephemeral_system_prompt=<candidate>`, and reads trajectories back from
  `data/<run_name>/batch_*.jsonl`. It is never imported: batch_runner is a
  `fire`-driven CLI that forks its own `multiprocessing.Pool`, so importing it
  would hand this process someone else's process tree. (The docstring on
  `BatchRunnerHarness` says the `fire.Fire` call is at module scope; in the
  reference checkout it sits under an `if __name__ == "__main__":` guard, so the
  worker pool, not the module-scope call, is the reason subprocess is the only
  safe option.) Transcripts are joined to scenarios by prompt text in
  `_match_transcripts`, because batch_runner reorders across batch files and
  keys resume on prompt text.
- `DirectPromptHarness` otherwise. There are no real tools, so the model reports
  the tools it *would* call. This measures whether the guidance reads correctly,
  not whether the agent behaves correctly, and every report built this way is
  labelled `direct`.

Two judges. `heuristic_outcome` is deterministic and offline: tool discipline is
exact (calling a forbidden tool caps the tool score at 0.15), rubric keyword
coverage is a proxy, and category-specific rules apply - long responses are
penalised for `identity_tone`, markdown emitted where the rubric demands plain
text costs 60 percent of the quality score for `platform_formatting`. The final
score is `0.6 * tool_score + 0.4 * quality` when the scenario declares tools,
`quality` otherwise, and `PASS_THRESHOLD` is 0.6. `BehavioralJudge` adds an LLM
rubric grader used on the holdout only, and falls back to the heuristic with the
reason appended if the model call fails.

**Constraints, in order, in `SectionInventory.validate`.**
`EvolvableSection.check_non_empty`, then `check_growth` (ratio compared against
`max_growth` with a 1e-9 epsilon, so exactly +20 percent passes and one
character more fails), then `check_identity_traits` for the identity section
only, then `check_caching_boundary`. The identity check uses the explicit
`IDENTITY_TRAITS` table: helpful, direct, admits uncertainty, each satisfied by
any of several regex phrasings so the sentence can be rewritten but the trait
cannot be dropped; a failure names the lost trait. The caching check estimates
tokens at `CHARS_PER_TOKEN = 4` over `assembled_preview()`, which is the four
evolvable sections joined in PLAN.md's order plus a filler string standing in
for `PLATFORM_HINTS`. The filler is `max(s.total_chars for s in structured)`,
and `StructuredSection.total_chars` is the **sum of every value in the dict**,
so with one dict constant present the padding is all 22 hints (13,250 chars),
not the single widest one (1,478). The local variable is named `widest` and the
docstring says "the widest one", which describes an intent the arithmetic does
not implement; the effect is a conservative over-estimate, and on the reference
checkout it puts the prefix at ~4,098 tokens in 5 cache blocks. Two results come
out at different severities: exceeding `cache_budget_tokens`
(`DEFAULT_CACHE_BUDGET_TOKENS` 8192) is an **error**, because past it the prefix
stops being a stable cacheable prefix; crossing into an additional
`CACHE_BLOCK_TOKENS` (1024) block is a **warning**, a real per-session cost that
a genuinely better prompt can be worth. `SectionValidation.passed` ignores
warnings, `passed_strict` does not, and `--strict-gates` selects the latter.

**Gate ladder.** `run_gate_ladder` measures benchmark baselines *before*
staging, then uses the `staged_prompt_write` context manager to put the
candidate sections on disk, runs `run_pytest_gate(repo, subset=("tests/", "-k",
"prompt"))` and each benchmark at `ZERO_REGRESSION_TOLERANCE = 0.0`, and
restores the original file in a `finally`. A copy of the untouched file is
written to `output/prompts/<timestamp>/prompt_builder.py.bak` first, so a
process killed mid-gate is recoverable by hand. Staging is necessary because
pytest and benchmarks have no ephemeral-prompt channel: they must see the
candidate on disk. The tolerance is 0.0 rather than the 0.02 used elsewhere
because PLAN.md says "zero tolerance for regression here".

**Decision and write-back.** A section is accepted only if it passed
constraints, survived the gate ladder, and cleared the paired holdout test
described in 3.3.1. Even then, `detect_active_session()` runs first and a live
session refuses the write with exit code 2. That refusal is mechanical, not
cosmetic: a running session has already assembled and cached its system prompt,
so an evolved section deploys on the *next* session and is never hot-swapped.
Detection is conservative and reports its evidence: session environment
variables, lock files under `$HERMES_HOME`, and pid files whose process still
exists.

`write_sections` re-discovers spans per update in a temp directory, then calls
`verify_only_sections_changed(original, final, updates)`, which parses both
versions and compares *every* module-level string constant by value. That is the
complementary check to `apply_prompt_section`'s span arithmetic: it catches a
neighbouring constant that shifted, vanished, or appeared, by value rather than
by trusting offsets.

**Cost.** The meter is a `UsageTracker` held open through an `ExitStack` rather
than a `with` block, because the stretch that can call a model spans five
numbered stages (baseline, optimize, constraints, gates, holdout) and another
level of indentation would bury the structure. `cost_meter.close()` fires once
the last holdout comparison is in, and also on the early return when no section
had scenarios to optimize, where the run prints "Spent getting here" rather than
losing the figure. `cost.to_dict()` lands in `metrics.json`.

**Deployment.** The three flag combinations that cannot work are refused at the
very top of `evolve`, before any money is spent: `--push`/`--open-pr` with
`--no-create-pr` (nothing to push), `--open-pr` without `--push` (gh cannot open
a PR for a branch that exists only locally), and either without `--write` (a PR
is only built when a run actually deploys something). All three exit 1.

After a successful `write_sections`, and only then, `emit_pull_request` builds
the branch. `--create-pr` defaults to **on** here, but it is reached only behind
a write, so the default is "if you deployed, leave a reviewable branch", not "if
you ran, phone home". It passes `holdout_score_lines(deployed)`,
`rejected_candidates(outcomes, deployed)`, `gate_lines(chain)`,
`holdout_statistics(deployed)` as the evidence block, the cost report, and notes
carrying the next-session notice, the harness that ran, the gate tolerance and
the artifact directory. `pr_target(names)` names the branch after the deployed
sections.

A repo that is not a git checkout raises `GitError`, which is reported and
shrugged off with exit code 0: the sections are already on disk, and failing the
run after a successful write would say something untrue about the write. A
failed `--push`, a failed `--open-pr`, or a failed restore each return
`EXIT_DEPLOYMENT_INCOMPLETE = 3`, so "you asked for this and it did not happen"
is distinguishable from "the run finished". `plan.restore()` runs in a `finally`;
if even that fails the run says which branch you are still on.

**Invocation and printed stages.**

```bash
python -m evolution.prompts.evolve_prompt_section --section MEMORY_GUIDANCE
python -m evolution.prompts.evolve_prompt_section --all-sections --strict-gates --write
python -m evolution.prompts.evolve_prompt_section --all-sections --write --push --open-pr
```

Banners are unnumbered: "Discovering sections", "Building behavioral suite",
"Baseline behaviour", "Optimizing (N iteration(s))", "Constraints", "Gate
ladder", "Holdout", the results table, a per-category holdout table,
"Write-back", then "Pull request" when one is built. `--dry-run` prints its own
"Dry run" banner after "Building behavioral suite" and stops there, reporting
the holdout power it would have had, including an underpowered warning, and
stating that a dry run builds neither a branch nor a body, before any money is
spent. Artifacts land in `output/prompts/<timestamp>/` as
`baseline_<SECTION>.txt`, `evolved_<SECTION>.txt`, `scenarios.jsonl`,
`prompt_builder.py.bak`, `metrics.json`, and `PULL_REQUEST.md` when a branch was
built. Exit codes: 0 normal, 1 setup failure (no repo, no sections, bad
`--section`, nothing to optimize, an impossible deployment flag combination),
2 refused because a session is live, 3 an explicitly requested push, PR open, or
checkout restore did not happen.

#### 3.3.1 The paired holdout test

The write decision used to be `holdout_improvement > 0`: the sign of a
subtraction between two means. Baseline and candidate answer the same scenarios
in the same order, so the run has matched pairs, and it now uses them.

`align_holdout_scores(baseline, candidate)` checks the pairing rather than
assuming it. `BehavioralJudge.score_all` emits one outcome per scenario in the
order it was given them, so two runs over the same list are already aligned;
the function compares the two `scenario_id` tuples and raises `UnpairedHoldout`
on a length or order mismatch. A silent misalignment still produces a number,
and the number looks fine, so it is treated as a bug in the run rather than
worked around. Both runs are also labelled `HOLDOUT_SECTION_LABEL =
"SYSTEM_PROMPT"`, because the label reaches the direct harness's instruction
scaffold and letting it differ would change the prompt on one side of a paired
comparison for no reason.

A scenario the harness produced no transcript for is **dropped from both
sides**, not scored as a behavioural failure. An unmeasured run is missing data:
scoring it 0.0 on the side that timed out and a real number on the side that
completed manufactures a difference out of a flake, and six baseline timeouts
against a clean candidate run read as a significant improvement. Losing more
than `MAX_UNMEASURED_FRACTION` (0.2) of the suite that way raises
`UnpairedHoldout` outright, because at that point the run measured too little to
compare.

`compare_holdout` then builds a `HoldoutComparison`: `compare_paired_continuous`
over all scenarios, and the same paired test again inside each behavioural
category. Three questions are kept separate because they fail separately.

1. *Did anything happen?* `overall.significant_improvement` is a Wilcoxon
   signed-rank test on the matched differences at `HOLDOUT_ALPHA = 0.05`, with a
   seeded paired bootstrap CI (`HOLDOUT_BOOTSTRAP_SEED = 20260731`, fixed so a
   gate verdict cannot flip on a rerun).
2. *Did enough happen?* `PRACTICAL_IMPROVEMENT = 0.10` is PLAN.md's bar
   ("behavioral test scores improve, >=10% on targeted sections"), applied to
   the overall delta and again to the targeted category's delta. A significant
   +2 percent is real and still not worth deploying a system prompt for.
3. *Did anything break?* `category_regressed` fails a category when its delta
   falls past `CATEGORY_REGRESSION_TOLERANCE = 0.05` **or** its paired test
   finds a significant drop. Same conservative OR as the cross-tool guard, for
   the same reason. `regressed_categories` is an intersection-union conjunction
   over categories, so no multiplicity correction is applied, and the docstring
   says why.

`HoldoutComparison.accepted` is `n > 0 and improved and not
regressed_categories`. The whole holdout is scored, not just the scenarios
belonging to the section being edited, because a section rewrite lands in the
one prompt every category is answered under and collateral damage shows up in
the categories the rewrite was not aimed at.

Power is reported at three points. `min_scenarios_for_significance(alpha)`
computes the floor empirically, by feeding the most favourable possible input
(every scenario moving the same way by the same amount, which minimises the
signed-rank statistic and maximises the tie correction) into the same Wilcoxon
routine the gate uses; at alpha = 0.05 that floor is **6 scenarios**. Below it,
"not significant" is a statement about the sample size and not about the
candidate, and `power_note` says exactly that. `min_detectable_paired_shift`
gives the fraction of the suite that would have to move, and `shortfall_note`
turns an inconclusive result into either "needs N more scenarios" or "the
direction is not consistent enough for more scenarios alone to settle it".
`headline` and `reason` report the failure in the order the checks are allowed
to fail: no evidence, category regression, inconclusive, significant but small,
targeted category under the bar.

### 3.4 Phase 4 - tool implementation code

**Target.** One Python file inside the hermes-agent checkout, resolved by
`resolve_tool_file(repo, tool)` from a bare module name, a filename, a
repo-relative path, or an absolute path, searching the repo root, `tools/`, and
`agent/`. Anything outside the repo is refused.

**Organism.** `CodeOrganism` is the lineage mechanism PLAN.md asks for. `start()`
refuses to run against a repo with uncommitted *tracked* changes unless
`allow_dirty=True` (untracked files survive branch switches and are ignored),
records the current ref, creates `evolve/code/<stem>-<timestamp>`, and captures
the baseline sha and source. `mutate(source, label)` writes the file, stages
exactly that one path, and commits with `--allow-empty --no-verify`, so the
lineage has one commit per candidate *considered*, not per candidate that
happened to differ. `revert_last()` resets hard to the recorded parent sha.
`close()` runs from `__exit__`, discards the target file's uncommitted state,
and checks the operator back onto their original branch; the evolution branch is
left behind on purpose, because it is the review artifact. Commits set
`commit.gpgsign=false` and supply a fallback author only when the repo has no
`user.email` configured.

**Mutation source.** Darwinian Evolver, driven by `ExternalEvolver` as a
subprocess and never imported: it is AGPL v3 and this package is MIT.
`find_evolver` checks `--evolver-cmd`, then `DARWINIAN_EVOLVER_CMD`, then three
plausible command names, and raises `EvolverNotInstalled` (exit code 2) rather
than substituting a weaker mutation source and calling the result evolution. The
job spec is written to `<out_dir>/job.json` and passed by path so a failed run
leaves the exact request behind. Candidates are collected from
`<output>/candidates/*.py`, then `candidates.jsonl`, then JSON lines on stdout.
A non-zero exit that still produced candidates is reported and the candidates
are scored anyway: the guardrails decide, not the evolver's opinion of its own
run. `evolve_tool_code(evolver=...)` is the injection point the tests use.

**Guardrails, before anything expensive.** `safety.run_safety_checks(before,
after)` runs `check_parses` first and short-circuits on a syntax error, then:

| check | what it refuses |
|---|---|
| `check_signatures_unchanged` | removed functions, changed names/params/defaults/star-args/async-ness. Annotations are excluded (tightening a hint breaks no caller) and *added* functions are allowed |
| `check_registry_calls_unchanged` | any change to a `register*(...)` call, compared as a `Counter` of normalized call text |
| `check_error_handling_not_reduced` | a net decrease in try / except / raise / finally, module-wide **and** per function |
| `check_safety_checks_not_removed` | fewer asserts, fewer guard-helper calls, or fewer guard returns |

The last one is shaped by the real file. `tools/file_tools.py` contains no
`assert` and no `raise` at all; its safety story is guard helpers
(`_check_sensitive_path`, `_is_blocked_device`) plus early returns of
`tool_error(...)`. Counting asserts alone would score that file as having zero
safety checks and let a candidate strip every guard it has, so `census_guards`
also counts guard calls by name pattern and *guard returns* - a return inside an
`if` whose test either is a guard call or reads a name bound to one, or whose
value looks like an error. `_guard_bound_names` is what catches a candidate that
keeps `blocked = _check_sensitive_path(path)` but deletes the
`if blocked: return blocked` underneath it.

`quality_signals` is scored, not gated: new bare `except:`, new silently
swallowed exceptions, sharp file growth, new TODO markers, and dropped
docstrings cost points; added exception handlers earn a small bonus.

**Fitness.** `CodeFitnessEvaluator.evaluate(before, after, label)` asserts the
candidate is already on disk (it raises `FitnessError` if the target file does
not match, since pytest runs against the working tree, not a string), then in
order: reject an unchanged candidate, reject on any safety failure with the
expensive gates skipped, run `GateChain(strict).run(pytest, *benchmarks)` and
reject on the first blocker, run the bug reproduction and reject when
`require_bug_fix` is set and the bug is not fixed, and only then compute the
weighted score. pytest is a hard gate in the literal sense: a failing suite
scores 0.0, not "0.0 for tests and full marks elsewhere". Weights
(`bug_fix` 0.5, `benchmark` 0.3, `quality` 0.2) are renormalized over the
components that were actually measurable, so an absent benchmark neither scores
zero (which would reject everything) nor passes (which would certify nothing).

`BugReproduction` runs a script whose contract is "exit 0 when fixed, non-zero
when it still reproduces", which a pytest file expressing the desired behaviour
satisfies for free; a script can override with a `BUG_FIXED` or `BUG_PRESENT`
marker in its output. `snapshot_baseline` catches the two conditions that
invalidate a whole run: a baseline suite that is already red, and a repro script
that already passes.

**One run is one Bernoulli trial.** `ReproTrials` runs the reproduction
`--repro-runs` times per candidate (default 1) and aggregates. `fixed` requires
*every* run to report the bug fixed, so a patch that clears the reproduction
three times in five has not fixed the bug; `flaky` marks the disagreement;
`interval()` is a Wilson interval on the fix rate, Wald being useless at 1 of 1
where it claims `[1.0, 1.0]`. The CLI says so out loud when `--repro-runs` is
left at 1.

**The test suite is compared per test, not per count.**
`parse_pytest_outcomes` reads node ids out of pytest's output, and
`compare_test_suites` pairs baseline against candidate **by node id, never by
position**, because a candidate can add, remove, skip, or reorder tests. Tests
only one run knows about are reported as `added` / `removed` rather than padded
with an invented outcome. The result is a `SuiteComparison` carrying a
`PairedBinary`, `newly_failing`, and `newly_passing`. It is information, not a
gate: any failing test already rejects the candidate outright, so there is no
tolerance for a significance test to soften and no underpowered case to report.

**Evidence coverage.** `CodeFitness.evidence_coverage` is the measured weight
over `FitnessWeights.total()`, and `missing_evidence` names what did not run.
`score_line()` prints them together, because a 0.85 backed by tests, a benchmark
and a reproduction and a 0.85 backed by one heuristic are the same number
otherwise.

**Decision.** Candidates are alternatives generated from the same baseline, not
a sequence, so each is committed, scored, and then reverted with
`organism.revert_last()`. `rank_candidates` picks the winner and returns a
`CandidateRanking` that also says when the sort order was arbitrary: a margin
inside `DEFAULT_RANKING_RESOLUTION = 0.02` is `within_noise`, because nothing in
the composite score resolves finer than that (a binary bug fix, a pass rate over
a handful of tasks, and quality penalties in steps of 0.05). It also flags
`fix_rate_inconclusive` when the top two candidates' Wilson intervals overlap
and `thinner_evidence` when the winner rests on less measured evidence than the
runner-up. The winner is re-applied with `organism.reapply`, and the deliverable
is a branch, `winner.diff`, a `PULL_REQUEST.md`, and a panel telling the reviewer
the exact `git diff` command. Nothing is merged.

**Cost.** A `UsageTracker` wraps everything from the mutation request through the
end of the candidate loop, which is every model call this *pipeline* makes. It is
not every model call the run causes: Darwinian Evolver is a separate process, so
its usage never enters dspy's history and no tracker inside this process can see
it. `EVOLVER_COST_NOTE` is attached to the PR body and stored in `metrics.json`
under `cost_excludes`, and it says exactly that, pointing at PLAN.md's own
~$2-9 per task budget for the engine. A total that quietly omitted the component
doing the actual work would read as the cost of the run and be wrong by most of
it.

**Deployment.** Phase 4 does not call `build_pull_request`. `CodeOrganism`
already made `evolve/code/<stem>-<timestamp>` and already committed the winner
onto it, and two branch mechanisms racing over one checkout is the failure this
phase cannot afford. `build_code_pull_request` therefore renders the body with
`pr_builder.render_body` and returns a `PullRequestPlan` **bound to the existing
branch**, with `created_branch=False` and an empty `original_ref` so its
`restore()` is a no-op and cannot second-guess the organism, which owns the
restore. Nothing in that function touches git or the network.

It is reached only when `winner` exists *and* `winner_diff` is non-empty. A run
with no survivor, or one whose winner is byte-identical to the baseline, writes
no body: a document implying a diff that does not exist is worse than no
document. `--write-pr` (default on) controls only whether `PULL_REQUEST.md` is
written next to the artifacts; it is a local file. The body carries the score
lines, the gate lines, `code_rejected_candidates(outcomes)` naming every
candidate the guardrails threw out with its rejection reason, the ranking
statistics, the diff, `EVOLVER_COST_NOTE`, `REVIEW_NOTE`, the target issue, the
branch, and the literal `git diff <baseline-sha> <branch> -- <target>` a
reviewer should run.

`--push` and `--open-pr` are separate and both default to off. They are the last
thing the run does, after the review panel has printed, and a failure in either
sets exit code 4 without disturbing the rest of the result. The remote
(`origin`) and base branch (`main`) are parameters of `evolve_tool_code`, not
CLI flags.

**Invocation and printed stages.**

```bash
python -m evolution.code.evolve_tool_code --tool file_tools \
    --bug-issue 742 --repro-script repros/issue_742.py --repro-runs 5 --iterations 10
python -m evolution.code.evolve_tool_code --tool file_tools --push --open-pr
```

Stages print as `_step` titles: Baseline, Mutation, Evaluation, Results.
Artifacts land in `output/code/<stem>/<timestamp>/`: `job.json` and
`evolver_out/` from the evolver call, `baseline.py`, one `<label>.py` per
candidate considered, and then `winner.py`, `winner.diff`, `metrics.json` and
`PULL_REQUEST.md` when there is a winner. Exit codes: 0 the run
completed (with or without a winner), 1 setup problem, 2 Darwinian Evolver not
installed, 3 the evolver ran but produced nothing to score, 4 the run completed
but an explicitly requested push or PR open failed. `--dry-run` validates the
setup and returns 0 having built nothing: no branch, no body. The `finally`
block always attempts `organism.close()` and reports a failure to restore rather
than raising over the real result.

### 3.5 Phase 5 - the continuous loop

**Store.** `metrics.MetricStore` is an append-only JSONL file of `MetricPoint`
records (`metric`, `target`, `value`, `timestamp`, `source`, `samples`,
`metadata`). Append-only is load bearing: a monitor that rewrites its own
history can erase the evidence that a regression happened. `archive_before`
rotates old points into a sibling `<name>.archive.jsonl` and rewrites the live
file atomically through `tempfile.mkstemp` plus `os.replace`; nothing is
deleted. An unparseable line is skipped and counted in `skipped_lines` rather
than being fatal, so a truncated final line from a killed process does not blind
the monitor to the history above it. The clock is injected everywhere, which is
what makes trend detection reproducible in tests. `samples` travels with each
value because a 50 percent success rate over two sessions and over five hundred
are the same number carrying different weight.

The tracked metrics are `skill_success_rate`, `tool_selection_accuracy`,
`benchmark_score`, and `user_correction`, plus `optimization_run`, which the
loop writes about itself. `HIGHER_IS_BETTER` marks `user_correction` as the one
metric where rising is bad.

**Trend detection.** `compute_trend(points, ...)` sorts by timestamp, converts
to days since the first point (so an irregular reporting cadence does not
distort the slope), and fits a least-squares line. `change` is the modelled
movement across the observed span, which is the number a human cares about.
Fewer than `min_points` observations yields `TrendDirection.UNKNOWN`. The
significance rule and the statistics behind it are in section 3.5.1.

**Triage.** `triage.rank_points` implements PLAN.md's rule: score is
`potential improvement x usage frequency`. Potential improvement is headroom
from the sample-weighted mean to `ceiling`. Usage frequency is `_usage_weight`,
which is *not* `samples / busiest`: plain division is linear against a single
reference point, so one high-traffic target drives every other weight toward
zero and the ranking silently becomes "usage" rather than "improvement x usage".
The observed failure was a target that fell 0.91 to 0.55 ranking below one that
fell 0.82 to 0.74 on sample count alone. The weight is
`log1p(k * samples) / log1p(k * busiest)` with `k = config.usage_compression`
(default 1.0): monotone, so more traffic still means more weight and the busiest
target still scores 1.0, but steep where counts are small and flat where they
are large. `usage_compression = 0` selects the old linear ratio exactly.

Two multipliers are applied on top, each recorded as a named `ScoreFactor` so
`TriageEntry.explain()` can state the arithmetic: `1 + decline_weight` when the
trend is significant, and `1 + correction_weight * pressure` where pressure is
corrections over `correction_saturation`, capped at 1. Separately from the
ranking, an entry is `triggered` when its headroom reaches `failure_threshold`
over at least `min_samples` observations, or when the trend is significant with
at least `min_samples`. Headroom and failure rate are reported as two different
numbers under their own names: `TriageEntry.failure_rate` is `1 - current_value`
while headroom is `ceiling - current_value`, and they coincide only when the
ceiling is 1.0. Targets whose only evidence is user corrections get their own
scoring path, `_score_correction_only`, where correction pressure stands in for
headroom and the same compressed usage weight applies.

Target typing matters because it decides which phase runs.
`_resolve_target_type` prefers an explicit `target_type` in recent metadata,
then the metric's own meaning, then what the target has been observed as
elsewhere in the *whole* history, then the metric default. That third rule is
what stops a correction logged against a skill from being dispatched to the tool
phase. `TargetType.BENCHMARK` has no phase mapping on purpose: a low benchmark
score says something is wrong without saying which artifact to evolve, so those
entries are ranked, reported, and marked advisory, never actionable.

**One cycle.** `loop.run_cycle`:

1. Scheduled checks. `_run_scheduled_checks` runs each benchmark against the
   previous recorded value as its baseline and records the score with
   `samples=<spec.fast_task_count>` rather than 1, so a benchmark is not
   outvoted by a tool measured over hundreds of turns. A real number is recorded
   even when the gate *failed* on a regression, because the regression is
   exactly what the trend should see. A benchmark that did not run records
   nothing at all; a zero would read as "the agent failed every task".
2. Triage. `AutoTriage(store, config).rank(now=cycle_now)`.
3. Dispatch. `preflight(target_type, repo)` checks, in order, that a phase entry
   point exists for the type, that the phase module is importable
   (`importlib.util.find_spec`), that the path looks like a hermes-agent
   checkout, and that an API key is present. Any failure is a *skip with a
   reason*, not an error. `_in_cooldown` additionally skips a target proposed
   within `cooldown_days` (14 by default). The dispatch itself is
   `python -m <phase module> <flag> <target>` as a subprocess with
   `HERMES_AGENT_REPO` in the child environment rather than on the command line.
4. Record. `_record_outcome` writes an `optimization_run` point back into the
   same history triage reads: 1.0 for a proposal that was produced, 0.0 for a
   skip or a failure, distinguished by `metadata["status"]`, so a later cycle
   can see that the loop has been unable to act rather than assuming the target
   was never picked.
5. Stop. Nothing merges, pushes, or deploys.

`--emit-cron` prints an installable crontab line and installs nothing.
`benchmark_runner`, `dispatcher`, and `module_available` are all injectable,
which is how the whole cycle is exercised offline in `tests/monitor/test_loop.py`.

```bash
python -m evolution.monitor.loop                 # status only, no side effects
python -m evolution.monitor.loop --once --dry-run --max-targets 3
python -m evolution.monitor.loop --emit-cron --threshold 0.25
```

Stages print as `_step` titles: Scheduled checks, Triage, Dispatch, Result. The
process exits 1 only when a dispatched optimization crashed; a skipped or empty
cycle is a normal outcome for an unattended job and exits 0.

#### 3.5.1 How a trend is called significant

`Trend.significant` used to be `is_deterioration and abs(change) >=
significant_change`: a magnitude threshold wearing the word "significant". On
the series `[0.90, 0.35, 0.85, 0.30, 0.88, 0.33, 0.60]`, which is pure
oscillation with no trend, that rule reported "declining, significant" and
would have fired an optimization run. A t-test on the slope gives p = 0.582 and
R squared = 0.06.

`compute_trend` now takes three conditions, all required:

1. **statistically significant** - `Trend.statistically_significant` is
   `n >= 3 and p_value < alpha`, where the p-value comes from
   `stats.ols_trend(xs, ys, alpha, confidence)`, a real t-test on the fitted
   slope. The `Trend` also carries `stderr`, `r_squared`, and `slope_ci`.
2. **a deterioration** - `is_deterioration` folds in `HIGHER_IS_BETTER`, so a
   rising `user_correction` count counts and a rising success rate does not.
3. **practically significant** - `abs(change) >= practical_threshold`, where the
   threshold is the `significant_change` argument the caller passed. An
   optimization run costs money, so a real but trivial drift should not buy one.

`direction` is still magnitude based against `flat_tolerance`, deliberately:
direction is a description of the movement, and significance is the claim about
it. `Trend.fitted` (`n >= 3`, a known direction, and a non-None `slope_ci`)
tells a caller when `p_value` and `r_squared` carry information at all, so a
table can print nothing rather than print `p=1.000` and invite a reader to
mistake it for a measured result. `confidence_note()` returns an empty string
when `fitted` is False and `p=..., R²=...` otherwise.

The descriptive slope is still computed inside `compute_trend` rather than taken
from the fit, so the degenerate cases behave exactly as they did: a two-point
series still reports the line through both points, and a series where every
point shares a timestamp still reports its raw movement with a zero slope. The
fit supplies the uncertainty and declines to answer in precisely those cases.

Backwards compatibility is preserved rather than broken: `Trend.significant`
still exists and still means "an actionable decline", every previous field and
constructor argument still works, and `flat_tolerance` and `significant_change`
keep their names. `alpha` and `confidence` are new keyword arguments on
`compute_trend` and on `MetricStore.trend`, both defaulted.

Triage reads it through `TriageConfig.trend_alpha` and `trend_confidence`, and
`TriageEntry` carries `trend_p_value` and `trend_r_squared` (populated only when
the trend was `fitted`) with `explain()` appending `[trend p=..., R²=...]`. The
consequence is that the `1 + decline_weight` ranking multiplier and the
"significant decline" trigger both now require evidence that the slope is real,
so an oscillating series no longer spends a cycle.

---

## 4. One candidate, end to end

Phase 2 exercises every stage, so it is the clearest walkthrough. A single
candidate rewrite of `read_file`'s description travels this path:

```
  discover   artifact_io.discover_tool_schemas(repo)
                -> ToolDescriptor(read_file, span=(start,end))
             tool_catalog.discover_toolsets(repo)     -> toolset="file"
             tool_catalog.load_catalog(repo, config)  -> ToolCatalog
             ToolCatalog.bundle()                     -> baseline bundle
      |
  dataset    ToolSelectionDatasetBuilder.generate()
                -> clear / confusable / no_tool cases, schema-validated
             split_examples()  -> stratified train / val / holdout
      |
  baseline   ToolSelector(baseline_bundle, signatures)
             evaluate_selection(dataset.val, selector_predict_fn(module))
                -> SelectionReport -> CrossToolReport.from_report()
      |
  optimize   dspy.GEPA(metric=gepa_selection_metric, max_full_evals=N).compile()
             extract_bundle(optimized, baseline)   -> candidate bundle
      |
  constrain  freeze_unselected(candidate, baseline, allowed)
             build_accuracy_checker(catalog, lm)     -> FactualAccuracyChecker
             enforce_constraints(candidate, baseline, ConstraintValidator,
                                 accuracy=checker)
                -> per-description revert on a budget, growth, or factual failure
             diff_bundles(baseline, candidate)      -> DescriptionChange list
      |
  measure    ToolSelector(candidate_bundle, signatures)
             evaluate_selection(dataset.val, ...)   -> CrossToolReport
      |
  compare    CrossToolGuard.compare(baseline_report, candidate_report)
                -> CrossToolVerdict (see 3.2.1)
      |
  gate       GateChain(strict).run(verdict.to_gate_result(),
                                   run_pytest_gate, run_benchmark_gate)
      |
  write      write_bundle(repo, candidate_bundle, dry_run=not may_write,
                          baseline=baseline_bundle)
                -> _descriptors_for_source (re-discover spans per edit)
                -> artifact_io.apply_tool_description
                     -> render_string_literal (repr)
                     -> replace_span
                     -> verify_structure_unchanged   [per edit]
                -> verify_structure_unchanged        [whole file, once more]
                -> path.write_text  (only when may_write)
      |
  deploy     [only when the write above actually happened]
             cost = usage.report                      (UsageTracker, opened
                                                       before the dataset stage)
             score_lines(...)          -> ScoreLine per measured split
             collect_rejections(...)   -> RejectedCandidate per revert and
                                          per regressed tool
             pr_builder.build_pull_request(repo, target, phase, timestamp,
                                           files, scores, cost, rejected,
                                           gates, dataset, optimizer,
                                           iterations, statistics, notes)
                -> git checkout -b evolve/<target>-<timestamp>
                -> git add -- <files>; git diff --cached
                -> render_body   (scores, evidence, gates, rejected, run+cost,
                                  clipped diff)
                -> git commit -m <PLAN.md-shaped message>
             plan.write_body(output_dir)  -> PULL_REQUEST.md
             plan.push()                  (only with --push)
             plan.open(base=pr_base)      (only with --open-pr)
             finally: _restore_checkout   -> back to the original ref
```

Every arrow is a function call in `evolution/tools/evolve_tool_descriptions.py`
except the `write` block, which is `tool_catalog.write_bundle`, and the git and
rendering calls under `deploy`, which are `evolution/core/pr_builder.py`. A
reviewer can follow a single description from `tools/file_tools.py` to a verdict
by reading `cross_tool_report.json` and `changes.json` in the run's output
directory, and to a reviewable branch by reading `PULL_REQUEST.md` beside them.

The equivalent path for Phase 3 is
`load_sections -> BehavioralSuite.from_seeds -> suite.evaluate (baseline) ->
_optimize_section -> SectionInventory.validate -> run_gate_ladder
(staged_prompt_write) -> suite.evaluate (holdout, LLM judge) -> compare_holdout
-> holm_adjust -> detect_active_session -> write_sections ->
verify_only_sections_changed -> emit_pull_request -> build_pull_request ->
plan.write_body -> [plan.push] -> [plan.open] -> plan.restore`. For Phase 4 it
is `resolve_tool_file -> CodeOrganism.start ->
CodeFitnessEvaluator.snapshot_baseline -> ExternalEvolver.propose ->
organism.mutate -> run_safety_checks -> CodeFitnessEvaluator.evaluate ->
organism.revert_last -> rank_candidates -> organism.reapply (winner) ->
build_code_pull_request (render_body only, bound to the organism's branch) ->
plan.write_body -> [plan.push] -> [plan.open] -> organism.close`.

---

## 5. Enforced mechanically versus by convention

**Mechanically enforced.** These fail loudly in code, with a named exception or
a blocking status, and cannot be bypassed by an optimizer producing more
persuasive text.

| Rule | Enforced by |
|---|---|
| Schema structure is frozen | `artifact_io.verify_structure_unchanged`, raising `StructureViolation`; run per edit and again over the whole file in `write_bundle` |
| Only allowlisted prompt constants may be written | `artifact_io.apply_prompt_section` and `sections.validate_section_names`, raising `StructureViolation` / `SectionWriteError` |
| No neighbouring prompt constant may drift | `sections.verify_only_sections_changed`, comparing every module-level string constant by value |
| Description char budgets and the growth ceiling | `constraints.ConstraintValidator` via `evolve_tool_descriptions.enforce_constraints`, which reverts the offending description |
| A description may not claim a call interface the schema does not have | `accuracy.FactualAccuracyChecker` structural checks, wired into `enforce_constraints`; a finding reverts the description exactly like a budget failure |
| Prompt growth ceiling at +20 percent | `sections.EvolvableSection.check_growth` |
| Caching budget | `sections.SectionInventory.check_caching_boundary`, error at the budget, warning at a block crossing |
| An unavailable gate is not a pass | `gates.GateStatus.UNAVAILABLE` plus `GateChain._is_blocking` under `strict` |
| pytest is a hard gate for code | `fitness_code.CodeFitnessEvaluator.evaluate` returns `total=0.0`, no partial credit |
| Function signatures and registry calls are frozen | `safety.check_signatures_unchanged`, `safety.check_registry_calls_unchanged` |
| Error handling and guards may not shrink | `safety.check_error_handling_not_reduced`, `safety.check_safety_checks_not_removed` |
| Only one file changes in Phase 4 | `CodeOrganism` stages exactly `self.relpath`; a target outside the repo raises `OrganismError` |
| The operator's branch is restored | `CodeOrganism.close()` from `__exit__`, plus the `finally` in `evolve_tool_code` |
| The operator's uncommitted work is not clobbered | `CodeOrganism.start()` raises `DirtyWorktreeError` without `allow_dirty` |
| No prompt write while a session is live | `sections.detect_active_session` plus the exit-2 refusal in `evolve_prompt_section.evolve` |
| Paired comparisons stay aligned | `stats.compare_paired_binary` / `compare_paired_continuous` raise `ValueError` on a length mismatch; Phase 2 matches by example key in `cross_tool.align_outcomes`, Phase 3 raises `UnpairedHoldout` from `align_holdout_scores`, Phase 4 pairs tests by node id in `compare_test_suites` |
| A prompt section deploys only on paired holdout evidence | `HoldoutComparison.accepted`: significant Wilcoxon improvement, at least +10 percent, and no regressed category |
| No multiplicity correction on a conjunction gate | intersection-union, stated and relied on in `CrossToolGuard.compare` and `HoldoutComparison.regressed_categories` |
| Writes are opt-in | `--write/--no-write` defaults to `no-write` in Phases 2 and 3; `write_bundle(dry_run=True)` still runs the full verification path |
| A branch is only built when something was deployed | Phase 2 stage 12 runs only on `metrics["written"]` plus non-empty `files_written`; Phase 3 reaches `emit_pull_request` only after `write_sections` returns; Phase 4 requires a winner with a non-empty diff. A dry run returns before any of them |
| Pushing and opening a PR are each opt-in | `--push/--no-push` and `--open-pr/--no-open-pr` default off in all three phases; `PullRequestPlan.push` and `.open` are separate methods that `build_pull_request` never calls. Phase 3 additionally refuses `--open-pr` without `--push`, `--push` without `--create-pr`, and either without `--write`, at the top of the run |
| A missing `gh` is named, not guessed around | `PullRequestPlan.open` checks `shutil.which("gh")` and raises `GitError` saying the branch and body are usable by hand |
| The operator's ref is restored after a deployment | `_restore_checkout` in a `finally` (Phase 2), `plan.restore()` in a `finally` (Phase 3), `organism.close()` in a `finally` (Phase 4, which owns the branch, so its plan carries `created_branch=False`) |
| A cost is never rounded up from an unknown | `cost.CostReport.known_cost` sums priced calls only; `unpriced_calls` and `truncated` are reported, `complete` is False, and `describe()` prefixes the total with "at least" |

**Convention only.** These are real properties of the current code that nothing
would stop a future change from violating.

- *Nothing is merged.* No code path calls `git merge` or passes `--merge` to
  `gh`. `git push` and `gh pr create` do exist, in
  `PullRequestPlan.push`/`.open`, but nothing calls them without an explicit
  flag. The absence of a merge is a property of what was written, not a check.
- *No AGPL code is linked.* `evolve_tool_code` only ever runs Darwinian Evolver
  through `subprocess.run`. Nothing verifies that no future import appears.
- *Evaluation order produces the pairing.* Both sides of a comparison are
  evaluated by iterating the same list object (`dataset.val` twice, `holdout`
  twice), which is what makes the outcomes matched in the first place. The
  checks above catch a misalignment after the fact; nothing prevents a future
  caller from evaluating the two sides over different lists.
- *Phase 1 does not write into hermes-agent.* It writes files under `output/`
  and a human copies them. There is no gate preventing a future `--write`, and
  no PR path either: Phase 1 is the one tier with no deployment step.
- *The holdout is not touched during optimization.* GEPA sees `trainset` and
  `valset`; `holdout` is only read afterwards. Nothing enforces the separation
  structurally.
- *The PR body is complete because each phase fills it in.* `render_body` omits
  any section it was given nothing for, so a caller that forgot to pass
  `rejected` or `cost` produces a body that is quietly missing what PLAN.md
  asks for. Nothing checks that the phases supply all of it; today all three
  do.
- *`core/benchmark_gate.py`.* PLAN.md names it; the implementation is
  `core/gates.py`, which covers pytest as well as benchmarks and adds the
  status ladder and `GateChain`.
- *Stale docstrings.* `evolution/tools/__init__.py`,
  `evolution/prompts/__init__.py`, and `evolution/monitor/__init__.py` say
  "Phase placeholder". They are not.

---

## 6. Known limits

Each of these is a place the implementation cannot deliver what PLAN.md
promises. They are stated with their consequence rather than worked around.

**No benchmarks exist in hermes-agent.** PLAN.md gates on
`environments/benchmarks/tblite/` and `.../yc_bench/`. Neither path exists in
the reference checkout, and `find_benchmark` also tries `benchmarks/<name>` and
`environments/<name>` without success. *Consequence:* every benchmark gate
reports `UNAVAILABLE`. Runs are permissive by default, so Phase 2's TBLite gate
and Phase 3's zero-tolerance TBLite and YC-Bench gates currently certify
nothing; `--strict-gates` turns that into a hard failure, which is the honest
setting but blocks every run. `HERMES_BENCH_TBLITE` and friends can point at a
benchmark kept outside the repo. Phase 4's benchmark component is dropped from
the weighted average rather than scored, and Phase 5 records no benchmark point
at all, so the `benchmark_score` metric never accumulates history and its trend
is permanently `UNKNOWN`.

**GEPA is not actually running in Phase 1.** `evolve_skill.evolve` calls
`dspy.GEPA(metric=..., max_steps=iterations)`. `max_steps` is not a parameter of
`dspy.GEPA.__init__` on dspy 3.2.1, the version installed here, so the call
raises `TypeError`, the broad `except Exception` catches it, and the run falls
back to `dspy.MIPROv2(auto="light")` after printing "GEPA not available". The
fallback is legitimate; the silence in the artifacts is the problem, because a
run reported as GEPA is MIPROv2. Phases 2 and 3 use `max_full_evals` and
`max_metric_calls`, which do exist, and both record which optimizer actually ran
in `metrics.json` (Phase 2 under `optimizer`, Phase 3 per section). `pyproject`
asks only for `dspy>=3.0.0`, so nothing pins the version this behaviour was
observed against. *Consequence:* Phase 1 results are not GEPA results, and
Phase 1 does not record which optimizer produced them.

**`read_file` is already over budget.** Its real description is 539 characters
against PLAN.md's 500-character budget, and `write_file.cross_profile` is 302
against a 200-character budget. `load_catalog` reports these as
`BudgetFinding`s and never raises, and `enforce_constraints` validates only
descriptions that *changed*. *Consequence:* the budget is enforced against
evolved text but not against the baseline, so an unchanged `read_file`
description ships over budget, and any rewrite of it must come in under 500 or
be reverted - which means the optimizer cannot make a small edit to that
description at all, only a 39-character-shorter one.

**Small eval sets are underpowered for the tolerances being enforced.**
`min_detectable_paired_shift` quantifies it: at alpha = 0.05, no result on fewer
than 100 examples can call a 5 percent regression significant, and at 40
examples the floor is 12.5 percent. Phase 2's per-tool opportunity counts come
from a generated dataset split three ways, so a per-tool count in the tens is
normal. *Consequence:* the per-tool statistical test can only ever fire on large
regressions, which is precisely why the guard rejects on the point estimate as
well and reports the shortfall through `CrossToolVerdict.underpowered` and
`power_note()` instead of letting a pass imply coverage it did not have. Nothing
here manufactures power that the sample size does not contain; it only stops the
gate from claiming it.

Phase 3 is smaller still. The seed bank is 60 scenarios across five categories;
split 0.5 / 0.25 / 0.25 that leaves 15 holdout scenarios for an
`--all-sections` run and 6 for a single section (its own 12 plus the 12 platform
scenarios carried as a regression signal). The Wilcoxon floor from
`min_scenarios_for_significance` is 6, so a single-section run sits exactly on
the floor and can only reach significance when every one of its scenarios moves
the same way, while `min_detectable_paired_shift` is 83 percent of the suite at
n = 6 and 33 percent at n = 15. *Consequence:* most real candidates will land on
"inconclusive" rather than on a verdict. That is the correct answer for that
sample size, and it is now what the run reports instead of deploying on the sign
of a subtraction. Per-category tests are smaller again, which is why
`category_regressed` also fails on the point estimate and why
`underpowered_categories` is printed.

**SessionDB mining has no access path.** PLAN.md's plan is to mine
`hermes_state.py` (SessionDB) for real usage. Nothing in this repo opens a
SessionDB. `core/external_importers.py` reads flat files instead:
`~/.claude/history.jsonl`, `~/.copilot/session-state/*/events.jsonl`, and
`~/.hermes/sessions/*.json`. *Consequence:* the `sessiondb` eval source is
really "external session files", it depends on the operator having used those
other tools, and it can produce nothing on a fresh machine (Phase 1 exits 1 in
that case). Phase 5's `skill_success_rate` and `user_correction` metrics have no
producer anywhere in this repo either: `loop.py` writes only `benchmark_score`
and `optimization_run`, so those two signals must be recorded by something
external before triage can rank on them.

**Benchmark-derived datasets do not exist.** PLAN.md suggests deriving eval
examples from benchmark tasks. With no benchmarks present there is no such path,
and none is implemented. *Consequence:* every dataset in the pipeline is either
synthetic, hand written (Phase 3's seed bank), or imported from another tool's
session files.

**Phase 3's default harness is degraded.** `select_harness` prefers
`BatchRunnerHarness`, but it requires `batch_runner.py` in the checkout and a
real model budget. Without it, `DirectPromptHarness` asks the model which tools
it *would* call. *Consequence:* the numbers measure whether the guidance reads
correctly, not whether the agent behaves correctly. The report is labelled
`direct` and the CLI prints the reason, but the two are not comparable and
should not be trended against each other.

**Phase 4 cannot run without Darwinian Evolver.** It is AGPL v3, an optional
dependency, and deliberately not substituted. *Consequence:* `--tool X` with no
evolver installed exits 2 having mutated nothing. There is no built-in mutation
source.

**Phase 1 applies no statistics, no gate ladder, and no deployment step.** It
reports a raw holdout mean difference and declares improvement when it is
positive. *Consequence:* a Phase 1 "improvement" on a 5-example holdout is not
evidence, and unlike Phases 2 through 4 there is no pytest or benchmark gate
between the optimizer and the file a human is invited to copy into hermes-agent.
It is also the only phase that never builds a branch or a PR body and never
measures what it spent: PLAN.md constraint 5 is unimplemented for skills, and
the reviewer is still told to copy `evolved_skill.md` across by hand. Its
`--run-tests` flag, documented as "Run full pytest suite as constraint gate",
sets `EvolutionConfig.run_pytest`, and nothing in the package reads that field.
The flag is inert. Phase 2 accepts the same flag and does gate on it, through
its own local variable rather than through the config.

**Cost is measured from dspy's history, so it misses what dspy cannot see.**
`UsageTracker` reads `dspy.clients.base_lm.GLOBAL_HISTORY`, which is
process-global, so it measures whatever entered that log inside its block rather
than what a specific caller made. *Consequence:* three real gaps. A model dspy
has no price for is counted as unpriced and excluded from the total, so the
figure in the PR body is a floor and `describe()` says "at least $X"; that is
the honest reading, not a bug, but it is not a full cost. A truncated history
sets `truncated` and turns the total into a lower bound for the same reason.
And Phase 4's mutation engine runs as its own process, so none of its usage
reaches dspy at all: `EVOLVER_COST_NOTE` states that and points at PLAN.md's
~$2-9 per task estimate, but the real total for a Phase 4 run has to be added up
by the reader.

**The factual-accuracy entailment tier needs a model.** The structural checks in
`accuracy.py` are deterministic and always run, but they only adjudicate claims
about the *call interface*, which is all the frozen schema can settle. Claims
about behaviour ("searches recursively", "returns results ranked by relevance")
are only ever checked by the optional `DescriptionEntailment` call.
*Consequence:* an offline run, or one with no LM configured, enforces PLAN.md's
accuracy constraint only for parameters, enums, requiredness, and caller-set
affordances. The report says which tier ran, in
`AccuracyReport.entailment_ran` and `entailment_skipped`.

**`--repro-runs` defaults to 1.** One run is one Bernoulli trial, so the default
configuration cannot distinguish a fix from a lucky draw on a flaky
reproduction. `ReproTrials.interval()` reports how little one run establishes,
and the CLI prints a hint, but the default is still 1 because each extra run
costs a full reproduction. *Consequence:* the `bug_fix` component of Phase 4
fitness is, by default, a single observation with a Wilson interval of
`[0.21, 1.0]` behind it.

**The heuristic judges are proxies.** `core/fitness.skill_fitness_metric` scores
keyword overlap with the rubric, and `behavioral_eval.heuristic_outcome` mixes
exact tool discipline with keyword coverage. Both are deterministic, free, and
hard to game by writing more text, which is why they are used in the inner loop.
*Consequence:* neither measures task success. Only Phase 3's holdout uses an LLM
rubric judge, and only Phase 2's selection metric is checking something
objectively verifiable (was the right tool named).
