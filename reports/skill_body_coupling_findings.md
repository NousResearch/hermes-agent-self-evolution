# Thick-skill body text couples to a capable agent's convention adherence (GREEN)

**Verdict: GREEN.** On a capable agent (Claude Sonnet, headless `claude -p`), the
*body* of a thick (314-line) SOP-style skill **drives** the agent's adherence to
a non-inferable output convention. Removing the convention from the body — and
only the body — collapses adherence from 1.00 to 0.00, uniformly across five
diverse repositories, with zero run-to-run variance.

This is the **skill-body analogue of the non-inferable tool-description finding**
([[project_mcp_description_coupling]]): artifact text is decoupled from a capable
agent's behavior *except* where the text carries a signal the agent cannot infer
from the task. A thick skill's arbitrary output convention (an exact directory +
filenames) is exactly such a signal.

## What was tested

Substrate: the real `codebase-summary` skill (a user SOP). Its body MUST-mandates
an arbitrary convention: write docs to `.sop/summary/` with the exact filenames
`architecture / components / interfaces / data_models / workflows / dependencies
/ index .md`, using Mermaid. None of that is inferable from "document this repo."

Two arms, differing ONLY in whether the convention is specified:
- **INTACT** — the skill verbatim.
- **LESIONED** — every mention of the path + the 7 filenames + the index spec
  stripped from the body and replaced with "organize the documentation into
  whatever files and structure are most useful." The `description` frontmatter is
  held **byte-identical** across arms, so skill *selection* is unconfounded — only
  the body convention changes. A grep gate enforces zero convention literals
  survive in LESIONED (the lesion leaked once during authoring; the gate caught it).

Closed loop: each arm delivered as a candidate skill to a sandboxed `claude -p`
via the `--plugin-dir` adapter. Task prompt is goal-only ("use the codebase-summary
skill to generate documentation … work non-interactively, use the skill's
defaults") — it never names a file or path, and the non-interactive clause defuses
the skill's "MUST ask for parameters upfront" step that a headless run can't satisfy.

Oracle: zero-LM, hand-authored (not generated from the skill → no circularity).
A bash `test_command`, kept at an absolute path **outside** the fixture so the
LESIONED agent cannot read the convention out of it, scores PASS iff ≥6 of the 7
mandated docs exist non-empty under the mandated path (accepts `.sop/summary` or
`.summary` — the skill names both). Diagnostics decompose path-adherence from
name-adherence.

Gate: the shipped `probe_discrimination` (baseline=LESIONED, ceiling=INTACT) with
the per-task A/A flip floor — the framework's real discrimination labeler, not an
ad-hoc significance test.

## Result

| repo (lang/domain)     | LESIONED | INTACT | A/A flip | label          |
|------------------------|---------:|-------:|---------:|----------------|
| cli_wordcount (py CLI) |     0.00 |   1.00 |     0.00 | discriminative |
| etl_pipeline (py data) |     0.00 |   1.00 |     0.00 | discriminative |
| flask_todo (py web)    |     0.00 |   1.00 |     0.00 | discriminative |
| go_kv (go svc)         |     0.00 |   1.00 |     0.00 | discriminative |
| react_todo (js lib)    |     0.00 |   1.00 |     0.00 | discriminative |

5/5 discriminative (GREEN bar was ≥3/5). 80 runs, **$23.02**, Sonnet.

### Controls that make it trustworthy (verified in the pilot, full diagnostics)
- **Not a delivery/firing artifact:** the skill fired (`cl-candidate:codebase-summary`)
  in **both** arms, every run (fired_rate=1.00). The only variable was the body text.
- **Not an oracle artifact:** the `anywhere` decomposition showed LESIONED produces
  only ~2 of the 7 mandated *names* anywhere, and the `.sop/summary/` path **never** —
  it writes real docs, just not the convention. INTACT produces 7/7 under the path.
  The oracle was independently verified on synthetic good/bad/wrong-path/stub dirs.
- **Noise floor is genuinely ~0:** convention adherence is deterministic
  instruction-following, so the A/A flip is 0.00 — the 1.00 gain trivially clears it.
- **Infrastructure-robust:** a first run was swamped by transient API 529s; a retry
  wrapper + error-exclusion fixed it. The reported rates are over clean runs only.

## Honest scope (what this does and does not establish)

CAN claim: a thick skill's body text drives a capable agent's adherence to a
**non-inferable output convention** (arbitrary path + filenames) it cannot infer
from the goal. The effect is maximal (0→1), uniform across five diverse repos, and
zero-variance.

CANNOT claim:
- Anything about documentation **quality or content** — the oracle is presence-only.
- That skills change **reasoning, methodology, or approach** — the harder axis where
  the campaign's decoupling prior still stands. This is the *expected-GREEN* axis.
- That a **subtler** (non-maximal) lesion couples — this removed the whole convention.
- Generalization beyond Sonnet (model-specific, like all noise/coupling results).

So: this is the skill analogue of the tool-description result — an expected-direction
confirmation on the one axis most likely to couple — not a refutation of the broader
"artifact text is decoupled from quality/reasoning" thesis.

## Why it matters for the project's promise

It is the **first thick-skill coupling GREEN**, and it reopens skill-evolution as a
product on the convention surface: if specifying the convention drives behavior, a
degraded skill should be *recoverable* by evolution. The natural next test (G2) is
whether the actual `evolve_skill` GEPA loop + noise-aware deploy gate can evolve a
LESIONED skill back toward the convention and deploy the recovered skill — i.e.,
the closed self-improvement loop delivering measurable behavior gain on a real,
non-inferable surface.

## Reproduce

```
uv run python spikes/codebase_summary_coupling/make_skills.py   # derive + verify LESIONED
uv run python spikes/codebase_summary_coupling/probe.py --gate check   # free: oracle/arms/suite
source ~/.zshrc && uv run python spikes/codebase_summary_coupling/probe.py --gate smoke --reps 3 --max-cost-usd 8
source ~/.zshrc && uv run python spikes/codebase_summary_coupling/probe.py --gate full --reps 4 --n 5 --max-cost-usd 60
```

Spike (gitignored, local): `spikes/codebase_summary_coupling/`.
