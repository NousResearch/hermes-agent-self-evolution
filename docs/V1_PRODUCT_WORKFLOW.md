# Hermes Agent Self-Evolution V1 Product Workflow

This is the production-safe loop now implemented in this repo:

```text
attempt traces
  -> failed-trace eval dataset
  -> registered evolution run
  -> deterministic / model-synthesis / DSPy-GEPA execution
  -> rubric or model-rubric scoring
  -> holdout gate
  -> review bundle
  -> optional local-only apply / PR draft
```

No auto-push. No auto-merge. Humans still own promotion. Shocking restraint from the robots.

## 1. Install

```bash
git clone https://github.com/NousResearch/hermes-agent-self-evolution.git
cd hermes-agent-self-evolution
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Use the console script:

```bash
.venv/bin/hermes-evolve --help
```

## 2. Initialize local state

```bash
.venv/bin/hermes-evolve --root .evolution-state init
```

This creates local SQLite state plus content-addressed artifact storage. It is deliberately boring. Boring survives production.

## 3. Register a target repo and scan skill targets

Point at a Hermes Agent repo or any repo containing `skills/**/SKILL.md` files:

```bash
.venv/bin/hermes-evolve --root .evolution-state repo add hermes-agent --path /path/to/hermes-agent
.venv/bin/hermes-evolve --root .evolution-state repo snapshot hermes-agent
.venv/bin/hermes-evolve --root .evolution-state targets scan --repo hermes-agent
.venv/bin/hermes-evolve --root .evolution-state targets list
```

Targets are referenced as:

```text
skill:<skill-name>
```

Example:

```text
skill:github-code-review
```

## 4. One-command deterministic demo

Core command shape:

```bash
hermes-evolve loop once
```

The included demo trace file is:

```text
examples/demo/failures.jsonl
```

Run one safe offline loop:

```bash
.venv/bin/hermes-evolve --root .evolution-state loop once \
  --target skill:github-code-review \
  --trace-path examples/demo/failures.jsonl \
  --strategy deterministic \
  --scoring-strategy deterministic-rubric \
  --preferred-metric rubric_score \
  --export-out .evolution-review
```

This performs:

```text
trace import -> dataset build -> run creation -> run execution -> rubric scoring -> gate -> review bundle export
```

No model key needed.

## 5. DeepSeek V4 DSPy-GEPA loop

Use this when you want the actual jet engine, not the wind-up toy.

```bash
export DEEPSEEK_API_KEY=...

.venv/bin/hermes-evolve --root .evolution-state loop once \
  --target skill:github-code-review \
  --trace-path examples/demo/failures.jsonl \
  --strategy dspy-gepa \
  --provider deepseek \
  --optimizer-model deepseek-v4-pro \
  --eval-model deepseek-v4-flash \
  --dspy-model-prefix openai \
  --extra-body-json '{"thinking":{"type":"disabled"}}' \
  --scoring-strategy model-rubric \
  --judge-model deepseek-v4-pro \
  --preferred-metric rubric_score \
  --export-out .evolution-review
```

Recommended model split:

```text
deepseek-v4-pro    candidate generation, GEPA optimizer, final/hard judge
deepseek-v4-flash  cheaper eval volume and fast checks
```

## 6. Inspect evidence

```bash
.venv/bin/hermes-evolve --root .evolution-state runs list
.venv/bin/hermes-evolve --root .evolution-state runs show RUN_ID
.venv/bin/hermes-evolve --root .evolution-state run export RUN_ID --out .evolution-review
```

Review bundle contents:

```text
baseline_SKILL.md
evolved_SKILL.md
candidate.diff
manifest.json
APPLY.md
```

## 7. Safe promotion

Core command shapes:

```bash
hermes-evolve run apply
hermes-evolve run pr-draft
```

Default promotion mode is dry-run:

```bash
.venv/bin/hermes-evolve --root .evolution-state run apply RUN_ID --branch evolve/github-code-review-RUNID
```

Actually write to a local branch only after reviewing the dry-run diff:

```bash
.venv/bin/hermes-evolve --root .evolution-state run apply RUN_ID \
  --branch evolve/github-code-review-RUNID \
  --apply \
  --commit
```

Draft PR text without pushing:

```bash
.venv/bin/hermes-evolve --root .evolution-state run pr-draft RUN_ID \
  --branch evolve/github-code-review-RUNID
```

Safety rules:

```text
- pass gate required by default
- HOLD export/apply requires explicit override
- dirty git repo blocks non-dry-run apply unless explicitly allowed
- no upstream push
- no merge
- rollback remains plain git
```

## 8. What “working product” means

V1 is working when this happens end-to-end:

```text
Given real failed Hermes traces,
the system builds an eval set,
runs deterministic/model/GEPA optimization,
judges baseline vs evolved on holdout,
gates the result,
exports a human-review bundle,
and optionally drafts/applies a local branch for human PR review.
```

That is now the product contract. Anything claiming more is wearing a cape indoors.
