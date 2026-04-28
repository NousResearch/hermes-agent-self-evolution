# 🧬 Hermes Agent Self-Evolution

Production-safe self-improvement control plane for Hermes Agent skills.

It implements the loop:

```text
attempt -> trace -> failure -> eval -> candidate improvement -> benchmark/judge -> gate -> review bundle -> safe local promotion -> repeat
```

No GPU training. No auto-push. No auto-merge. The machine proposes; the human still owns the steering wheel. Civilization survives another sprint.

## Current V1 status

| Capability | Status |
|---|---:|
| SQLite control plane + content-addressed artifacts | ✅ built |
| Repo snapshots + skill target scanning | ✅ built |
| Golden eval datasets | ✅ built |
| Attempt trace ingestion | ✅ built |
| Failed-trace eval dataset generation | ✅ built |
| Run registration and resumable state | ✅ built |
| Deterministic offline execution | ✅ built |
| Model-backed synthesis | ✅ built |
| DSPy/GEPA optimizer strategy | ✅ built |
| Rubric and model-rubric scoring | ✅ built |
| Holdout gate + constraint checks | ✅ built |
| Review bundle export | ✅ built |
| Safe local promotion + PR draft text | ✅ built |
| One-command `loop once` | ✅ built |

## Install

```bash
git clone https://github.com/NousResearch/hermes-agent-self-evolution.git
cd hermes-agent-self-evolution
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Check the CLI:

```bash
.venv/bin/hermes-evolve --help
```

## Quick deterministic demo

1. Initialize local state:

```bash
.venv/bin/hermes-evolve --root .evolution-state init
```

2. Register a repo containing Hermes-style skills:

```bash
.venv/bin/hermes-evolve --root .evolution-state repo add hermes-agent --path /path/to/hermes-agent
.venv/bin/hermes-evolve --root .evolution-state repo snapshot hermes-agent
.venv/bin/hermes-evolve --root .evolution-state targets scan --repo hermes-agent
```

3. Run the one-command loop against the included demo traces:

```bash
.venv/bin/hermes-evolve --root .evolution-state loop once \
  --target skill:github-code-review \
  --trace-path examples/demo/failures.jsonl \
  --strategy deterministic \
  --scoring-strategy deterministic-rubric \
  --preferred-metric rubric_score \
  --export-out .evolution-review
```

Output includes a run id, gate decision, and review bundle path.

## DeepSeek V4 / DSPy-GEPA run

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

Model guidance:

```text
deepseek-v4-pro    candidate generation, GEPA optimizer, final/hard judge
deepseek-v4-flash  volume eval and fast checks
```

## Manual staged workflow

```bash
.venv/bin/hermes-evolve --root .evolution-state traces import \
  --target skill:github-code-review \
  --path examples/demo/failures.jsonl

.venv/bin/hermes-evolve --root .evolution-state traces dataset \
  --target skill:github-code-review

.venv/bin/hermes-evolve --root .evolution-state run skill \
  --target skill:github-code-review \
  --dataset DATASET_ID \
  --iterations 5

.venv/bin/hermes-evolve --root .evolution-state run execute RUN_ID \
  --strategy dspy-gepa \
  --provider deepseek \
  --optimizer-model deepseek-v4-pro \
  --eval-model deepseek-v4-flash \
  --extra-body-json '{"thinking":{"type":"disabled"}}' \
  --scoring-strategy model-rubric \
  --judge-model deepseek-v4-pro

.venv/bin/hermes-evolve --root .evolution-state run gate RUN_ID --preferred-metric rubric_score
.venv/bin/hermes-evolve --root .evolution-state run export RUN_ID --out .evolution-review
```

## Safe promotion

Dry-run first:

```bash
.venv/bin/hermes-evolve --root .evolution-state run apply RUN_ID --branch evolve/github-code-review-RUNID
```

Apply locally only after review:

```bash
.venv/bin/hermes-evolve --root .evolution-state run apply RUN_ID \
  --branch evolve/github-code-review-RUNID \
  --apply \
  --commit
```

Draft PR text:

```bash
.venv/bin/hermes-evolve --root .evolution-state run pr-draft RUN_ID --branch evolve/github-code-review-RUNID
```

Safety contract:

```text
No auto-push. No auto-merge.
PASS gate required by default.
HOLD export/apply requires explicit override.
Dirty target repos block non-dry-run apply unless explicitly overridden.
```

## Evidence artifacts

Review bundles contain:

```text
baseline_SKILL.md
evolved_SKILL.md
candidate.diff
manifest.json
APPLY.md
```

State and artifacts are inspectable under the selected `--root` directory.

## Docs

- Product workflow: `docs/V1_PRODUCT_WORKFLOW.md`
- Full-system blueprint: `docs/KARPATHY_LOOP_FULL_SYSTEM.md`
- Demo traces: `examples/demo/failures.jsonl`
- Original plan: `PLAN.md`

## License

MIT — © 2026 Nous Research
