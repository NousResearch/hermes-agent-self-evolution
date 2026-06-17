# GEPA Cycle 21 Report — LLM-as-Judge (Strategy A)

**Date:** 2026-06-11 20:51 UTC  
**Method:** Direct Parent Execution (Strategy A) — no API calls, no subagents  
**Selection:** Word-count-driven fallback (0 delta candidates available)  
**Skills evolved:** 5  
**Total delta:** +60 across all skills (avg +12.0 per skill)

## Selection Rationale

Delta-driven: 0 skills modified since last cycle (no prior cycle reports exist, last cycle predates this timestamp). Falling back to word-count-driven heuristic.

Top 5 candidates by 4-gap deficiency + word count, with category diversity:

| # | Skill | Category | Bytes | Gaps |
|---|-------|----------|-------|------|
| 1 | `llava` | mlops/models | 7,858 | 4/4 |
| 2 | `simpo-training` | mlops/training | 5,941 | 4/4 |
| 3 | `cron-noninteractive-guardrails` | devops | 5,885 | 4/4 |
| 4 | `calcom-api` | openclaw-imports | 4,879 | 4/4 |
| 5 | `nostrx` | social-media | 4,518 | 4/4 |

## Changes Applied

All 5 skills received the 4-gap template:
- **trigger_conditions** (YAML frontmatter): 10–13 phrase-match triggers each
- **When to Use** (body): 7 concrete use cases each
- **Not For** (body): 6 disambiguating entries, each with `→ use \`skill-name\` instead`
- **Pitfalls** (body): 10 numbered failure modes, bold title + em-dash explanation + recovery action
- **Version bump**: 1.0.0 → 1.1.0 (or added where missing)

### Per-Skill Details

**llava** (mlops/models) — Baseline: 19 → Evolved: 32 (+13)  
Added 13 triggers, 7 When to Use, 6 Not For, 10 pitfalls covering GPU OOM, tokenizer mismatch, image preprocessing, Gradio conflicts, conversation corruption, batch CUDA cache.

**simpo-training** (mlops/training) — Baseline: 24 → Evolved: 35 (+11)  
Added 10 triggers, 7 When to Use, 6 Not For. Converted old "Common issues" table into 10 numbered pitfalls: hinge loss on small datasets, gamma/beta ratio limits, LR sensitivity, Flash Attention verification, dataset split naming, sft_weight ceiling.

**cron-noninteractive-guardrails** (devops) — Baseline: 20 → Evolved: 32 (+12)  
Added 10 triggers, 7 When to Use, 6 Not For, 10 pitfalls: post-success ioctl, masked .env keys, brv --detach, bash -i flags, urllib IPv6 hang, tee/pipe subshells, unnecessary pty, missing timeouts, gh auth prompts, cron-model-optimization confusion.

**calcom-api** (openclaw-imports) — Baseline: 19 → Evolved: 32 (+13)  
Added 11 triggers, 7 When to Use, 6 Not For, 10 pitfalls: 401 auth, timezone off-by-hours, slot query params, UID vs ID confusion, webhook signature verification, 429 Retry-After units, eventTypeId mismatch, OAuth headers, past-time bookings, cursor pagination.

**nostrx** (social-media) — Baseline: 21 → Evolved: 32 (+11)  
Added 10 triggers, 7 When to Use, 6 Not For, 10 pitfalls: SSH key auth, pkill matching, stale Twitter creds, sync_state.json desync, silent relay failures, thread tweet limit, media upload formats, cron run pile-up, npub validation, venv activate in SSH.

## Aggregate Metrics

| Metric | Value |
|--------|-------|
| Baseline average | 20.6 / 50 |
| Evolved average | 32.6 / 50 |
| Average delta | +12.0 |
| Total delta | +60 |
| Total patch lines | 1,060 |

## Repos Updated

- **Live path:** `~/.hermes/skills/` (5 SKILL.md files)
- **hermes-agent repo:** `~/.hermes/hermes-agent/` on branch `gepa/phase1-skill-optimization-cycle-21` (5 new skill files)
- **Skills backup:** `~/hermes-skills/` (5 files synced)
- **Patches:** `~/hermes-agent-self-evolution/patches/<skill>/gepa-cycle-21.patch` (5 patches, 1,060 total lines)
- **Metrics:** `~/hermes-agent-self-evolution/reports/gepa_cycle_21_metrics.json`

## Notable Observations

1. **All 5 skills were new to the hermes-agent repo** — not tracked in HEAD. Staged with `git add` and diffs generated via `--cached`.
2. **nostrx already had legacy `triggers:`** — kept the existing structure and added `metadata.hermes.trigger_conditions` in parallel. The old triggers field will eventually be deprecated by the skill loader.
3. **simpo-training's "Common issues" section converted to numbered pitfalls** — the old table-based approach is inferior to numbered failure modes with bold titles and recovery actions.
4. **No API calls consumed** — all work done via direct parent execution with targeted `patch` calls. Zero cost, zero timeouts.

## Recommendations for Next Cycle

- Delta-driven should find these 5 skills as candidates (if other work hasn't touched them)
- Consider running the DSPy CLI pipeline with `--eval-source synthetic` on these skills for automated baseline scoring
- After 3+ cycles, start using session-based evaluation (`--eval-source sessiondb`) to catch real-world gaps
