# A worked example: what a Phase 2 run leaves behind

These files are the verbatim output directory of one real `hermes-evolve tools`
run, kept here so you can see the artifact format without having to execute a
run first. Every phase writes the same shape of directory.

| File | What it holds |
|---|---|
| `PULL_REQUEST.md` | The rendered PR body: scores, evidence, gate results, run provenance, diff |
| `baseline_descriptions.json` | Every tool description as it stood before the run |
| `evolved_descriptions.json` | Every tool description after the run |
| `changes.json` | Only the descriptions that actually moved |
| `metrics.json` | Per-split scores, per-example outcomes, statistics |
| `gates.json` | Each gate's status: `passed`, `failed`, or `unavailable` |
| `cross_tool_report.json` | Per-tool selection accuracy, used to catch a change that helps one tool by hurting another |

## What this example is not

It is a format sample, not a performance claim. Read the numbers before quoting
them: the run scored 1.000 to 1.000 on both splits, recorded no model calls, and
completed in 0.0s of optimization wall clock. It exercised the pipeline against a
toy fixture, so the only interesting thing in it is the diff, where a tool
description that had the same sentence repeated six times collapses to one.

Paths written inside `PULL_REQUEST.md` and `metrics.json` point at the temporary
directory the run used. They are left exactly as generated rather than rewritten
to match this location, because an artifact that has been edited after the fact
is no longer a record of anything.

For an actual measured result, run a phase against your own hermes-agent
checkout. `hermes-evolve status` will tell you first what is reachable and which
gates can run.
