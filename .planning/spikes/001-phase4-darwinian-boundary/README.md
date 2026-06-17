# SPK-001: Phase 4 Darwinian External CLI Boundary

## Brief

Phase 4 is the highest-risk HSE tier because it targets actual Hermes tool/core source code. This spike is intentionally planning/read-only: it validates the boundary that must exist before Darwinian Evolver can be used for code evolution.

## Spike table

| # | Spike | Validates (Given/When/Then) | Risk | Current verdict |
|---|---|---|---|---|
| SPK-001 | darwinian-external-cli-boundary | Given Darwinian Evolver is AGPL, when Phase 4 invokes it, then HSE preserves an external CLI/subprocess boundary and does not import upstream classes into Hermes-native code. | High | PARTIAL: boundary documented, not yet executable |
| SPK-002 | isolated-dependency-smoke | Given an isolated Phase 4 venv, when the dependency is installed there, then CLI/help smoke runs without touching active Hermes runtime/source. | Medium | NOT RUN: requires separate install/network approval |
| SPK-003 | signature-registry-freeze-checker | Given a candidate code diff, when static freeze checks run, then function signatures and registry.register calls are unchanged. | High | PLANNED |
| SPK-004 | first-known-bug-reproduction | Given one narrow known Hermes tool bug, when a reproduction test runs before mutation, then Phase 4 optimizes against an observable failing case. | High | PLANNED |

## Current evidence

- HSE repo branch: `hse/phase2e-closeout-phase3-prep`
- HSE repo head: `f489f5c`
- Repo status before artifact creation: clean
- Current `evolution/code/__init__.py`: placeholder only
- `darwinian_evolver` import in current environment: not installed
- Phase 3 upstream PR: <https://github.com/NousResearch/hermes-agent/pull/39233>
- Phase 3 PR status at gate time: open draft, mergeable, no checks reported
- Phase 3 full external benchmark execution: not performed

## External boundary facts checked

The Hermes Darwinian Evolver skill/docs describe the upstream tool as AGPL-3.0 and instruct Hermes to invoke it only via upstream CLI/subprocess/`uv run`, not by importing upstream classes into Hermes itself.

The upstream Imbue Darwinian Evolver README describes a population of organisms, mutation, and scoring loop for evolving solutions. The upstream license text is GNU Affero General Public License v3.

The HSE PLAN also records Darwinian Evolver as the Phase 4 code-evolution engine and marks its integration as external CLI only.

## Verdict: PARTIAL

### What worked

- Phase 4 should not proceed as execution yet.
- The external CLI/subprocess boundary is clear enough to encode as a Seed constraint.
- The first safe next action is design/scaffold work, not code mutation.

### What did not run

- No Darwinian Evolver installation.
- No import smoke beyond confirming the current environment lacks the package.
- No code mutation.
- No Hermes source edits.
- No benchmark spend.

### Recommendation for the real build

Proceed in this order:

1. Keep this spike read-only until Sunwoo approves isolated dependency smoke.
2. Build a Phase 4 scaffold that writes only under a fresh Phase 4 output root.
3. Implement freeze checkers before any optimizer mutation:
   - function signature freeze;
   - `registry.register()` freeze;
   - target file allowlist;
   - safety/error-handling removal detector.
4. Select one narrow known bug or adversarial edge case and write the failing reproduction test first.
5. Only then request approval for a bounded isolated Darwinian Evolver CLI smoke.

## Stop conditions

Stop and request confirmation if the next action would:

- install or upgrade external dependencies;
- run networked benchmark commands;
- run Darwinian Evolver mutation;
- edit Hermes Agent source files;
- push or publish an evolved-code branch/PR;
- alter gateway, credentials, memory, skills, SOUL, or active runtime config.
