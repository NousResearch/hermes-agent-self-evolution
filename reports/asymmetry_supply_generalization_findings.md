# Bet #4: does the oracle-bearing bug supply port to a second repo (httpx)?

Status: DRAFT finding for owner review (local). Cost ~$0 (no proposer; harvest + pytest validity only).
Data: `reports/bet4_httpx_supply.json` + the gitignored `reports/_bet4_*` probe artifacts.

## Question
The code-evolution campaign's organism supply came from ONE repo (hermes-agent, a tool registry). Does the oracle-bearing supply — (buggy parent, upstream-fix oracle, red→green bug tests) — port to a second mature Python library, `encode/httpx`? Pre-registered: **≥10 valid → portable CI-repair tool; <5 → Wall-2 reasserts on the supply side.**

## Method
Reuse the harvester (`harvest_candidates` + the red→green worktree validity filter) with an httpx test→source mapper (`tests/test_X.py → httpx/_X.py`; `tests/<sub>/test_*.py → httpx/_<sub>.py`). The mapper seam is ~30 LOC as predicted — but porting also required installing httpx's test-only deps (trio, trustme, uvicorn, cryptography) for the validity filter to run, which the "30-LOC" estimate missed. Three increasingly generous validity passes:
1. **Single-file** (the production model): write back the one mapped source file.
2. **Multi-file source**: write back ALL source files the fix touched.
3. **All touched tests**: write back all source + run EVERY test file the commit touched.

## Results
- **Structural supply is abundant**: **272** bug-fix-shaped candidates (commits touching source + a name-matched test) across 27 mapped targets — vs ~12–46 for hermes-agent.
- **Validity is ZERO across all three passes**: **0/35** single-file, **0/7** multi-file-source, **0/5** all-touched-tests. In every checked case the parent source **passes the fix's tests** (`par_fail = fix_fail = 0`) — the bug is never reproduced red→green.
- **Mechanism**: (a) 28/35 fixes are multi-file (several touch 13–19 source files), so reverting one file can't reintroduce the bug; (b) more fundamentally, the parent source passes the fix-version tests even when fully reverted and all touched test files run — so these source+test co-touching commits are dominated by **refactors / features / test-maintenance, not bugs with failing tests**. Example: the traced `_utils.py` candidate is literally *"Move utility functions from `_utils.py` to `_client.py`"* — a refactor, not a bug.

## Conclusion — pre-registered verdict CONFIRMED (<5 → Wall-2 reasserts on the supply side)
The clean oracle-bearing organism supply is a property of hermes-agent's **special tool-registry architecture** — isolated single-file tools, name-matched test files, and commits that are predominantly isolated bug fixes — **not a general feature of mature Python repos**. A conventional library (interdependent modules, multi-file fixes, cross-cutting tests, and co-touching commits dominated by non-bugs) does not reduce to the harvest's one-tool↔one-test organism model. **Porting the supply is not a ~30-LOC mapper; it requires SWE-bench-grade harvesting** (issue→PR linkage + a real fail→pass execution filter, which SWE-bench itself reports keeps only a small fraction of candidates). The code-evolution deliverable is a **single-(special-)codebase instrument, not a portable CI-repair tool.**

## Honest limitations
- One second repo (httpx), one heuristic. A repo structurally similar to hermes (a plugin/tool registry) might port more cheaply.
- The cleanly-running test files (test_utils/test_decoders/test_config) confirm "parent passes → not a red→green bug"; for server-needing test files (client/model) validity is additionally confounded by test-runnability in the harness. Both subsets yield 0 valid.
- I tested the LIGHTWEIGHT harvest (the "cheap porting" the bet asked about). A proper SWE-bench-style F2P harvest might recover organisms from httpx — but that substantial build is itself the finding: porting is not cheap.
- Says nothing about LIVE in-session reproducers (Wall-2's binding constraint), by design.

## Implication for the roadmap
Combined with the gate-gaming finding (the gate holds on the special-repo bugs, trust = oracle richness), the honest product framing is: **a single-codebase CI-repair instrument with an honest per-organism coverage map** — not a general self-improving agent, and not a portable tool without SWE-bench-grade harvesting investment. If portability is wanted, the next real bet is a proper F2P harvest (issue→PR + execution filter) on a curated repo set — a substantial build, not a quick probe.

## SWE-bench-grade follow-on probe (the real porting cost)
Tested the *proper* method on httpx — select the PR's **added** test functions (the bug-reproducing tests) and run those node-ids through fail→pass with a full multi-file source revert. Result: **inconclusive, but it surfaced the binding barrier.** 35/40 candidates failed with `WorktreeError: could not import 'httpx'` — older merge commits do not build/import in a single modern env (Python 3.13 + current deps). The 5 recent-enough commits that ran still showed the added tests passing on the reverted parent (no F2P), but that sample is tiny and recency-biased.

**Decisive lesson:** a proper harvest needs **per-commit-era environment reconstruction** (the right interpreter + pinned deps per commit) — exactly why SWE-bench/SWE-smith build per-instance environments, and why porting the supply is a major infrastructure effort, not a probe. **Recommendation:** do NOT rebuild this from scratch — **adopt SWE-smith's existing 50k-instance feed + per-repo environments** (~$1360 already spent upstream) if a portable supply is wanted. The project's durable contribution — the deploy GATE and the measured boundary — *composes* with an external feed; it does not need to manufacture one.

Env note: the validity filter needs the target repo's test deps (for httpx: trio/trustme/uvicorn/cryptography) and a local read-only clone of the source repo (configured via `$HERMES_REPO`).
