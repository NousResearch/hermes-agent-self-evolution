"""Per-instance env-validity gate for the SWE-bench campaign. An instance becomes
an Organism only if, in our container: (1) the bug reproduces (F2P fail on the
buggy base), (2) the gold patch resolves it (F2P pass, P2P green), and (3) the gold
is in scope (single non-test file, surface-freeze clean). Else a Drop with a reason.
Every outcome carries gold LOC + hunk count for the honesty report. Verdict comes
from env.graded_report() (official grading, keyed to dataset ids); _eval_ok=False
(infra failure / timeout) is dropped as 'eval_error', never miscounted as a real
test failure. Mirrors evolution.code.harvest.validate_candidate."""
from __future__ import annotations

from dataclasses import dataclass

from evolution.code.freeze_check import DEFAULT_MIN_RETAIN_RATIO, freeze_violations
from evolution.code.swebench.loader import SWEInstance, files_in_patch, patch_hunks, patch_loc


@dataclass(frozen=True)
class Organism:
    instance: SWEInstance
    base_src: str
    bug_tests: tuple[str, ...]
    oracle_failures: frozenset
    gold_loc: int
    gold_hunks: int


@dataclass(frozen=True)
class Drop:
    instance_id: str
    reason: str  # gold_multifile | eval_error | bug_not_reproduced | gold_unresolved | gold_violates_freeze
    gold_loc: int = 0
    gold_hunks: int = 0


def _failures(rep: dict, scope: str) -> set[str]:
    return set(rep[scope]["failure"])


def validate_instance(instance: SWEInstance, env, *,
                      min_retain_ratio: float = DEFAULT_MIN_RETAIN_RATIO) -> "Organism | Drop":
    loc, hunks = patch_loc(instance.gold_patch), patch_hunks(instance.gold_patch)
    nontest = [p for p in files_in_patch(instance.gold_patch)
               if not p.startswith("tests/") and "/tests/" not in p]
    if len(nontest) != 1:
        return Drop(instance.instance_id, "gold_multifile", loc, hunks)

    base_src = env.base_source(instance.gold_file)

    # 1. buggy state (container is at base_commit): the bug must reproduce
    rep_buggy = env.graded_report()
    if not rep_buggy.get("_eval_ok", True):
        return Drop(instance.instance_id, "eval_error", loc, hunks)
    reproducible = tuple(sorted(set(instance.fail_to_pass) & _failures(rep_buggy, "FAIL_TO_PASS")))
    if not reproducible:
        return Drop(instance.instance_id, "bug_not_reproduced", loc, hunks)

    # 2. gold state: apply gold, grade, then restore the buggy source
    env.apply_patch(instance.gold_patch)
    gold_src = env.read_tool(instance.gold_file)
    rep_gold = env.graded_report()
    env.reset_file(instance.gold_file)
    if not rep_gold.get("_eval_ok", True):
        return Drop(instance.instance_id, "eval_error", loc, hunks)
    gold_f2p_fail = _failures(rep_gold, "FAIL_TO_PASS")
    bug_tests = tuple(t for t in reproducible if t not in gold_f2p_fail)
    if not bug_tests:
        return Drop(instance.instance_id, "gold_unresolved", loc, hunks)

    # 3. gold in scope: surface-freeze clean (single-file already checked)
    if freeze_violations(base_src, gold_src, min_retain_ratio=min_retain_ratio):
        return Drop(instance.instance_id, "gold_violates_freeze", loc, hunks)

    oracle_failures = frozenset(gold_f2p_fail | _failures(rep_gold, "PASS_TO_PASS"))
    return Organism(instance=instance, base_src=base_src, bug_tests=bug_tests,
                    oracle_failures=oracle_failures, gold_loc=loc, gold_hunks=hunks)
