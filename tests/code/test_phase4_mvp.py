"""Tests + runnable end-to-end demo for the Phase 4 verifier MVP.

Realistic target: a buggy ``clamp(value, low, high)`` whose low branch returns
``high`` instead of ``low``. The bug only manifests on ``value < low`` inputs, so
a partial fix can pass a visible suite that lacks such cases yet regress on the
sealed hidden suite that contains them. That is exactly the overfit-to-visible
failure the verifier layer exists to catch.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

from evolution.code.verifier_harness import CheckResult, CodeEvalTask, ToolActionSession
from evolution.code.phase4_mvp import (
    LineageArchive,
    ResidueCurriculum,
    StrictCodeFitnessHarness,
    evaluate_on_unseen_seed,
    evaluate_with_lineage,
    render_markdown,
    summarize_decision,
)

BUGGY_SRC = (
    "def clamp(value, low, high):\n"
    "    if value < low:\n"
    "        return high\n"  # BUG: should return low
    "    if value > high:\n"
    "        return high\n"
    "    return value\n"
)
CORRECT_OLD = "    if value < low:\n        return high\n"
CORRECT_NEW = "    if value < low:\n        return low\n"
COSMETIC_OLD = "    if value > high:\n        return high\n"
COSMETIC_NEW = "    if value > high:\n        return high  # clamp high\n"


# --- check construction --------------------------------------------------- #
def _load_clamp(workspace: Path):
    ns: dict = {}
    src = (workspace / "clamp.py").read_text(encoding="utf-8")
    exec(compile(src, "clamp.py", "exec"), ns)
    return ns["clamp"]


def _cases(seed: int, n: int, include_low: bool) -> list[tuple[int, int, int, int]]:
    rng = random.Random(seed)
    cases: list[tuple[int, int, int, int]] = []
    for _ in range(n):
        low = rng.randint(-5, 5)
        high = low + rng.randint(1, 10)
        if include_low and rng.random() < 0.5:
            value = low - rng.randint(1, 5)  # exercises the buggy branch
        else:
            value = rng.randint(low, high)
        expected = low if value < low else high if value > high else value
        cases.append((value, low, high, expected))
    if include_low and not any(v < lo for (v, lo, _h, _e) in cases):
        cases[0] = (-99, 0, 10, 0)  # guarantee at least one low case
    return cases


def _run_cases(workspace: Path, cases) -> tuple[float, bool]:
    fn = _load_clamp(workspace)
    passed = 0
    low_failed = False
    for value, low, high, expected in cases:
        try:
            got = fn(value, low, high)
        except Exception:
            got = object()
        if got == expected:
            passed += 1
        elif value < low:
            low_failed = True
    return passed / len(cases), low_failed


def _make_check(seed: int, n: int, include_low: bool):
    cases = _cases(seed, n, include_low)

    def check(workspace: Path) -> CheckResult:
        score, low_failed = _run_cases(workspace, cases)
        if score >= 1.0:
            return CheckResult.pass_(f"{len(cases)} cases passed", score=1.0)
        residues = ("clamp_low_failed",) if low_failed else ("cases_failed",)
        return CheckResult.fail(f"{score:.3f} of cases passed", residues, score=score)

    return check


def _task(name: str, visible_seed: int, hidden_seed: int, *, full_suite: bool = True,
          visible_low: bool = False) -> CodeEvalTask:
    return CodeEvalTask(
        name=name,
        workspace_files={"clamp.py": BUGGY_SRC},
        visible_check=_make_check(visible_seed, 6, include_low=visible_low),
        hidden_check=_make_check(hidden_seed, 12, include_low=True),
        full_suite_check=_make_check(hidden_seed + 777, 20, include_low=True)
        if full_suite
        else None,
    )


# --- candidate runners ---------------------------------------------------- #
def correct_runner(session: ToolActionSession) -> None:
    session.read_file("clamp.py")
    session.apply_patch("clamp.py", CORRECT_OLD, CORRECT_NEW)
    session.run_visible_check()


def cosmetic_runner(session: ToolActionSession) -> None:
    """Passes a no-low visible suite, leaves the low bug in place -> fails hidden."""
    session.read_file("clamp.py")
    session.apply_patch("clamp.py", COSMETIC_OLD, COSMETIC_NEW)
    session.run_visible_check()


def frozen_runner(session: ToolActionSession) -> None:
    session.read_file("clamp.py")
    session.run_visible_check()


# --- tests ---------------------------------------------------------------- #
def test_correct_candidate_accepted_on_unseen_seed():
    harness = StrictCodeFitnessHarness(require_full_suite=True)
    factory = lambda vs, hs: _task(f"clamp_{vs}_{hs}", vs, hs, visible_low=True)
    result = evaluate_on_unseen_seed(
        harness, factory,
        adaptive_runner=correct_runner, frozen_runner=frozen_runner,
        candidate_id="correct", visible_seed=1, hidden_seed=2,
    )
    assert result.decision.accepted
    assert result.decision.reason == "accepted"
    assert result.decision.adaptive.hidden.score > result.decision.frozen.hidden.score


def test_cosmetic_candidate_rejected_by_hidden_gate():
    harness = StrictCodeFitnessHarness(require_full_suite=True)
    task = _task("clamp_overfit", visible_seed=10, hidden_seed=20, visible_low=False)
    decision = harness.evaluate(
        task, adaptive_runner=cosmetic_runner, frozen_runner=frozen_runner,
        candidate_id="cosmetic",
    )
    # passes visible (no low cases) but regresses on sealed hidden cases
    assert decision.adaptive.visible.passed
    assert not decision.accepted
    assert decision.reason == "hidden_gate_failed"


def test_unseen_seed_must_differ():
    harness = StrictCodeFitnessHarness()
    factory = lambda vs, hs: _task("x", vs, hs)
    try:
        evaluate_on_unseen_seed(
            harness, factory, adaptive_runner=correct_runner,
            frozen_runner=frozen_runner, candidate_id="c",
            visible_seed=5, hidden_seed=5,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-held-out seeds")


def test_full_suite_required_blocks_free_pass():
    strict = StrictCodeFitnessHarness(require_full_suite=True)
    task = _task("no_suite", 1, 2, full_suite=False, visible_low=True)
    decision = strict.evaluate(
        task, adaptive_runner=correct_runner, frozen_runner=frozen_runner,
        candidate_id="correct",
    )
    assert not decision.accepted
    assert decision.reason == "full_suite_required"

    lenient = StrictCodeFitnessHarness(require_full_suite=False)
    decision2 = lenient.evaluate(
        _task("no_suite2", 1, 2, full_suite=False, visible_low=True),
        adaptive_runner=correct_runner, frozen_runner=frozen_runner,
        candidate_id="correct",
    )
    assert decision2.accepted  # base behavior: missing suite is a free pass


def test_residue_curriculum_hardens_visible_without_touching_gate():
    # a follow-up task for clamp_low_failed makes the *visible* surface include
    # low cases; the harness thresholds are identical objects throughout.
    curriculum = ResidueCurriculum()
    curriculum.register(
        "clamp_low_failed",
        lambda r: _task("clamp_hardened", visible_seed=31, hidden_seed=32, visible_low=True),
    )
    harness = StrictCodeFitnessHarness(require_full_suite=True)
    base = _task("clamp_base", visible_seed=10, hidden_seed=20, visible_low=False)
    decision = harness.evaluate(
        base, adaptive_runner=cosmetic_runner, frozen_runner=frozen_runner,
        candidate_id="cosmetic",
    )
    scheduled = curriculum.observe(decision)
    assert any(t.name == "clamp_hardened" for t in scheduled)

    followup = curriculum.next_task()
    assert followup is not None and followup.name == "clamp_hardened"
    # the cosmetic candidate now fails the *visible* gate on the hardened task
    redo = harness.evaluate(
        followup, adaptive_runner=cosmetic_runner, frozen_runner=frozen_runner,
        candidate_id="cosmetic2",
    )
    assert not redo.adaptive.visible.passed  # hardened visible catches it early


def test_lineage_archive_diff_and_replay():
    harness = StrictCodeFitnessHarness(require_full_suite=True)
    archive = LineageArchive()
    task = _task("clamp_lin", visible_seed=3, hidden_seed=4, visible_low=True)
    with tempfile.TemporaryDirectory() as tmp:
        decision, entry = evaluate_with_lineage(
            harness, task, adaptive_runner=correct_runner, frozen_runner=frozen_runner,
            candidate_id="correct", archive=archive, root=Path(tmp),
        )
        assert decision.accepted
        # the recorded patch is auditable
        diff = archive.unified_diff(entry)
        assert "return low" in diff and "return high" in diff
        # deterministic integrity replay + stronger evaluation replay
        assert archive.verify_replay(entry) is True
        assert archive.replay_evaluation(entry, task) is True


def test_summary_emitter_has_all_mvp_fields():
    harness = StrictCodeFitnessHarness(require_full_suite=True)
    task = _task("clamp_sum", visible_seed=7, hidden_seed=8, visible_low=True)
    with tempfile.TemporaryDirectory() as tmp:
        decision, entry = evaluate_with_lineage(
            harness, task, adaptive_runner=correct_runner, frozen_runner=frozen_runner,
            candidate_id="correct", archive=LineageArchive(), root=Path(tmp),
        )
    summary = summarize_decision(decision, rejected_count=2, lineage_entry=entry)
    for key in (
        "hidden_score_delta", "frozen_vs_adaptive", "patch_trace_digest",
        "rejected_candidate_count", "diff_summary", "lineage",
    ):
        assert key in summary
    md = render_markdown(summary)
    assert "Phase 4 verifier decision" in md and "hidden score delta" in md


# --- end-to-end demo ------------------------------------------------------ #
def _demo() -> None:
    print("=" * 70)
    print("PHASE 4 VERIFIER MVP - end-to-end run")
    print("=" * 70)
    harness = StrictCodeFitnessHarness(min_hidden_delta=0.0, require_full_suite=True)
    archive = LineageArchive()
    curriculum = ResidueCurriculum()
    curriculum.register(
        "clamp_low_failed",
        lambda r: _task("clamp_hardened", visible_seed=31, hidden_seed=32, visible_low=True),
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # 1) a cosmetic candidate that overfits a low-free visible suite
        base = _task("clamp_base", visible_seed=10, hidden_seed=20, visible_low=False)
        d1, e1 = evaluate_with_lineage(
            harness, base, adaptive_runner=cosmetic_runner, frozen_runner=frozen_runner,
            candidate_id="cosmetic", archive=archive, root=root / "r1",
        )
        print(f"\n[1] cosmetic candidate -> accepted={d1.accepted} reason={d1.reason}")
        print(f"    visible passed={d1.adaptive.visible.passed} "
              f"hidden score={d1.adaptive.hidden.score:.3f} (overfit caught by sealed gate)")
        scheduled = curriculum.observe(d1)
        print(f"    residues={curriculum.residue_counts()} -> scheduled follow-up: "
              f"{[t.name for t in scheduled]}")

        # 2) curriculum hands back a hardened task; correct candidate solves it
        followup = curriculum.next_task()
        d2, e2 = evaluate_with_lineage(
            harness, followup, adaptive_runner=correct_runner, frozen_runner=frozen_runner,
            candidate_id="correct", archive=archive, root=root / "r2", parent_id=e1.candidate_id,
        )
        print(f"\n[2] correct candidate on hardened task -> accepted={d2.accepted} "
              f"reason={d2.reason}")
        print(f"    frozen hidden={d2.frozen.hidden.score:.3f}  "
              f"adaptive hidden={d2.adaptive.hidden.score:.3f}  "
              f"delta={e2.hidden_delta:+.3f}")

        # 3) audit: lineage, diff, replay
        print(f"\n[3] lineage chain: "
              f"{[e.candidate_id for e in archive.lineage_of(e2.candidate_id)]}")
        print(f"    deterministic integrity replay: {archive.verify_replay(e2)}")
        print(f"    full-suite evaluation replay:    {archive.replay_evaluation(e2, followup)}")
        print("    audited patch diff:")
        for line in archive.unified_diff(e2).splitlines():
            print("      " + line)

        # 4) the issue/PR summary block
        summary = summarize_decision(
            d2, rejected_count=archive.rejected_count(), lineage_entry=e2,
        )
        print("\n[4] emitted PR summary:\n")
        print(render_markdown(summary))


if __name__ == "__main__":
    _demo()
