"""Manual end-to-end smoke for skill-side closed-loop validation.

Why this exists:
  Unit tests mock the validator so they catch wiring bugs without paying
  for real LM calls. They don't exercise the real Hermes agent against
  the real planted-bug suite, so a regression in (e.g.) how Hermes
  discovers skills inside the per-task sandbox, or how `python
  test_solution.py` actually scores in the validator, would slip
  through CI. This smoke runs the entire closed-loop layer end-to-end
  against:
    - the checked-in fake target skill at tests/fixtures/skills/systematic_debugging/
    - the checked-in planted-bug suite at evolution/validation/suites/systematic_debugging.jsonl
    - whatever Hermes is configured with (uses the user's real ~/.hermes/config.yaml)

  Drives one ClosedLoopValidator.validate() call directly — 5 tasks ×
  2 phases (baseline + evolved) = 10 hermes -z invocations. With a
  cheap eval model (gpt-4o-mini-ish) this is pennies, not dollars.

How to run:
  uv run python tests/manual/skill_closed_loop_smoke.py

  Exits 0 on success. Not part of CI — heavyweight (drives real
  hermes -z subprocesses + real LM spend).

Prerequisites:
  - `hermes` on PATH and configured (`hermes model` to select one)
  - Whatever credentials Hermes needs in `~/.hermes/auth.json`
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
FAKE_SKILL_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "skills" / "systematic_debugging" / "SKILL.md"
)
SUITE_PATH = (
    REPO_ROOT / "evolution" / "validation" / "suites" / "systematic_debugging.jsonl"
)
SKILL_NAME = "systematic_debugging"


# A deliberately different candidate body so the validator has something
# to do — also serves as a smoke for the structural-difference path.
EVOLVED_BODY = (
    "# Systematic debugging — evolved\n"
    "\n"
    "When a test fails:\n"
    "1. Read the test to understand the spec.\n"
    "2. Read the buggy code with the spec in mind.\n"
    "3. Form a hypothesis about which line is wrong.\n"
    "4. Make the smallest change you can.\n"
    "5. Re-run the test. If it passes, stop. If not, revert and try again.\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _watermark_tempdirs(prefix_substrings: list[str]) -> set[Path]:
    tmp = Path(tempfile.gettempdir())
    found: set[Path] = set()
    for entry in tmp.iterdir():
        if any(sub in entry.name for sub in prefix_substrings):
            found.add(entry)
    return found


def _section(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


# ---------------------------------------------------------------------------
# The smoke
# ---------------------------------------------------------------------------


def main() -> int:
    _section("Pre-flight checks")
    if not FAKE_SKILL_PATH.is_file():
        print(f"  ✗ Missing fixture skill at {FAKE_SKILL_PATH}")
        return 1
    if not SUITE_PATH.is_file():
        print(f"  ✗ Missing suite at {SUITE_PATH}")
        return 1
    if shutil.which("hermes") is None:
        print(
            "  ✗ `hermes` not on PATH. Install hermes-agent first "
            "(see README); this smoke drives real hermes -z calls."
        )
        return 1
    if shutil.which("python") is None:
        print("  ✗ `python` not on PATH. Suite tasks use `python test_solution.py`.")
        return 1
    print(f"  ✓ Fixture skill present: {FAKE_SKILL_PATH}")
    print(f"  ✓ Suite present: {SUITE_PATH}")
    print(f"  ✓ hermes binary present: {shutil.which('hermes')}")

    _section("Constructing closed-loop cache")
    from evolution.skills.evolve_skill import _maybe_build_closed_loop_cache_skill
    from evolution.skills.skill_module import load_skill

    skill = load_skill(FAKE_SKILL_PATH)
    print(f"  Baseline body ({len(skill['body'])} chars):")
    print("  ┌" + "─" * 58)
    for line in skill["body"].splitlines()[:5]:
        print(f"  │ {line}")
    if len(skill["body"].splitlines()) > 5:
        print("  │ ...")
    print("  └" + "─" * 58)

    tempdirs_before = _watermark_tempdirs(["cl_skill_workdir_", "cl_feedback_"])
    cache = _maybe_build_closed_loop_cache_skill(
        skill_name=SKILL_NAME,
        skill_path=FAKE_SKILL_PATH,
        baseline_skill_body=skill["body"],
        suite_path=SUITE_PATH,
        saturation_threshold=0.95,
        min_iters=1,
        window_size=4,
        gate_mode="always",  # force the validator to fire
    )
    assert cache is not None, "cache should be constructed when suite_path is set"
    print(f"  ✓ Cache constructed (gate_mode={cache.gate_mode})")
    installer = cache._validator.installer
    print(f"  ✓ Installer skills_src: {installer.skills_src}")
    target = installer.skills_src / SKILL_NAME / "SKILL.md"
    assert target.is_file(), f"baseline skill not copied into workdir: {target}"
    print(f"  ✓ Baseline staged at: {target}")

    _section("Running validator (5 tasks × 2 phases = 10 hermes -z calls)")
    print("  This invokes hermes -z 10 times. Wait ~60-180s depending on LM…\n")
    start = time.time()
    report = cache.get_or_run(EVOLVED_BODY)
    elapsed = time.time() - start
    print(f"\n  ✓ Validator completed in {elapsed:.1f}s")

    if report is None:
        print(
            "  ✗ Validator returned None — gate closed or validator raised. "
            "Re-run with logging at WARNING+ to see the cause."
        )
        return 1

    _section("Report")
    print(f"  Decision: {report.decision}")
    for reason in report.decision_reasons:
        print(f"    • {reason}")
    print(
        f"\n  Baseline:  {report.baseline.n_passed}/"
        f"{report.baseline.n_passed + report.baseline.n_failed} "
        f"({report.baseline.pass_rate:.0%})"
    )
    print(
        f"  Evolved:   {report.evolved.n_passed}/"
        f"{report.evolved.n_passed + report.evolved.n_failed} "
        f"({report.evolved.pass_rate:.0%})"
    )
    print(f"  Δ:         {report.delta.pass_rate_change:+.2f}")
    print(
        f"  Per-task:  {report.delta.n_wins} wins, "
        f"{report.delta.n_losses} losses, {report.delta.n_ties} ties"
    )

    print("\n  Per-task verdicts (baseline → evolved):")
    b_by_id = {t.task_id: t for t in report.baseline.tasks}
    for ev in report.evolved.tasks:
        b = b_by_id.get(ev.task_id)
        b_mark = "✓" if (b and b.passed) else ("◌" if (b and b.abstained) else "✗")
        e_mark = "✓" if ev.passed else ("◌" if ev.abstained else "✗")
        print(f"    [{b_mark} → {e_mark}] {ev.task_id}")

    _section("Cache + temp dir housekeeping")
    second_report = cache.get_or_run(EVOLVED_BODY)
    assert second_report is report, (
        "second get_or_run with same candidate should be a cache hit"
    )
    print("  ✓ Candidate-text cache hit on second call (no re-run)")

    tempdirs_after = _watermark_tempdirs(["cl_skill_workdir_", "cl_feedback_"])
    leaked = tempdirs_after - tempdirs_before
    # Cache + installer workdirs persist until process exit by design.
    # Count them — they should appear here. What we DON'T want to see is
    # cl_fixture_ or cl_hermes_home_ persisting (those are per-task tmpdirs
    # that the validator/runner cleans up in their own scope).
    leaked_per_task = _watermark_tempdirs(["cl_fixture_", "cl_hermes_home_"])
    print(f"  New persistent workdirs (expected, ≤ 2): {len(leaked)}")
    print(f"  Orphaned per-task tmpdirs (should be 0): {len(leaked_per_task)}")
    if leaked_per_task:
        print(f"  ✗ Orphaned: {leaked_per_task}")
        return 1

    _section("All assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
