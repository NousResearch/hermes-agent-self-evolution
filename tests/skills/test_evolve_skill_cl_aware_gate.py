"""Integration tests for the skill-side deploy-gate CL-aware branch.

Symmetric to tests/tools/test_evolve_tool_cl_aware_gate.py — mocks the
synthetic dataset builder + closed-loop cache so each test can pin a
saturation band and verify the deploy gate's branch behaviour plus
``gate_decision.json`` shape. No real LM calls.

Tests 1-10 mirror the tool-side suite. Tests 11-13 cover skill-specific
invariants:

  11. force_run is called with the skill BODY (not the full
      frontmatter+body file). Guards against the cache-key-mismatch
      silent failure where the evolved variant would be re-validated
      under a different key, double-spending ~$1-3 per run.
  12. abort paths produce ``evolved_FAILED.md`` (not ``.json``). The
      skill-side convention matches how baseline/evolved are written
      so post-run diff tooling continues to work.
  13. v4 skill-specific payload fields (``bap_max_growth``,
      ``bap_safety_margin``, ``eval_source``, ``fitness_profile``,
      ``proposer_mode``, ``knee_point.band_roster``) survive the v5
      bump in the CL-primary path.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from evolution.core.dataset_builder import EvalDataset, EvalExample
from evolution.core.saturation_check import SaturationReport
from evolution.skills.evolve_skill import evolve
from evolution.validation.report import (
    PhaseResult,
    TaskResult,
    ValidationReport,
    WinLoss,
)


# Demo SKILL.md used as the baseline. Kept tiny + stable so the
# growth-pct math in every test is predictable. Lengths:
#   frontmatter (between ---s) = "name: demo-skill\ndescription: a test skill" (42 chars)
#   body (after the second ---) = "Do X." (5 chars)
#   raw (full file content)    = 58 chars
# After reassemble_skill: "---\n{frontmatter}\n---\n\n{body}\n" = 53 + len(body) chars.
_SKILL_FRONTMATTER = "name: demo-skill\ndescription: a test skill"
_BASELINE_BODY = "Do X."
_BASELINE_RAW = f"---\n{_SKILL_FRONTMATTER}\n---\n\n{_BASELINE_BODY}\n"
assert len(_BASELINE_RAW) == 58, (
    f"Test pre-condition: baseline raw must be 58 chars, got {len(_BASELINE_RAW)}"
)


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    """Write a minimal SKILL.md so skill discovery succeeds."""
    skills_root = tmp_path / "skills"
    skill_path = skills_root / "demo-skill"
    skill_path.mkdir(parents=True)
    (skill_path / "SKILL.md").write_text(_BASELINE_RAW)
    return skills_root


def _fake_skill_dataset(n: int = 50) -> EvalDataset:
    """Build a real-shaped EvalDataset with n fake examples (no LM calls).

    Mirrors the helper in test_evolve_skill_saturation_preflight.py.
    Default n=50 yields 30/10/10 splits — holdout must be ≥
    EvolutionConfig.min_holdout_size (default 10) or evolve() aborts
    before the deploy-gate branch even runs.
    """
    examples = [
        EvalExample(task_input=f"task {i}", expected_behavior=f"rubric {i}")
        for i in range(n)
    ]
    return EvalDataset(
        train=examples[:30], val=examples[30:40], holdout=examples[40:50],
    )


def _fake_validation_report(
    *,
    baseline_pass: list[bool],
    evolved_pass: list[bool],
    evolved_abstain: Optional[list[bool]] = None,
) -> ValidationReport:
    """Build a ValidationReport with the given per-task verdicts.

    Mirrors what ClosedLoopFeedbackCache.force_run returns; ``evolved``
    is the only phase the deploy-gate branch actually reads (it pulls
    baseline pass-counts from the cached preflight data). Skill-side
    suites score via test_command rather than tool_calls_seq, so we
    leave tool_calls_seq empty.
    """
    n = len(baseline_pass)
    evolved_abstain = evolved_abstain or [False] * n
    assert len(evolved_pass) == n
    assert len(evolved_abstain) == n

    baseline_tasks = [
        TaskResult(
            task_id=f"task_{i}",
            passed=p,
            abstained=False,
            tool_calls_seq=[],
            duration_seconds=0.1,
        )
        for i, p in enumerate(baseline_pass)
    ]
    evolved_tasks = [
        TaskResult(
            task_id=f"task_{i}",
            passed=p,
            abstained=a,
            tool_calls_seq=[],
            duration_seconds=0.1,
            error="runner timeout" if a else None,
        )
        for i, (p, a) in enumerate(zip(evolved_pass, evolved_abstain))
    ]

    def _phase(tasks: list[TaskResult]) -> PhaseResult:
        n_p = sum(1 for t in tasks if t.passed and not t.abstained)
        n_f = sum(1 for t in tasks if not t.passed and not t.abstained)
        n_a = sum(1 for t in tasks if t.abstained)
        scored = n_p + n_f
        return PhaseResult(
            pass_rate=(n_p / scored) if scored else 0.0,
            n_passed=n_p,
            n_failed=n_f,
            n_abstained=n_a,
            tasks=tasks,
        )

    return ValidationReport(
        schema_version="1",
        tool="demo-skill",
        task_suite_path="fake_suite.jsonl",
        task_suite_sha256="0" * 64,
        baseline=_phase(baseline_tasks),
        evolved=_phase(evolved_tasks),
        delta=WinLoss(
            n_wins=0, n_losses=0, n_ties=n, pass_rate_change=0.0,
        ),
        decision="pass",
        decision_reasons=[],
    )


def _make_fake_gepa(evolved_body: str):
    """Build a fake dspy.GEPA whose ``compile()`` returns a module with
    the detailed_results shape the knee-point path expects."""

    class _FakeGEPA:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def compile(self, baseline_module, *, trainset, valset):
            fake_module = MagicMock()
            fake_module.detailed_results = SimpleNamespace(
                candidates=[fake_module],
                val_aggregate_scores=[1.0],
                best_idx=0,
            )
            fake_module.skill_text = evolved_body
            return fake_module

    return _FakeGEPA


# A few body strings hand-picked to keep growth_pct in the zones the
# tests need. baseline raw = 58. evolved_full = 53 + len(body).
#
# _LOW_GROWTH_BODY: growth_pct ≈ 5.2% → required_gain=1 → a +2 CL win
# clears CL-primary. evolved_full = 53 + 8 = 61, growth = (61-58)/58 = 5.17%.
_LOW_GROWTH_BODY = "Find X."  # 8 chars; under the 0.20 growth-free threshold.

# Default body for tests that don't care about growth: stays under the
# default non-inferiority static_ceiling and keeps the structure intact.
_EVOLVED_BODY = "Do X better."  # 12 chars.


@contextlib.contextmanager
def _patch_stack(
    *,
    sat_report: SaturationReport,
    fake_cache: Optional[MagicMock],
    holdout_baseline_mean: float = 0.95,
    holdout_evolved_mean: float = 0.96,
    holdout_n: int = 10,
    evolved_body: str = _EVOLVED_BODY,
):
    """Single context manager wrapping every seam patch each test needs.

    Tests stay focused on the band/cache/assertion they're verifying.
    """
    fake_builder = MagicMock()
    fake_builder.generate.return_value = _fake_skill_dataset()
    evolved_per = [holdout_evolved_mean] * holdout_n

    def _maybe_build(**kwargs):
        # Honour the real "no suite path → no cache" contract; if a test
        # forgets to pass a suite path the use_cl_primary branch can't
        # fire (None cache) instead of getting a confusingly-active mock.
        if kwargs.get("suite_path") is None:
            return None
        return fake_cache

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch(
            "evolution.skills.evolve_skill.SyntheticDatasetBuilder",
            return_value=fake_builder,
        ))
        stack.enter_context(patch(
            "evolution.skills.evolve_skill.saturation_preflight",
            return_value=sat_report,
        ))
        stack.enter_context(patch(
            "evolution.skills.evolve_skill._preflight_lm_credentials",
        ))
        stack.enter_context(patch(
            "evolution.skills.evolve_skill._maybe_build_closed_loop_cache_skill",
            side_effect=_maybe_build,
        ))
        stack.enter_context(patch(
            "evolution.skills.evolve_skill.dspy.GEPA",
            new=_make_fake_gepa(evolved_body),
        ))
        stack.enter_context(patch(
            "evolution.skills.evolve_skill._holdout_evaluate_with_metric",
            return_value=(holdout_evolved_mean, evolved_per),
        ))
        # In headless test envs stdin is non-TTY. For non-healthy bands
        # the orchestrator otherwise sys.exit(3)s before the deploy gate.
        stack.enter_context(patch(
            "evolution.skills.evolve_skill.is_non_interactive",
            return_value=False,
        ))
        stack.enter_context(patch(
            "evolution.skills.evolve_skill.interactive_confirm",
            return_value=True,
        ))
        yield


def _run_evolve(
    *,
    skill_dir: Path,
    extra_kwargs: Optional[dict] = None,
):
    """Invoke evolve() with the minimum kwargs every test in this module
    shares. Wraps the long, repetitive call so each test stays focused
    on the band/cache/assertion that's actually being exercised.

    output_dir is NOT a kwarg on the skill-side evolve(); the function
    hardcodes ``Path("output") / skill_name / timestamp``. Tests
    monkeypatch.chdir(tmp_path) before calling, so the output lands
    under ``tmp_path/output/demo-skill/<timestamp>/``.
    """
    kwargs = dict(
        skill_name="demo-skill",
        skill_source_dirs=[str(skill_dir)],
        iterations=1,
        eval_dataset_size=50,
        holdout_ratio=0.2,
        quality_gate="non-inferiority",
        closed_loop_suite_path=Path("/fake/suite.jsonl"),
        closed_loop_mode="feedback",
        closed_loop_in_valset=False,
        closed_loop_agent_model="openai/gpt-5-mini",
        max_total_cost_usd=5.0,
        skip_preflight=True,
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return evolve(**kwargs)


def _latest_gate_decision(tmp_path: Path) -> dict:
    """Find the most-recently-written gate_decision.json under
    ``tmp_path/output/demo-skill/<timestamp>/`` and return its payload.

    The skill-side evolve() hardcodes its output path, so tests can't
    pin a known location and must enumerate timestamp-named subdirs.
    """
    runs_root = tmp_path / "output" / "demo-skill"
    assert runs_root.exists(), f"No run output under {runs_root}"
    runs = sorted(runs_root.iterdir())
    assert runs, f"No timestamped run dirs under {runs_root}"
    payload_path = runs[-1] / "gate_decision.json"
    assert payload_path.exists(), f"No gate_decision.json at {payload_path}"
    return json.loads(payload_path.read_text())


def _latest_run_dir(tmp_path: Path) -> Path:
    runs_root = tmp_path / "output" / "demo-skill"
    runs = sorted(runs_root.iterdir())
    return runs[-1]


def _weak_signal_report() -> SaturationReport:
    """The one band that triggers the CL-aware deploy gate."""
    return SaturationReport(
        band="weak_signal",
        holdout_score=0.95,
        holdout_n=10,
        holdout_per_example=[0.95] * 10,
        closed_loop_score=5 / 7,
        closed_loop_n=7,
        # 5/7 baseline pass-rate — the deploy gate reads this list
        # verbatim to compute baseline_cl_passes.
        closed_loop_per_example=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        suggestions=[],
        thresholds={},
    )


def _healthy_report() -> SaturationReport:
    """No CL data needed; the band routes through the synthetic gate."""
    return SaturationReport(
        band="healthy",
        holdout_score=0.5,
        holdout_n=10,
        holdout_per_example=[0.5] * 10,
        closed_loop_score=None,
        closed_loop_n=None,
        closed_loop_per_example=None,
        suggestions=[],
        thresholds={},
    )


def _no_headroom_report(*, with_cl_data: bool) -> SaturationReport:
    """no_headroom band with optional CL data. CL-primary must NOT fire
    on no_headroom regardless of data presence."""
    cl_per = [1.0] * 7 if with_cl_data else None
    return SaturationReport(
        band="no_headroom",
        holdout_score=0.99,
        # holdout_n must match the _patch_stack holdout_n (10) so the
        # cached baseline list and the post-GEPA evolved list line up
        # for paired_bootstrap.
        holdout_n=10,
        holdout_per_example=[1.0] * 10,
        closed_loop_score=1.0 if with_cl_data else None,
        closed_loop_n=7 if with_cl_data else None,
        closed_loop_per_example=cl_per,
        suggestions=["Try a harder suite"],
        thresholds={},
    )


# ---------------------------------------------------------------------------
# Tests 1-10: mirror the tool-side suite
# ---------------------------------------------------------------------------


def test_weak_signal_band_triggers_evolved_cl_eval(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """weak_signal + +2 task win → force_run is called post-GEPA,
    decision == deploy, decision_signal == closed_loop, cl_tasks_gained == 2."""
    monkeypatch.chdir(tmp_path)
    fake_cache = MagicMock()
    # Baseline preflight per-example is [1]*5 + [0]*2 = 5/7.
    # Evolved 7/7 — a +2 task gain that beats required_gain at small
    # growth_pct.
    fake_cache.force_run.return_value = _fake_validation_report(
        baseline_pass=[True, True, True, True, True, False, False],
        evolved_pass=[True, True, True, True, True, True, True],
    )

    # _LOW_GROWTH_BODY keeps required_gain at 1 task so the +2 CL win
    # clears the cl_primary_gate.
    with _patch_stack(
        sat_report=_weak_signal_report(),
        fake_cache=fake_cache,
        evolved_body=_LOW_GROWTH_BODY,
    ):
        _run_evolve(skill_dir=skill_dir)

    fake_cache.force_run.assert_called_once_with(_LOW_GROWTH_BODY)

    payload = _latest_gate_decision(tmp_path)
    assert payload["decision"] == "deploy", (
        f"weak_signal + 5→7 should deploy, got {payload['decision']} "
        f"(reason: {payload.get('reason')})"
    )
    assert payload["decision_signal"] == "closed_loop"
    assert payload["cl_tasks_gained"] == 2


def test_healthy_band_does_not_trigger_cl_aware_gate(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """healthy band → CL-primary never fires; gate falls through to
    synthetic, force_run is NOT called post-GEPA, no CL fields written."""
    monkeypatch.chdir(tmp_path)
    fake_cache = MagicMock()

    with _patch_stack(sat_report=_healthy_report(), fake_cache=fake_cache):
        _run_evolve(skill_dir=skill_dir)

    fake_cache.force_run.assert_not_called()
    payload = _latest_gate_decision(tmp_path)
    assert payload["decision_signal"] == "synthetic"
    for cl_field in (
        "cl_tasks_gained",
        "cl_required_gain",
        "synthetic_sanity_check",
        "baseline_closed_loop_per_example",
        "evolved_closed_loop_per_example",
    ):
        assert cl_field not in payload, (
            f"CL field {cl_field!r} should not be in synthetic-gate payload"
        )


def test_no_headroom_falls_through_to_synthetic_gate(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """no_headroom + non-empty CL data → CL-primary STILL must NOT fire.
    The spec triggers CL-primary only on weak_signal."""
    monkeypatch.chdir(tmp_path)
    fake_cache = MagicMock()

    with _patch_stack(
        sat_report=_no_headroom_report(with_cl_data=True),
        fake_cache=fake_cache,
    ):
        _run_evolve(
            skill_dir=skill_dir,
            extra_kwargs={"force_saturation_check": True},
        )

    fake_cache.force_run.assert_not_called()
    payload = _latest_gate_decision(tmp_path)
    assert payload["decision_signal"] == "synthetic"
    for cl_field in (
        "cl_tasks_gained",
        "cl_required_gain",
        "synthetic_sanity_check",
    ):
        assert cl_field not in payload


def test_no_headroom_without_cl_data_falls_through_to_synthetic_gate(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """no_headroom + no CL data + --force-saturation-check → synthetic gate
    runs without KeyError. CL was never measured, so no CL fields."""
    monkeypatch.chdir(tmp_path)
    fake_cache = MagicMock()

    with _patch_stack(
        sat_report=_no_headroom_report(with_cl_data=False),
        fake_cache=fake_cache,
    ):
        _run_evolve(
            skill_dir=skill_dir,
            extra_kwargs={"force_saturation_check": True},
        )

    payload = _latest_gate_decision(tmp_path)
    assert payload["decision_signal"] == "synthetic"


def test_no_saturation_check_falls_through_to_synthetic_with_reason_recorded(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """--no-saturation-check → no preflight, falls through to synthetic.
    decision_signal == synthetic AND reason_synthetic == preflight_skipped
    so downstream consumers can distinguish 'preflight saw nothing weak'
    from 'preflight didn't run'."""
    monkeypatch.chdir(tmp_path)
    fake_cache = MagicMock()

    # sat_report is unused (skip_saturation_check=True bypasses preflight)
    # but _patch_stack requires one.
    with _patch_stack(sat_report=_healthy_report(), fake_cache=fake_cache):
        _run_evolve(
            skill_dir=skill_dir,
            extra_kwargs={"skip_saturation_check": True},
        )

    payload = _latest_gate_decision(tmp_path)
    assert payload["decision_signal"] == "synthetic"
    assert payload["reason_synthetic"] == "preflight_skipped"


def test_cl_primary_decision_persists_to_gate_decision_json(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """weak_signal → all v5 CL fields present in gate_decision.json with
    correct types. Pins the JSON contract downstream consumers depend on."""
    monkeypatch.chdir(tmp_path)
    fake_cache = MagicMock()
    fake_cache.force_run.return_value = _fake_validation_report(
        baseline_pass=[True, True, True, True, True, False, False],
        evolved_pass=[True, True, True, True, True, True, True],
    )

    # _LOW_GROWTH_BODY → required_gain=1 → +2 win clears the gate so
    # the deploy path populates every v5 CL field we're pinning here.
    with _patch_stack(
        sat_report=_weak_signal_report(),
        fake_cache=fake_cache,
        evolved_body=_LOW_GROWTH_BODY,
    ):
        _run_evolve(skill_dir=skill_dir)

    payload = _latest_gate_decision(tmp_path)

    assert payload["schema_version"] == "5"
    assert payload["decision_signal"] == "closed_loop"

    assert isinstance(payload["baseline_closed_loop_per_example"], list)
    assert all(
        isinstance(x, (int, float))
        for x in payload["baseline_closed_loop_per_example"]
    )
    assert isinstance(payload["evolved_closed_loop_per_example"], list)
    assert all(
        isinstance(x, (int, float))
        for x in payload["evolved_closed_loop_per_example"]
    )

    assert isinstance(payload["cl_tasks_gained"], int)
    assert isinstance(payload["cl_required_gain"], int)

    sanity = payload["synthetic_sanity_check"]
    assert isinstance(sanity, dict)
    for key in ("tolerance", "baseline_mean", "evolved_mean", "passed"):
        assert key in sanity, f"synthetic_sanity_check missing {key!r}"
    assert isinstance(sanity["tolerance"], (int, float))
    assert isinstance(sanity["baseline_mean"], (int, float))
    assert isinstance(sanity["evolved_mean"], (int, float))
    assert isinstance(sanity["passed"], bool)

    # cost_usd may be None (tests don't exercise the cost ledger), float,
    # or int — accept any; we only pin field presence here.
    assert "evolved_cl_eval_cost_usd" in payload
    cost = payload["evolved_cl_eval_cost_usd"]
    assert cost is None or isinstance(cost, (int, float))

    band_score = payload["band_trigger_score"]
    assert isinstance(band_score, dict)
    assert "holdout" in band_score
    assert "closed_loop" in band_score

    assert isinstance(payload["validator_agent_model"], str)


def test_synthetic_only_decision_unchanged_in_gate_decision_json(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """healthy → synthetic path. All v4 skill fields present alongside
    the new decision_signal marker, no CL fields leak in.

    The v4 skill-specific fields (``bap_max_growth``, ``bap_safety_margin``,
    ``eval_source``, ``fitness_profile``, ``proposer_mode``) MUST be
    preserved post-v5 bump.
    """
    monkeypatch.chdir(tmp_path)
    fake_cache = MagicMock()

    with _patch_stack(sat_report=_healthy_report(), fake_cache=fake_cache):
        _run_evolve(skill_dir=skill_dir)

    payload = _latest_gate_decision(tmp_path)

    assert payload["schema_version"] == "5"
    assert payload["decision_signal"] == "synthetic"

    # v4-and-earlier fields the synthetic path has always written.
    for required in (
        "baseline_per_example",
        "evolved_per_example",
        "bootstrap",
        "growth_pct",
        "required_improvement",
        "baseline_chars",
        "evolved_chars",
        "absolute_char_ceiling",
        "knee_point",
        "dataset",
        "run_inputs",
        # v4 skill-specific fields (the plan calls these out as needing
        # explicit preservation in test 7's assertion).
        "bap_max_growth",
        "bap_safety_margin",
        "fitness_profile",
        "proposer_mode",
    ):
        assert required in payload, f"missing v4 field {required!r}"

    assert payload["run_inputs"]["eval_source"] == "synthetic"

    for cl_field in (
        "cl_tasks_gained",
        "cl_required_gain",
        "synthetic_sanity_check",
        "baseline_closed_loop_per_example",
        "evolved_closed_loop_per_example",
        "band_trigger_score",
        "validator_agent_model",
    ):
        assert cl_field not in payload, (
            f"CL-only field {cl_field!r} leaked into synthetic-gate payload"
        )


def test_force_run_failure_writes_aborted_decision_with_diagnostic_payload(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """weak_signal + force_run raises → aborted decision,
    reason=cl_eval_failed, exception text recorded, evolved_FAILED.md
    written for forensic inspection of the rejected candidate."""
    monkeypatch.chdir(tmp_path)
    fake_cache = MagicMock()
    fake_cache.force_run.side_effect = RuntimeError("validator crashed")

    with _patch_stack(sat_report=_weak_signal_report(), fake_cache=fake_cache):
        _run_evolve(skill_dir=skill_dir)

    payload = _latest_gate_decision(tmp_path)
    assert payload["decision"] == "aborted"
    assert payload["reason"] == "cl_eval_failed"
    assert "validator crashed" in payload["cl_eval_exception"]

    run_dir = _latest_run_dir(tmp_path)
    assert (run_dir / "evolved_FAILED.md").exists(), (
        "evolved_FAILED.md must be written so the rejected variant "
        "is inspectable"
    )


def test_evolved_task_error_writes_cl_eval_incomplete_decision(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """weak_signal + one evolved task abstained → cl_eval_incomplete
    (NOT a regression). An infrastructure flake on the evolved phase
    isn't evidence of quality loss; conflating them would silently
    reject good candidates."""
    monkeypatch.chdir(tmp_path)
    fake_cache = MagicMock()
    # task_2 abstains; others pass. Without the incomplete-detection
    # branch this would score as 6/7 (+1 vs 5/7 baseline) and deploy.
    fake_cache.force_run.return_value = _fake_validation_report(
        baseline_pass=[True, True, True, True, True, False, False],
        evolved_pass=[True, True, False, True, True, True, True],
        evolved_abstain=[False, False, True, False, False, False, False],
    )

    with _patch_stack(sat_report=_weak_signal_report(), fake_cache=fake_cache):
        _run_evolve(skill_dir=skill_dir)

    payload = _latest_gate_decision(tmp_path)
    assert payload["decision"] == "aborted"
    assert payload["reason"] == "cl_eval_incomplete"
    assert payload["evolved_closed_loop_errored_tasks"] == ["task_2"]

    run_dir = _latest_run_dir(tmp_path)
    assert (run_dir / "evolved_FAILED.md").exists()


def test_absolute_char_ceiling_still_enforced_in_cl_primary_path(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """weak_signal + +2 CL win + evolved body exceeding the absolute
    char ceiling → reject. CL-primary mustn't bypass the wallpaper-
    protection backstop.

    Pinned production flow: ``validate_static`` at ``evolve_skill.py:1034``
    only runs ``size_limit``/``non_empty``/``skill_structure``; the
    ``absolute_char_ceiling`` check lives inside the CL-primary branch at
    line 1271 (``validator._check_absolute_chars``) and runs AFTER
    ``force_run``. So this test exercises the in-branch ceiling check —
    the rejection carries ``decision_signal: "closed_loop"`` and the CL
    cache must have been consulted for the evolved body.

    Baseline raw = 58 chars. With max_absolute_chars=50, the effective
    ceiling = max(50, 1.5*58) = 87. evolved_full = 53 + len(body) chars,
    so a 200-char body produces a 253-char evolved_full — trips the
    ceiling. Body stays under config.max_skill_size so non-ceiling
    static checks still pass.
    """
    monkeypatch.chdir(tmp_path)
    fake_cache = MagicMock()
    fake_cache.force_run.return_value = _fake_validation_report(
        baseline_pass=[True, True, True, True, True, False, False],
        evolved_pass=[True, True, True, True, True, True, True],
    )
    long_body = (
        "Find files in the repository by name pattern or glob; returns "
        "matching file paths from anywhere under the project root."
    ) * 2
    assert len(long_body) > 87 - 53, (
        f"Test pre-condition: long_body={len(long_body)} must trip the ceiling"
    )

    with _patch_stack(
        sat_report=_weak_signal_report(),
        fake_cache=fake_cache,
        evolved_body=long_body,
    ):
        _run_evolve(
            skill_dir=skill_dir,
            extra_kwargs={"max_absolute_chars": 50},
        )

    payload = _latest_gate_decision(tmp_path)
    assert payload["decision"] == "reject", (
        f"absolute_char_ceiling must reject even on a winning CL gate; "
        f"got decision={payload['decision']} (reason={payload.get('reason')})"
    )
    assert "absolute_char_ceiling" in payload.get("failed_constraints", []), (
        f"failed_constraints={payload.get('failed_constraints')}"
    )
    # Pin the actual code path: rejection comes from the in-branch
    # _check_absolute_chars at evolve_skill.py:1271 (NOT the early
    # validate_static at line 1034, which doesn't include the ceiling).
    # That means CL eval already ran and the signal is "closed_loop".
    assert payload["decision_signal"] == "closed_loop", (
        f"CL-primary ceiling reject must emit decision_signal='closed_loop'; "
        f"got {payload.get('decision_signal')!r} — did the ceiling check move "
        f"out of the CL-primary branch?"
    )
    fake_cache.force_run.assert_called_once_with(long_body)


# ---------------------------------------------------------------------------
# Tests 11-13: skill-specific guards
# ---------------------------------------------------------------------------


def test_force_run_called_with_skill_body_not_full(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """force_run must be called with the BODY (no YAML frontmatter), not
    with the full evolved file (frontmatter + body).

    The closed-loop cache keys its memoisation on the artifact text. The
    preflight populated the cache with ``skill["body"]``; if the post-GEPA
    eval site passes ``evolved_full`` instead, the cache key won't match
    and the validator silently double-spends ~$1-3 per run.

    This is the highest-value guard in the file — the failure mode is
    silent: the run still produces a decision, no error surfaces, but
    cost ledger 2x's and the CL "cache hit" telemetry goes haywire.
    """
    monkeypatch.chdir(tmp_path)
    fake_cache = MagicMock()
    fake_cache.force_run.return_value = _fake_validation_report(
        baseline_pass=[True, True, True, True, True, False, False],
        evolved_pass=[True, True, True, True, True, True, True],
    )

    with _patch_stack(
        sat_report=_weak_signal_report(),
        fake_cache=fake_cache,
        evolved_body=_LOW_GROWTH_BODY,
    ):
        _run_evolve(skill_dir=skill_dir)

    # The single, exact assertion this test exists to make: force_run
    # receives the body string only, never the full frontmatter+body file.
    fake_cache.force_run.assert_called_once_with(_LOW_GROWTH_BODY)
    call_arg = fake_cache.force_run.call_args.args[0]
    assert "---" not in call_arg, (
        f"force_run received the full frontmatter+body file (cache-key "
        f"mismatch bug): {call_arg!r}"
    )
    assert "name:" not in call_arg, (
        f"force_run received YAML frontmatter, not body alone: {call_arg!r}"
    )


def test_evolved_failed_md_written_not_json(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Abort paths produce ``evolved_FAILED.md`` (not ``.json``).

    The tool-side equivalent writes ``evolved_FAILED.json`` because tool
    manifests are JSON. Skills are markdown files, and post-run diff
    tooling reads ``evolved_FAILED.md`` to compare against ``baseline_skill.md``.
    A silent rename to ``.json`` would break that workflow.
    """
    monkeypatch.chdir(tmp_path)
    fake_cache = MagicMock()
    fake_cache.force_run.side_effect = RuntimeError("validator crashed")

    with _patch_stack(sat_report=_weak_signal_report(), fake_cache=fake_cache):
        _run_evolve(skill_dir=skill_dir)

    run_dir = _latest_run_dir(tmp_path)
    assert (run_dir / "evolved_FAILED.md").exists(), (
        f"evolved_FAILED.md missing — got {list(run_dir.iterdir())}"
    )
    assert not (run_dir / "evolved_FAILED.json").exists(), (
        "evolved_FAILED.json must NOT exist on skill-side aborts; "
        "skill convention is .md (matches baseline_skill.md). If someone "
        "intentionally added .json support, update this test deliberately."
    )

    # Verify the .md is the full reassembled file (frontmatter + body),
    # not the body alone — diff tooling expects evolved_FAILED.md to be
    # directly diffable against baseline_skill.md.
    failed_text = (run_dir / "evolved_FAILED.md").read_text()
    assert failed_text.startswith("---"), (
        f"evolved_FAILED.md should include frontmatter for diff parity, "
        f"got {failed_text[:80]!r}"
    )
    assert "name: demo-skill" in failed_text


def test_skill_v4_payload_fields_preserved_in_v5_cl_primary(
    skill_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Schema regression: ``bap_max_growth``, ``bap_safety_margin``,
    ``eval_source``, ``fitness_profile``, ``proposer_mode``, and
    ``knee_point.band_roster`` all present in v5 CL-primary output.

    These are the v4 skill-specific fields downstream calibration scripts
    read. Future schema bumps must keep them populated.
    """
    monkeypatch.chdir(tmp_path)
    fake_cache = MagicMock()
    fake_cache.force_run.return_value = _fake_validation_report(
        baseline_pass=[True, True, True, True, True, False, False],
        evolved_pass=[True, True, True, True, True, True, True],
    )

    with _patch_stack(
        sat_report=_weak_signal_report(),
        fake_cache=fake_cache,
        evolved_body=_LOW_GROWTH_BODY,
    ):
        _run_evolve(skill_dir=skill_dir)

    payload = _latest_gate_decision(tmp_path)
    assert payload["schema_version"] == "5"
    assert payload["decision_signal"] == "closed_loop"

    # Skill-specific v4 payload fields — must persist across v5.
    for field in (
        "bap_max_growth",
        "bap_safety_margin",
        "fitness_profile",
        "proposer_mode",
    ):
        assert field in payload, (
            f"v4 skill field {field!r} missing in v5 CL-primary payload"
        )
        assert payload[field] is not None, (
            f"v4 skill field {field!r} present but null in v5 CL-primary payload"
        )

    # eval_source is nested under run_inputs (not at top level).
    assert payload["run_inputs"]["eval_source"] == "synthetic"

    # knee_point.band_roster must serialise as a list (empty here; the
    # downstream calibration script accesses it via .get('band_roster', [])).
    knee = payload["knee_point"]
    assert isinstance(knee, dict)
    assert "band_roster" in knee, (
        f"knee_point.band_roster missing in v5 CL-primary payload; "
        f"knee_point keys: {list(knee.keys())}"
    )
    assert isinstance(knee["band_roster"], list)


def test_summary_panel_reflects_cl_decision_when_cl_primary_deploys(
    skill_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    """CL-primary deploy → summary line announces the CL gain instead of
    the synthetic delta. Without the CL-aware branch the panel says
    'did not improve' even though gate_decision.json deployed the artifact,
    so the operator gets a contradictory signal."""
    monkeypatch.chdir(tmp_path)
    fake_cache = MagicMock()
    fake_cache.force_run.return_value = _fake_validation_report(
        baseline_pass=[True, True, True, True, True, False, False],
        evolved_pass=[True, True, True, True, True, True, True],
    )

    # _weak_signal_report pins baseline holdout to 0.95; evolved=0.90
    # forces a negative synthetic improvement so the pre-change panel
    # would render 'did not improve' even though CL-primary just deployed.
    with _patch_stack(
        sat_report=_weak_signal_report(),
        fake_cache=fake_cache,
        evolved_body=_LOW_GROWTH_BODY,
        holdout_evolved_mean=0.90,
    ):
        _run_evolve(skill_dir=skill_dir)

    out = capsys.readouterr().out
    assert "CL gained +2" in out, f"missing CL-gain line in summary: {out!r}"
    assert "did not improve" not in out, (
        f"synthetic 'did not improve' line leaked through CL-primary deploy: {out!r}"
    )


def test_summary_panel_reflects_cl_decision_when_cl_primary_rejects(
    skill_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    """CL-primary reject → summary line explains the CL shortfall instead
    of falling back to the generic synthetic-rejected line."""
    monkeypatch.chdir(tmp_path)
    fake_cache = MagicMock()
    # 5/7 → 5/7: zero CL gain, required_gain stays at 1 → reject.
    fake_cache.force_run.return_value = _fake_validation_report(
        baseline_pass=[True, True, True, True, True, False, False],
        evolved_pass=[True, True, True, True, True, False, False],
    )

    with _patch_stack(
        sat_report=_weak_signal_report(),
        fake_cache=fake_cache,
        evolved_body=_LOW_GROWTH_BODY,
    ):
        _run_evolve(skill_dir=skill_dir)

    out = capsys.readouterr().out
    assert "CL gain 0 < required 1" in out, (
        f"missing CL-reject line in summary: {out!r}"
    )


def test_summary_panel_uses_synthetic_delta_when_not_cl_primary(
    skill_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    """healthy band → synthetic gate → existing 'improved/did not improve'
    wording is unchanged. Regression guard for the synthetic path.

    _healthy_report() pins baseline holdout to 0.5 (cached per-example);
    evolved=0.5 produces a zero synthetic delta that still clears the
    non-inferiority gate (within tolerance) so the deploy-path 'did not
    improve' line fires — that's the legacy branch we must preserve.
    """
    monkeypatch.chdir(tmp_path)
    fake_cache = MagicMock()

    with _patch_stack(
        sat_report=_healthy_report(),
        fake_cache=fake_cache,
        holdout_evolved_mean=0.5,
    ):
        _run_evolve(skill_dir=skill_dir)

    out = capsys.readouterr().out
    assert "did not improve" in out, (
        f"synthetic path must keep 'did not improve' on a zero delta: {out!r}"
    )
    assert "CL gained" not in out
    assert "CL gain" not in out
