"""Integration tests for the deploy-gate CL-aware branch.

Mocks the synthetic dataset builder + closed-loop cache so each test
can pin a saturation band and verify the deploy gate's branch behavior
plus ``gate_decision.json`` shape. No real LM calls.

Pairs with unit tests at ``tests/core/test_check_cl_primary_gate.py``
which cover the decision-rule math in isolation. These tests run the
full ``evolve()`` orchestrator end-to-end with seams stubbed at the
saturation pre-flight, closed-loop cache, GEPA, knee-point, and holdout
evaluator, so they exercise the branch logic added in this PR rather
than the helper math.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from evolution.core.dataset_builder import EvalExample
from evolution.core.saturation_check import SaturationReport
from evolution.skills.knee_point import CandidatePick
from evolution.tools.evolve_tool import evolve
from evolution.validation.report import (
    PhaseResult,
    TaskResult,
    ValidationReport,
    WinLoss,
)


FIXTURES = Path(__file__).parent.parent / "fixtures" / "tool_manifests"


@pytest.fixture
def temp_manifest(tmp_path: Path) -> Path:
    """Copy multiple_tools.json to a tmp location."""
    src = FIXTURES / "multiple_tools.json"
    dst = tmp_path / "manifest.json"
    dst.write_text(src.read_text())
    return dst


def _fake_tool_examples(n: int = 30) -> list[EvalExample]:
    """Build n fake EvalExamples without calling an LM."""
    return [
        EvalExample(task_input=f"task {i}", expected_behavior=f"rubric {i}")
        for i in range(n)
    ]


def _fake_validation_report(
    *,
    baseline_pass: list[bool],
    evolved_pass: list[bool],
    evolved_abstain: Optional[list[bool]] = None,
) -> ValidationReport:
    """Build a ValidationReport with the given per-task verdicts.

    Mirrors what ClosedLoopFeedbackCache.force_run returns; ``evolved``
    is the only phase the deploy-gate branch actually reads (it pulls
    baseline pass-counts from the cached preflight data).
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
        tool="search_files",
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


def _make_knee_pick(evolved_description: str) -> CandidatePick:
    """Build a CandidatePick that select_knee_point would return."""
    fake_module = MagicMock()
    return CandidatePick(
        module=fake_module,
        skill_text=evolved_description,
        body_chars=len(evolved_description),
        val_score=0.8,
        val_rank_in_band=1,
        band_size=1,
        epsilon=0.1,
        fallback="knee",
        picked_idx=0,
        gepa_default_idx=0,
        gepa_default_body_chars=len(evolved_description),
        band_roster=[],
    )


def _make_fake_gepa(evolved_description: str):
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
            fake_module.description_text = evolved_description
            return fake_module

    return _FakeGEPA


# Baseline description for search_files in multiple_tools.json is
# "Find things." (12 chars). With static_ceiling=5000 (default preset),
# effective_absolute_char_ceiling = max(5000, 1.5*12) = 5000 — so a
# plausible-length evolved description passes by default.
_EVOLVED_DESCRIPTION = (
    "Find files in the repository by name or glob pattern. "
    "Returns matching file paths."
)

# CL-primary path tests that want the gate to ACCEPT need growth_pct
# below CL_PRIMARY_GROWTH_FREE_THRESHOLD (0.20) so required_gain stays
# at 1 task and a +2 win clears it. 12-char baseline × 1.20 = 14.4, so
# evolved must be ≤ 14 chars. "Locate files." is 13 chars (8.3% growth)
# which lands required_gain=1.
_LOW_GROWTH_EVOLVED = "Locate files."


@contextlib.contextmanager
def _patch_stack(
    *,
    sat_report: SaturationReport,
    fake_cache: Optional[MagicMock],
    holdout_baseline_mean: float = 0.95,
    holdout_evolved_mean: float = 0.96,
    holdout_n: int = 10,
    evolved_description: str = _EVOLVED_DESCRIPTION,
):
    """Single context manager wrapping every seam patch each test needs.

    Tests stay focused on the band/cache/assertion they're verifying.
    """
    fake_builder = MagicMock()
    fake_builder.generate_tool_selection.return_value = _fake_tool_examples()
    knee_pick = _make_knee_pick(evolved_description)
    evolved_per = [holdout_evolved_mean] * holdout_n

    def _maybe_build(**kwargs):
        # Honour the real "no suite path → no cache" contract; if a test
        # forgets to pass a suite path the use_cl_primary branch can't fire
        # (None cache) instead of getting a confusingly-active mock.
        if kwargs.get("suite_path") is None:
            return None
        return fake_cache

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch(
            "evolution.tools.evolve_tool.SyntheticDatasetBuilder",
            return_value=fake_builder,
        ))
        stack.enter_context(patch(
            "evolution.tools.evolve_tool.saturation_preflight",
            return_value=sat_report,
        ))
        stack.enter_context(patch(
            "evolution.tools.evolve_tool._preflight_lm_credentials",
        ))
        stack.enter_context(patch(
            "evolution.tools.evolve_tool._maybe_build_closed_loop_cache",
            side_effect=_maybe_build,
        ))
        stack.enter_context(patch(
            "evolution.tools.evolve_tool.dspy.GEPA",
            new=_make_fake_gepa(evolved_description),
        ))
        stack.enter_context(patch(
            "evolution.tools.evolve_tool.select_knee_point",
            return_value=knee_pick,
        ))
        stack.enter_context(patch(
            "evolution.tools.evolve_tool._candidate_description",
            return_value=evolved_description,
        ))
        stack.enter_context(patch(
            "evolution.tools.evolve_tool._holdout_evaluate_with_metric",
            return_value=(holdout_evolved_mean, evolved_per),
        ))
        # In headless test envs stdin is non-TTY. For non-healthy bands
        # the orchestrator otherwise sys.exit(3)s before the deploy gate.
        stack.enter_context(patch(
            "evolution.tools.evolve_tool.is_non_interactive",
            return_value=False,
        ))
        stack.enter_context(patch(
            "evolution.tools.evolve_tool.interactive_confirm",
            return_value=True,
        ))
        yield


def _run_evolve(
    *,
    manifest_path: Path,
    output_dir: Path,
    extra_kwargs: Optional[dict] = None,
):
    """Invoke evolve() with the minimum kwargs every test in this module
    shares. Wraps the long, repetitive call so each test stays focused
    on the band/cache/assertion that's actually being exercised."""
    kwargs = dict(
        tool_name="search_files",
        manifest_path=manifest_path,
        iterations=1,
        eval_dataset_size=30,
        holdout_ratio=0.5,
        quality_gate="non-inferiority",
        closed_loop_suite_path=Path("/fake/suite.jsonl"),
        closed_loop_hermes_repo=Path("/fake/hermes"),
        # mode="feedback" avoids _load_behavioral_examples_from_suite,
        # which would read the suite file on disk. The deploy-gate
        # CL-primary branch is mode-agnostic; it pulls verdicts via
        # closed_loop_cache.force_run regardless.
        closed_loop_mode="feedback",
        closed_loop_in_valset=False,
        closed_loop_agent_model="openai/gpt-5-mini",
        max_total_cost_usd=5.0,
        skip_preflight=True,
        output_dir=output_dir,
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return evolve(**kwargs)


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
# The 10 tests
# ---------------------------------------------------------------------------


def test_weak_signal_band_triggers_evolved_cl_eval(
    temp_manifest: Path, tmp_path: Path,
):
    """weak_signal + +2 task win → force_run is called post-GEPA,
    decision == deploy, decision_signal == closed_loop, cl_tasks_gained == 2."""
    fake_cache = MagicMock()
    # Baseline preflight per-example is [1]*5 + [0]*2 = 5/7.
    # Evolved 7/7 — a +2 task gain that beats required_gain at small
    # growth_pct.
    fake_cache.force_run.return_value = _fake_validation_report(
        baseline_pass=[True, True, True, True, True, False, False],
        evolved_pass=[True, True, True, True, True, True, True],
    )
    run_dir = tmp_path / "run"

    # _LOW_GROWTH_EVOLVED keeps required_gain at 1 task so the +2 CL win
    # clears the cl_primary_gate.
    with _patch_stack(
        sat_report=_weak_signal_report(),
        fake_cache=fake_cache,
        evolved_description=_LOW_GROWTH_EVOLVED,
    ):
        result = _run_evolve(manifest_path=temp_manifest, output_dir=run_dir)

    fake_cache.force_run.assert_called_once_with(_LOW_GROWTH_EVOLVED)

    payload = json.loads((run_dir / "gate_decision.json").read_text())
    assert payload["decision"] == "deploy", (
        f"weak_signal + 5→7 should deploy, got {payload['decision']} "
        f"(reason: {payload.get('reason')})"
    )
    assert payload["decision_signal"] == "closed_loop"
    assert payload["cl_tasks_gained"] == 2
    # The deploy result echoes the metrics dict, not the gate decision.
    assert isinstance(result, dict)


def test_healthy_band_does_not_trigger_cl_aware_gate(
    temp_manifest: Path, tmp_path: Path,
):
    """healthy band → CL-primary never fires; gate falls through to
    synthetic, force_run is NOT called post-GEPA, no CL fields written."""
    fake_cache = MagicMock()
    run_dir = tmp_path / "run"

    with _patch_stack(sat_report=_healthy_report(), fake_cache=fake_cache):
        _run_evolve(manifest_path=temp_manifest, output_dir=run_dir)

    fake_cache.force_run.assert_not_called()
    payload = json.loads((run_dir / "gate_decision.json").read_text())
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


def test_no_headroom_without_cl_data_falls_through_to_synthetic_gate(
    temp_manifest: Path, tmp_path: Path,
):
    """no_headroom + no CL data + --force-saturation-check → synthetic gate
    runs without KeyError. CL was never measured, so no CL fields."""
    fake_cache = MagicMock()
    run_dir = tmp_path / "run"

    with _patch_stack(
        sat_report=_no_headroom_report(with_cl_data=False),
        fake_cache=fake_cache,
    ):
        _run_evolve(
            manifest_path=temp_manifest,
            output_dir=run_dir,
            extra_kwargs={"force_saturation_check": True},
        )

    payload = json.loads((run_dir / "gate_decision.json").read_text())
    assert payload["decision_signal"] == "synthetic"


def test_no_headroom_with_cl_data_falls_through_to_synthetic_gate(
    temp_manifest: Path, tmp_path: Path,
):
    """no_headroom + non-empty CL data → CL-primary STILL must NOT fire.
    The spec triggers CL-primary only on weak_signal."""
    fake_cache = MagicMock()
    run_dir = tmp_path / "run"

    with _patch_stack(
        sat_report=_no_headroom_report(with_cl_data=True),
        fake_cache=fake_cache,
    ):
        _run_evolve(
            manifest_path=temp_manifest,
            output_dir=run_dir,
            extra_kwargs={"force_saturation_check": True},
        )

    fake_cache.force_run.assert_not_called()
    payload = json.loads((run_dir / "gate_decision.json").read_text())
    assert payload["decision_signal"] == "synthetic"
    for cl_field in (
        "cl_tasks_gained",
        "cl_required_gain",
        "synthetic_sanity_check",
    ):
        assert cl_field not in payload


def test_uniform_failure_band_falls_through_to_synthetic_gate(
    temp_manifest: Path, tmp_path: Path,
):
    """uniform_failure band (CL all-zero, e.g. validator broken) is NOT
    covered by use_cl_primary — only weak_signal triggers CL-primary.
    Verifies the gate falls through to the synthetic path with no
    KeyError and no CL eval. If someone later expands use_cl_primary
    to include uniform_failure, this test catches the change so it
    must be accompanied by a deliberate spec update."""
    fake_cache = MagicMock()
    sat_report = SaturationReport(
        band="uniform_failure",
        holdout_score=0.99,
        holdout_n=10,
        holdout_per_example=[1.0] * 10,
        closed_loop_score=0.0,
        closed_loop_n=7,
        closed_loop_per_example=[0.0] * 7,
        suggestions=[],
        thresholds={},
    )
    run_dir = tmp_path / "run"

    with _patch_stack(sat_report=sat_report, fake_cache=fake_cache):
        _run_evolve(
            manifest_path=temp_manifest,
            output_dir=run_dir,
            extra_kwargs={"force_saturation_check": True},
        )

    fake_cache.force_run.assert_not_called()
    payload = json.loads((run_dir / "gate_decision.json").read_text())
    assert payload["decision_signal"] == "synthetic"
    assert "baseline_closed_loop_per_example" not in payload
    assert "cl_tasks_gained" not in payload


def test_no_saturation_check_falls_through_to_synthetic_with_reason_recorded(
    temp_manifest: Path, tmp_path: Path,
):
    """--no-saturation-check → no preflight, falls through to synthetic.
    decision_signal == synthetic AND reason_synthetic == preflight_skipped
    so downstream consumers can distinguish 'preflight saw nothing weak'
    from 'preflight didn't run'."""
    fake_cache = MagicMock()
    run_dir = tmp_path / "run"

    # sat_report is unused (skip_saturation_check=True bypasses preflight)
    # but _patch_stack requires one.
    with _patch_stack(sat_report=_healthy_report(), fake_cache=fake_cache):
        _run_evolve(
            manifest_path=temp_manifest,
            output_dir=run_dir,
            extra_kwargs={"skip_saturation_check": True},
        )

    payload = json.loads((run_dir / "gate_decision.json").read_text())
    assert payload["decision_signal"] == "synthetic"
    assert payload["reason_synthetic"] == "preflight_skipped"


def test_cl_primary_decision_persists_to_gate_decision_json(
    temp_manifest: Path, tmp_path: Path,
):
    """weak_signal → all v5 CL fields present in gate_decision.json with
    correct types. Pins the JSON contract downstream consumers depend on."""
    fake_cache = MagicMock()
    fake_cache.force_run.return_value = _fake_validation_report(
        baseline_pass=[True, True, True, True, True, False, False],
        evolved_pass=[True, True, True, True, True, True, True],
    )
    run_dir = tmp_path / "run"

    # _LOW_GROWTH_EVOLVED → required_gain=1 → +2 win clears the gate so
    # the deploy path populates every v5 CL field we're pinning here.
    with _patch_stack(
        sat_report=_weak_signal_report(),
        fake_cache=fake_cache,
        evolved_description=_LOW_GROWTH_EVOLVED,
    ):
        _run_evolve(manifest_path=temp_manifest, output_dir=run_dir)

    payload = json.loads((run_dir / "gate_decision.json").read_text())

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
    temp_manifest: Path, tmp_path: Path,
):
    """healthy → synthetic path. All v4 fields present, schema_version=5,
    decision_signal=synthetic, no CL fields."""
    fake_cache = MagicMock()
    run_dir = tmp_path / "run"

    with _patch_stack(sat_report=_healthy_report(), fake_cache=fake_cache):
        _run_evolve(manifest_path=temp_manifest, output_dir=run_dir)

    payload = json.loads((run_dir / "gate_decision.json").read_text())

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
    ):
        assert required in payload, f"missing v4 field {required!r}"

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
    temp_manifest: Path, tmp_path: Path,
):
    """weak_signal + force_run raises → aborted decision,
    reason=cl_eval_failed, exception text recorded, evolved_FAILED.json
    written for forensic inspection of the rejected candidate."""
    fake_cache = MagicMock()
    fake_cache.force_run.side_effect = RuntimeError("validator crashed")
    run_dir = tmp_path / "run"

    with _patch_stack(sat_report=_weak_signal_report(), fake_cache=fake_cache):
        result = _run_evolve(manifest_path=temp_manifest, output_dir=run_dir)

    assert result == {"decision": "aborted", "reason": "cl_eval_failed"}

    payload = json.loads((run_dir / "gate_decision.json").read_text())
    assert payload["decision"] == "aborted"
    assert payload["reason"] == "cl_eval_failed"
    assert "validator crashed" in payload["cl_eval_exception"]

    assert (run_dir / "evolved_FAILED.json").exists(), (
        "evolved_FAILED.json must be written so the rejected variant "
        "is inspectable"
    )


def test_evolved_task_error_writes_cl_eval_incomplete_decision(
    temp_manifest: Path, tmp_path: Path,
):
    """weak_signal + one evolved task abstained → cl_eval_incomplete
    (NOT a regression). An infrastructure flake on the evolved phase
    isn't evidence of quality loss; conflating them would silently
    reject good candidates."""
    fake_cache = MagicMock()
    # task_2 abstains; others pass. Without the incomplete-detection
    # branch this would score as 6/7 (+1 vs 5/7 baseline) and deploy.
    fake_cache.force_run.return_value = _fake_validation_report(
        baseline_pass=[True, True, True, True, True, False, False],
        evolved_pass=[True, True, False, True, True, True, True],
        evolved_abstain=[False, False, True, False, False, False, False],
    )
    run_dir = tmp_path / "run"

    with _patch_stack(sat_report=_weak_signal_report(), fake_cache=fake_cache):
        result = _run_evolve(manifest_path=temp_manifest, output_dir=run_dir)

    assert result == {"decision": "aborted", "reason": "cl_eval_incomplete"}

    payload = json.loads((run_dir / "gate_decision.json").read_text())
    assert payload["decision"] == "aborted"
    assert payload["reason"] == "cl_eval_incomplete"
    assert payload["evolved_closed_loop_errored_tasks"] == ["task_2"]
    assert (run_dir / "evolved_FAILED.json").exists()


def test_absolute_char_ceiling_still_enforced_in_cl_primary_path(
    temp_manifest: Path, tmp_path: Path,
):
    """weak_signal + +2 CL win + evolved description exceeding the
    absolute char ceiling → reject. CL-primary mustn't bypass the
    wallpaper-protection backstop."""
    fake_cache = MagicMock()
    fake_cache.force_run.return_value = _fake_validation_report(
        baseline_pass=[True, True, True, True, True, False, False],
        evolved_pass=[True, True, True, True, True, True, True],
    )
    # Baseline = 12 chars. ceiling = max(50, 1.5*12) = 50.
    # Evolved ~480 chars; trips the absolute_char_ceiling backstop.
    # Stays under max_tool_desc_size=500 so static checks still pass.
    long_evolved = (
        "Find files in the repository by name pattern or glob; "
        "returns matching file paths from anywhere under the project root. "
    ) * 4
    assert 50 < len(long_evolved) <= 500, (
        f"Test pre-condition: expected 50 < len(long_evolved)={len(long_evolved)} <= 500"
    )

    run_dir = tmp_path / "run"

    with _patch_stack(
        sat_report=_weak_signal_report(),
        fake_cache=fake_cache,
        evolved_description=long_evolved,
    ):
        result = _run_evolve(
            manifest_path=temp_manifest,
            output_dir=run_dir,
            extra_kwargs={"max_absolute_chars": 50},
        )

    payload = json.loads((run_dir / "gate_decision.json").read_text())
    assert payload["decision"] == "reject", (
        f"absolute_char_ceiling must reject even on a winning CL gate; "
        f"got decision={payload['decision']} (reason={payload.get('reason')})"
    )
    assert "absolute_char_ceiling" in payload.get("failed_constraints", []), (
        f"failed_constraints={payload.get('failed_constraints')}"
    )
    # The deploy-gate reject path returns the reject reason from the dict.
    assert result["decision"] == "reject"


class TestSchemaV5Regression:
    """V5 must be additive over v4. Old consumers should see all v4 fields
    plus the new decision_signal field (and the CL-specific fields when
    use_cl_primary fired). Future schema bumps should add a parallel
    TestSchemaV{N}Regression class following the same pattern."""

    # V4 fields that MUST persist in v5 output regardless of code path.
    # Verified against the decision_payload literal in
    # evolution/tools/evolve_tool.py.
    V4_REQUIRED_FIELDS = frozenset({
        "schema_version", "decision", "reason", "decision_rule_used",
        "gate_mode", "inferiority_tolerance", "growth_pct",
        "required_improvement", "baseline_chars", "evolved_chars",
        "absolute_char_ceiling", "effective_absolute_char_ceiling",
        "growth_free_threshold", "fitness_profile", "proposer_mode",
        "growth_quality_slope", "baseline_per_example",
        "evolved_per_example",
    })

    def test_synthetic_path_writes_all_v4_fields(
        self, temp_manifest: Path, tmp_path: Path,
    ):
        """healthy band → synthetic gate. Every v4 field must still be
        present alongside the new decision_signal marker."""
        fake_cache = MagicMock()
        run_dir = tmp_path / "run"

        with _patch_stack(sat_report=_healthy_report(), fake_cache=fake_cache):
            _run_evolve(manifest_path=temp_manifest, output_dir=run_dir)

        payload = json.loads((run_dir / "gate_decision.json").read_text())

        missing = self.V4_REQUIRED_FIELDS - payload.keys()
        assert not missing, f"v4 fields missing in v5 synthetic payload: {sorted(missing)}"
        assert payload["schema_version"] == "5"
        assert payload["decision_signal"] == "synthetic"

    def test_cl_primary_path_writes_all_v4_fields_plus_cl_fields(
        self, temp_manifest: Path, tmp_path: Path,
    ):
        """weak_signal + +2 CL win → CL-primary gate. Every v4 field must
        still be present AND every new v5 CL-specific field must be
        populated."""
        cl_fields = frozenset({
            "decision_signal", "baseline_closed_loop_per_example",
            "evolved_closed_loop_per_example",
            "evolved_closed_loop_errored_tasks", "cl_tasks_gained",
            "cl_required_gain", "synthetic_sanity_check",
            "evolved_cl_eval_cost_usd", "band_trigger_score",
            "validator_agent_model",
        })
        fake_cache = MagicMock()
        # 5/7 baseline → 7/7 evolved with _LOW_GROWTH_EVOLVED keeps
        # required_gain=1 so the +2 win clears the gate and the deploy
        # branch writes every CL-specific field.
        fake_cache.force_run.return_value = _fake_validation_report(
            baseline_pass=[True, True, True, True, True, False, False],
            evolved_pass=[True, True, True, True, True, True, True],
        )
        run_dir = tmp_path / "run"

        with _patch_stack(
            sat_report=_weak_signal_report(),
            fake_cache=fake_cache,
            evolved_description=_LOW_GROWTH_EVOLVED,
        ):
            _run_evolve(manifest_path=temp_manifest, output_dir=run_dir)

        payload = json.loads((run_dir / "gate_decision.json").read_text())

        missing = (self.V4_REQUIRED_FIELDS | cl_fields) - payload.keys()
        assert not missing, f"v5 fields missing in CL-primary payload: {sorted(missing)}"
        assert payload["schema_version"] == "5"
        assert payload["decision_signal"] == "closed_loop"

    def test_cl_eval_failed_payload_has_schema_v5_and_decision_signal(
        self, temp_manifest: Path, tmp_path: Path,
    ):
        """Abort payloads are diagnostic-only (no full v4 field set), but
        must still pin schema_version="5" and a decision_signal so abort
        rows route the same way as deploy/reject rows in downstream jq."""
        fake_cache = MagicMock()
        fake_cache.force_run.side_effect = RuntimeError("validator crashed")
        run_dir = tmp_path / "run"

        with _patch_stack(sat_report=_weak_signal_report(), fake_cache=fake_cache):
            _run_evolve(manifest_path=temp_manifest, output_dir=run_dir)

        payload = json.loads((run_dir / "gate_decision.json").read_text())
        assert payload["schema_version"] == "5"
        assert payload["decision_signal"] == "closed_loop"
        assert payload["decision"] == "aborted"
        assert payload["reason"] == "cl_eval_failed"

    def test_cl_eval_incomplete_payload_has_schema_v5_and_decision_signal(
        self, temp_manifest: Path, tmp_path: Path,
    ):
        """Abort payloads from the incomplete-eval branch must also pin
        schema_version="5" and decision_signal so abort rows participate
        in v5 cohort queries alongside deploy/reject rows."""
        fake_cache = MagicMock()
        # task_2 abstains; mirrors the incomplete-detection scenario.
        fake_cache.force_run.return_value = _fake_validation_report(
            baseline_pass=[True, True, True, True, True, False, False],
            evolved_pass=[True, True, False, True, True, True, True],
            evolved_abstain=[False, False, True, False, False, False, False],
        )
        run_dir = tmp_path / "run"

        with _patch_stack(sat_report=_weak_signal_report(), fake_cache=fake_cache):
            _run_evolve(manifest_path=temp_manifest, output_dir=run_dir)

        payload = json.loads((run_dir / "gate_decision.json").read_text())
        assert payload["schema_version"] == "5"
        assert payload["decision_signal"] == "closed_loop"
        assert payload["decision"] == "aborted"
        assert payload["reason"] == "cl_eval_incomplete"

    def test_static_constraint_failure_payload_has_schema_v5_and_decision_signal(
        self, temp_manifest: Path, tmp_path: Path,
    ):
        """Static-fail fires before any CL evaluation could run, so the
        user never got into the CL-primary path → decision_signal must be
        "synthetic". Triggered by patching _candidate_description to
        return an empty string, which fails the non_empty constraint."""
        fake_cache = MagicMock()
        run_dir = tmp_path / "run"

        # Use the healthy band so we route through the synthetic-only
        # path conceptually, then make _candidate_description return ""
        # to trip the non_empty static constraint. The _patch_stack
        # context manager already patches _candidate_description; we
        # override it here with an empty string.
        with _patch_stack(
            sat_report=_healthy_report(),
            fake_cache=fake_cache,
            evolved_description="",
        ):
            _run_evolve(manifest_path=temp_manifest, output_dir=run_dir)

        payload = json.loads((run_dir / "gate_decision.json").read_text())
        assert payload["schema_version"] == "5"
        assert payload["decision_signal"] == "synthetic"
        assert payload["decision"] == "reject"
        assert payload["reason"] == "static_constraint_failure"


def test_summary_panel_reflects_cl_decision_when_cl_primary_deploys(
    temp_manifest: Path, tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """CL-primary deploy → summary line announces the CL gain instead of
    the synthetic delta. Without the CL-aware branch the panel says
    'did not improve' even though gate_decision.json deployed the artifact,
    so the operator gets a contradictory signal."""
    fake_cache = MagicMock()
    fake_cache.force_run.return_value = _fake_validation_report(
        baseline_pass=[True, True, True, True, True, False, False],
        evolved_pass=[True, True, True, True, True, True, True],
    )
    run_dir = tmp_path / "run"

    # _weak_signal_report pins baseline holdout to 0.95; evolved=0.90
    # forces a negative synthetic improvement so the pre-change panel
    # would render 'did not improve' even though CL-primary just deployed.
    with _patch_stack(
        sat_report=_weak_signal_report(),
        fake_cache=fake_cache,
        evolved_description=_LOW_GROWTH_EVOLVED,
        holdout_evolved_mean=0.90,
    ):
        _run_evolve(manifest_path=temp_manifest, output_dir=run_dir)

    out = capsys.readouterr().out
    assert "CL gained +2" in out, f"missing CL-gain line in summary: {out!r}"
    assert "did not improve" not in out, (
        f"synthetic 'did not improve' line leaked through CL-primary deploy: {out!r}"
    )


def test_summary_panel_reflects_cl_decision_when_cl_primary_rejects(
    temp_manifest: Path, tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """CL-primary reject → summary line explains the CL shortfall instead
    of falling back to the generic synthetic-rejected line."""
    fake_cache = MagicMock()
    # 5/7 → 5/7: zero CL gain, required_gain stays at 1 → reject.
    fake_cache.force_run.return_value = _fake_validation_report(
        baseline_pass=[True, True, True, True, True, False, False],
        evolved_pass=[True, True, True, True, True, False, False],
    )
    run_dir = tmp_path / "run"

    with _patch_stack(
        sat_report=_weak_signal_report(),
        fake_cache=fake_cache,
        evolved_description=_LOW_GROWTH_EVOLVED,
    ):
        _run_evolve(manifest_path=temp_manifest, output_dir=run_dir)

    out = capsys.readouterr().out
    assert "CL gain 0 < required 1" in out, (
        f"missing CL-reject line in summary: {out!r}"
    )


def test_summary_panel_uses_synthetic_delta_when_not_cl_primary(
    temp_manifest: Path, tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """healthy band → synthetic gate → existing 'improved/did not improve'
    wording is unchanged. Regression guard for the synthetic path.

    _healthy_report() pins baseline holdout to 0.5 (cached per-example);
    evolved=0.5 produces a zero synthetic delta that still clears the
    non-inferiority gate (within tolerance) so the deploy-path 'did not
    improve' line fires — that's the legacy branch we must preserve.
    """
    fake_cache = MagicMock()
    run_dir = tmp_path / "run"

    with _patch_stack(
        sat_report=_healthy_report(),
        fake_cache=fake_cache,
        holdout_evolved_mean=0.5,
    ):
        _run_evolve(manifest_path=temp_manifest, output_dir=run_dir)

    out = capsys.readouterr().out
    assert "did not improve" in out, (
        f"synthetic path must keep 'did not improve' on a zero delta: {out!r}"
    )
    assert "CL gained" not in out
    assert "CL gain" not in out
