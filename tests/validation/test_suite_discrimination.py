"""Per-task discrimination labeler. Logic anchored to the real conventions
leave-one-out numbers (bin/run unfillable, bin/check fillable, fmt/lint easy)."""
import pytest

from evolution.validation.report import (
    PhaseResult,
    TaskResult,
    ValidationReport,
    WinLoss,
)
from evolution.validation.suite_discrimination import (
    BASELINE_FAILS,
    DISCRIMINATIVE,
    NOISE_LIMITED,
    TOO_EASY,
    UNFILLABLE,
    classify_task,
    discrimination_report,
    probe_discrimination,
    write_discrimination_sidecar,
)


class TestClassifyTask:
    def test_too_easy_when_baseline_saturates(self):
        assert classify_task(0.95) == TOO_EASY
        assert classify_task(0.90, ceiling_rate=1.0) == TOO_EASY  # bin/fmt-like (0.9)

    def test_baseline_only_low_is_baseline_fails(self):
        # No ceiling → can't tell unfillable from fillable.
        assert classify_task(0.0) == BASELINE_FAILS
        assert classify_task(0.10) == BASELINE_FAILS

    def test_baseline_only_mid_is_discriminative(self):
        assert classify_task(0.5) == DISCRIMINATIVE

    def test_unfillable_when_ceiling_also_low(self):
        # bin/run: baseline 0.0, ceiling 0.0 — nothing lifts it.
        assert classify_task(0.0, ceiling_rate=0.0) == UNFILLABLE

    def test_discriminative_when_ceiling_high(self):
        # bin/check: baseline 0.0 → ceiling 1.0 (fillable headroom).
        assert classify_task(0.0, ceiling_rate=1.0, flip=0.1) == DISCRIMINATIVE

    def test_noise_limited_when_gain_within_flip(self):
        # Real but tiny fillable gain, swamped by the A/A flip floor.
        assert classify_task(0.45, ceiling_rate=0.60, flip=0.4) == NOISE_LIMITED

    def test_gain_above_flip_is_discriminative(self):
        assert classify_task(0.20, ceiling_rate=0.90, flip=0.15) == DISCRIMINATIVE


class TestDiscriminationReport:
    def test_labels_and_summary_with_ceiling(self):
        # The conventions LOO regime: run unfillable, check discriminative,
        # fmt too_easy, lint discriminative.
        rpt = discrimination_report(
            {"run": 0.0, "check": 0.0, "fmt": 0.9, "lint": 0.6},
            ceiling_rates={"run": 0.0, "check": 1.0, "fmt": 1.0, "lint": 1.0},
            flips={"run": 0.0, "check": 0.1, "fmt": 0.1, "lint": 0.2},
        )
        assert rpt.labels == {
            "run": UNFILLABLE, "check": DISCRIMINATIVE,
            "fmt": TOO_EASY, "lint": DISCRIMINATIVE,
        }
        assert rpt.summary[DISCRIMINATIVE] == 2
        assert "2/4 tasks discriminate" in rpt.recommendation

    def test_zero_discriminative_recommends_authoring(self):
        rpt = discrimination_report(
            {"a": 1.0, "b": 0.0},
            ceiling_rates={"a": 1.0, "b": 0.0},  # a too_easy, b unfillable
        )
        assert rpt.summary.get(DISCRIMINATIVE, 0) == 0
        assert "cannot justify search spend" in rpt.recommendation

    def test_to_dict_roundtrips_via_sidecar(self, tmp_path):
        rpt = discrimination_report({"a": 0.5}, reps=8, suite_sha256="sha")
        suite_path = tmp_path / "s.jsonl"
        out = write_discrimination_sidecar(rpt, suite_path)
        assert out.name == "s.jsonl.discrimination.json"
        import json
        loaded = json.loads(out.read_text())
        assert loaded["labels"]["a"] == DISCRIMINATIVE
        assert loaded["reps"] == 8


def _task(tid, passed, abstained=False):
    return TaskResult(task_id=tid, passed=passed, abstained=abstained,
                      tool_calls_seq=[], duration_seconds=0.0)


def _phase(tasks):
    np = sum(1 for t in tasks if t.passed and not t.abstained)
    nf = sum(1 for t in tasks if not t.passed and not t.abstained)
    na = sum(1 for t in tasks if t.abstained)
    scored = np + nf
    return PhaseResult(pass_rate=(np / scored) if scored else 0.0,
                       n_passed=np, n_failed=nf, n_abstained=na, tasks=tasks)


def _report(baseline_tasks, evolved_tasks):
    b, e = _phase(baseline_tasks), _phase(evolved_tasks)
    return ValidationReport(
        schema_version="x", tool="t", task_suite_path="s", task_suite_sha256="sha",
        baseline=b, evolved=e,
        delta=WinLoss(n_wins=0, n_losses=0, n_ties=0, pass_rate_change=0.0),
        decision="pass", decision_reasons=[],
    )


class _StubValidator:
    reps = 2

    def __init__(self, by_artifact):
        # by_artifact: Path -> ValidationReport
        self._by = by_artifact

    def validate(self, inputs):
        return self._by[inputs.baseline_artifact]


def test_probe_pools_both_phases_and_uses_ceiling(tmp_path):
    from evolution.validation.task import TaskSuite
    suite = TaskSuite(path=tmp_path / "s.jsonl", sha256="sha", tasks=())
    base, ceil = tmp_path / "base.txt", tmp_path / "ceil.txt"
    # baseline: task "check" fails both phases (0/2 pooled → rate 0.0)
    base_rep = _report([_task("check", False)], [_task("check", False)])
    # ceiling: "check" passes both phases (2/2 → rate 1.0)
    ceil_rep = _report([_task("check", True)], [_task("check", True)])
    validator = _StubValidator({base: base_rep, ceil: ceil_rep})
    rpt = probe_discrimination(validator, suite, base, ceiling_artifact=ceil, reps=2)
    assert rpt.baseline_rates["check"] == pytest.approx(0.0)
    assert rpt.ceiling_rates["check"] == pytest.approx(1.0)
    assert rpt.labels["check"] == DISCRIMINATIVE
