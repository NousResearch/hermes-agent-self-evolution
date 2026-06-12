"""Saturation calibration: guards the honesty logic (binning, Wilson, sweep).

The script lives in scripts/analysis/ (not an importable package), so load it
by path.
"""
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "analysis" / "calibrate_saturation.py"
)
_spec = importlib.util.spec_from_file_location("calibrate_saturation", _SCRIPT)
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)


def _gate(avg_baseline, lower_bound=None, avg_evolved=None, decision="reject",
          fitness_profile="balanced", gate_mode="no_regression",
          decision_signal="synthetic", n_examples=50):
    return {
        "avg_baseline": avg_baseline,
        "avg_evolved": avg_evolved if avg_evolved is not None else avg_baseline,
        "decision": decision,
        "fitness_profile": fitness_profile,
        "gate_mode": gate_mode,
        "decision_signal": decision_signal,
        "bootstrap": {"lower_bound": lower_bound, "n_examples": n_examples},
    }


def test_assign_bin_edges():
    assert cs.assign_bin(0.80) == "<0.90"
    assert cs.assign_bin(0.90) == "0.90-0.95"
    assert cs.assign_bin(0.949) == "0.90-0.95"
    assert cs.assign_bin(0.95) == "0.95-0.99"
    assert cs.assign_bin(0.989) == "0.95-0.99"
    assert cs.assign_bin(0.99) == ">=0.99"
    assert cs.assign_bin(1.0) == ">=0.99"


def test_wilson_zero_successes_has_wide_upper_bound():
    # 0/17 — the data-starved gated region. Point estimate 0, but upper bound
    # must be substantial (≈18%), which is the honest read the report leads with.
    lo, hi = cs.wilson_interval(0, 17)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert 0.15 < hi < 0.25


def test_wilson_empty_is_fully_uncertain():
    assert cs.wilson_interval(0, 0) == (0.0, 1.0)


def test_real_improvement_primary_is_lower_bound_positive():
    assert cs.real_improvement(_gate(0.6, lower_bound=0.04)) is True
    assert cs.real_improvement(_gate(0.6, lower_bound=0.0)) is False
    assert cs.real_improvement(_gate(0.6, lower_bound=-0.02)) is False
    assert cs.real_improvement(_gate(0.6, lower_bound=None)) is False


def test_real_improvement_decision_definition_is_circular():
    g = _gate(0.6, lower_bound=-0.1, decision="deploy")  # deployed but regressed
    assert cs.real_improvement(g, definition="lower_bound") is False
    assert cs.real_improvement(g, definition="decision") is True


def test_real_improvement_gain_definition_respects_delta():
    g = _gate(0.60, avg_evolved=0.63)  # +0.03
    assert cs.real_improvement(g, definition="gain", delta=0.02) is True
    assert cs.real_improvement(g, definition="gain", delta=0.05) is False


def test_false_abort_sweep_degenerate_gated_region():
    # Mirrors the real archive: every positive-signal run sits below 0.95,
    # the gated region has runs but none improve.
    runs = (
        [_gate(0.6, lower_bound=0.05) for _ in range(40)]   # positive, below gate
        + [_gate(0.97, lower_bound=0.0) for _ in range(15)]  # in gate, no signal
        + [_gate(0.995, lower_bound=None) for _ in range(5)]
    )
    sweep = {r["threshold"]: r for r in cs.false_abort_sweep(runs)}
    at95 = sweep[0.95]
    assert at95["would_abort_n"] == 20            # 15 + 5 at/above 0.95
    assert at95["n_real_improvement"] == 0
    assert at95["false_abort_rate"] == 0.0
    assert at95["wilson_upper"] > 0.0             # absence of evidence, not safety
    # The 40 positive runs (0.6) are below every candidate threshold, and even
    # the saturated runs (0.97/0.995) sit below τ=1.0 → nothing aborts at 1.0.
    assert sweep[1.0]["would_abort_n"] == 0


def test_false_abort_sweep_detects_a_real_false_abort():
    # A high-baseline run that genuinely improved — the case the threshold would
    # wrongly kill. The sweep must surface a nonzero false-abort rate.
    runs = [
        _gate(0.97, lower_bound=0.06),   # gated AND improved → false abort
        _gate(0.97, lower_bound=0.0),    # gated, no signal
        _gate(0.6, lower_bound=0.05),    # below gate
    ]
    at95 = {r["threshold"]: r for r in cs.false_abort_sweep(runs)}[0.95]
    assert at95["would_abort_n"] == 2
    assert at95["n_real_improvement"] == 1
    assert at95["false_abort_rate"] == pytest.approx(0.5)


def test_analyze_excludes_closed_loop_and_splits_strata():
    runs = (
        [_gate(0.6, lower_bound=0.05) for _ in range(3)]
        + [_gate(0.6, lower_bound=0.05, decision_signal="closed_loop")]
        + [_gate(0.6, lower_bound=0.05, fitness_profile="compression")]  # off-profile
    )
    a = cs.analyze(runs)
    assert a["n_total"] == 5
    assert a["n_closed_loop_excluded"] == 1
    assert a["n_paired_pool"] == 4          # closed-loop dropped from pool
    assert a["n_homogeneous"] == 3          # compression run is off-profile
    assert a["n_off_profile"] == 1


def test_analyze_handles_empty_pool():
    # Only closed-loop runs: synthetic pool is empty, must not raise.
    a = cs.analyze([_gate(0.6, lower_bound=0.05, decision_signal="closed_loop")])
    assert a["n_paired_pool"] == 0
    assert a["n_homogeneous"] == 0
    assert all(r["would_abort_n"] == 0 for r in a["false_abort_primary"])


def test_bin_stats_populated_bin_with_no_op_and_real_deploy():
    # Two runs in the 0.95-0.99 bin: one true no-op deploy (zero gain, lb=0),
    # one real improvement (+0.03, lb=0.04). Validates the no_op_deploy_frac
    # predicate that encodes the report's central honesty claim.
    runs = [
        _gate(0.96, avg_evolved=0.96, lower_bound=0.0, decision="deploy"),  # no-op
        _gate(0.97, avg_evolved=1.00, lower_bound=0.04, decision="deploy"),  # real
    ]
    a = cs.analyze(runs)
    bins = {b["bin"]: b for b in a["bins_homogeneous"]}
    b = bins["0.95-0.99"]
    assert b["n"] == 2
    assert b["deploy_rate"] == pytest.approx(1.0)
    assert b["no_op_deploy_frac"] == pytest.approx(0.5)
    assert b["frac_lower_bound_pos"] == pytest.approx(0.5)
    assert b["mean_realized_gain"] == pytest.approx((0.0 + 0.03) / 2)


# --- render_markdown: the script's primary deliverable; was 0% covered ---

def _analysis_and_friends(runs):
    return (
        cs.analyze(runs),
        {"n_lineage_runs": 0, "plateau_flagged": []},
        {"exists": False, "n_rows": 0, "n_aborted": 0},
    )


def test_render_markdown_data_starved_leads_with_absence_of_evidence():
    runs = [_gate(0.97, lower_bound=0.0) for _ in range(5)]  # gated, no signal
    md = cs.render_markdown(*_analysis_and_friends(runs))
    assert "cannot yet settle" in md
    assert "absence of evidence" in md
    assert "Wilson" in md


def test_render_markdown_carries_the_calibration_recipe():
    # The recommendation must always spell out the deliberate-campaign recipe,
    # not imply the ledger fills passively — guards against the prose dropping out.
    md = cs.render_markdown(*_analysis_and_friends([_gate(0.6, lower_bound=0.05)]))
    assert "--force-saturation-check" in md
    assert "does not fill passively" in md


def test_render_markdown_flips_when_gated_region_shows_signal():
    # 40 gated runs, several with real improvement → not data-starved; the
    # report must stop calling itself a survivorship counterfactual.
    runs = (
        [_gate(0.97, avg_evolved=1.0, lower_bound=0.05, decision="deploy") for _ in range(20)]
        + [_gate(0.97, lower_bound=0.0) for _ in range(20)]
    )
    md = cs.render_markdown(*_analysis_and_friends(runs))
    assert "becoming a real measurement" in md
    assert "Gated-region coverage" in md
    assert "fatal to the headline" not in md


def test_render_markdown_survives_threshold_sweep_without_0_95(monkeypatch):
    # If THRESHOLD_SWEEP is edited away from 0.95/0.99, render must not KeyError.
    monkeypatch.setattr(cs, "THRESHOLD_SWEEP", (0.90, 0.98))
    runs = [_gate(0.6, lower_bound=0.05) for _ in range(3)]
    md = cs.render_markdown(*_analysis_and_friends(runs))
    assert "Headline" in md  # rendered without raising


# --- scan_saturation_ledger: round-trip with the producer + missing file ---

def test_scan_saturation_ledger_missing(tmp_path):
    assert cs.scan_saturation_ledger(tmp_path) == {
        "exists": False, "n_rows": 0, "n_aborted": 0,
    }


def test_scan_saturation_ledger_round_trip_with_producer(tmp_path):
    # Lock the wire format between producer (telemetry module) and consumer.
    from evolution.core.saturation_check import SaturationReport
    from evolution.core.saturation_telemetry import (
        append_saturation_telemetry,
        build_saturation_telemetry_row,
    )

    def _row(run_id, holdout, proceeded, reason=None):
        rep = SaturationReport(
            band="no_headroom" if not proceeded else "healthy",
            holdout_score=holdout, holdout_n=10, holdout_per_example=[holdout] * 10,
        )
        return build_saturation_telemetry_row(
            rep, run_id=run_id, artifact="s", artifact_type="skill",
            proceeded=proceeded, abort_reason=reason,
        )

    append_saturation_telemetry(tmp_path, row=_row("a", 0.94, True))
    append_saturation_telemetry(tmp_path, row=_row("b", 0.96, False, "user_decline"))
    out = cs.scan_saturation_ledger(tmp_path)
    assert out["exists"] is True
    assert out["n_rows"] == 2
    assert out["n_aborted"] == 1
    assert out["n_in_gated_region"] == 1  # only holdout 0.96 >= 0.95


def test_scan_saturation_ledger_tolerates_torn_line(tmp_path):
    ledger = tmp_path / "saturation_ledger.jsonl"
    ledger.write_text(
        '{"proceeded": true, "holdout_score": 0.5, "band": "healthy"}\n'
        '{"proceeded": false, "holdout_sco\n'  # torn final line
    )
    out = cs.scan_saturation_ledger(tmp_path)
    assert out["n_rows"] == 1  # bad line skipped, not a crash


# --- load_gate_decisions: since filter, malformed skip + count, run_id stamp ---

def _write_gate(root, run_dir, payload):
    d = root / run_dir
    d.mkdir(parents=True)
    (d / "gate_decision.json").write_text(__import__("json").dumps(payload))


def test_load_gate_decisions_since_filter_and_skip_count(tmp_path):
    _write_gate(tmp_path, "skillA/20260101_000000", _gate(0.6, lower_bound=0.05))
    _write_gate(tmp_path, "skillA/20260601_000000", _gate(0.7, lower_bound=0.05))
    bad = tmp_path / "skillB" / "20260601_111111"
    bad.mkdir(parents=True)
    (bad / "gate_decision.json").write_text("{not valid json")

    runs, n_skipped = cs.load_gate_decisions(tmp_path, since="20260301_000000")
    assert n_skipped == 1                       # the corrupt file counted
    run_ids = {r["_run_id"] for r in runs}
    assert run_ids == {"20260601_000000"}       # older run dropped by since
    assert all("_run_id" in r for r in runs)


# --- scan_overfitting: plateau predicate (forward-only today) ---

def _lineage(tmp_path, run_id, candidates):
    d = tmp_path / "skill" / run_id
    d.mkdir(parents=True)
    (d / "lineage.json").write_text(__import__("json").dumps({"candidates": candidates}))


def test_scan_overfitting_flags_plateau_before_budget_exhausted(tmp_path):
    # val peaks at eval 10, search keeps spending to eval 30 → flagged.
    _lineage(tmp_path, "20260612_000000", [
        {"discovery_eval_count": 0, "val_aggregate": 0.5},
        {"discovery_eval_count": 10, "val_aggregate": 0.9},
        {"discovery_eval_count": 30, "val_aggregate": 0.9},
    ])
    out = cs.scan_overfitting(tmp_path)
    assert out["n_lineage_runs"] == 1
    assert len(out["plateau_flagged"]) == 1
    assert out["plateau_flagged"][0]["peak_at_eval"] == 10


def test_scan_overfitting_does_not_flag_peak_at_last_eval(tmp_path):
    _lineage(tmp_path, "20260612_000001", [
        {"discovery_eval_count": 0, "val_aggregate": 0.5},
        {"discovery_eval_count": 10, "val_aggregate": 0.7},
        {"discovery_eval_count": 30, "val_aggregate": 0.95},
    ])
    out = cs.scan_overfitting(tmp_path)
    assert out["plateau_flagged"] == []


def test_scan_overfitting_skips_short_lineage(tmp_path):
    _lineage(tmp_path, "20260612_000002", [
        {"discovery_eval_count": 0, "val_aggregate": 0.5},
        {"discovery_eval_count": 10, "val_aggregate": 0.9},
    ])
    out = cs.scan_overfitting(tmp_path)
    assert out["n_lineage_runs"] == 1
    assert out["plateau_flagged"] == []


# --- main: writes both report files; empty archive returns cleanly ---

def test_main_writes_reports(tmp_path):
    _write_gate(tmp_path / "output", "skill/20260601_000000", _gate(0.6, lower_bound=0.05))
    reports = tmp_path / "reports"
    rc = cs.main([
        "--output-root", str(tmp_path / "output"),
        "--reports-dir", str(reports),
    ])
    assert rc == 0
    assert (reports / "saturation_calibration_findings.md").exists()
    assert (reports / "saturation_calibration.json").exists()


def test_main_empty_archive_returns_zero_without_writing(tmp_path, capsys):
    reports = tmp_path / "reports"
    rc = cs.main(["--output-root", str(tmp_path / "empty"), "--reports-dir", str(reports)])
    assert rc == 0
    assert "No gate_decision.json found" in capsys.readouterr().out
    assert not reports.exists()
