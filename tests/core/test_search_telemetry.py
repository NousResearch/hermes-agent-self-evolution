"""Search-telemetry: val-discrimination signal from GEPA detailed_results."""
import json

import pytest

from pathlib import Path

from evolution.core.search_telemetry import (
    LEDGER_NAME,
    append_search_telemetry,
    build_search_telemetry_row,
    main,
    read_ledger,
    resolve_ledger_root,
    summarize_ledger,
)


def test_resolve_ledger_root_finds_output_ancestor():
    # skill / tool / prompt layouts all share one output/ root.
    assert resolve_ledger_root(Path("output/my-skill/20260101")).name == "output"
    assert resolve_ledger_root(Path("output/tools/write_file/20260101")).name == "output"
    assert resolve_ledger_root(Path("output/prompts/SKILLS/20260101")).name == "output"


def test_resolve_ledger_root_falls_back_to_run_dir(tmp_path):
    # No "output" ancestor (tests / custom --output-dir): stays in its own tree.
    run_dir = tmp_path / "run123"
    assert resolve_ledger_root(run_dir) == run_dir.resolve()


def test_row_fields_on_discriminating_run():
    row = build_search_telemetry_row(
        artifact="my-skill",
        artifact_type="skill",
        val_scores=[0.2, 0.5, 0.5, 0.9],
        best_idx=3,
        decision="pass",
    )
    assert row.artifact == "my-skill"
    assert row.artifact_type == "skill"
    assert row.n_candidates == 4
    assert row.n_distinct_val == 3  # 0.2, 0.5, 0.9
    assert row.distinct_val_frac == pytest.approx(0.75)
    assert row.best_val == pytest.approx(0.9)
    assert row.median_val == pytest.approx(0.5)
    assert row.val_spread == pytest.approx(0.7)
    assert row.best_idx == 3
    assert row.best_idx_frac == pytest.approx(1.0)  # 3 / (4-1)
    assert row.decision == "pass"


def test_row_captures_tie_saturation():
    # All four candidates share one val level: selection is a coin flip.
    row = build_search_telemetry_row(
        artifact="t", artifact_type="tool",
        val_scores=[0.5, 0.5, 0.5, 0.5], best_idx=2,
    )
    assert row.n_distinct_val == 1
    assert row.distinct_val_frac == pytest.approx(0.25)
    assert row.val_spread == pytest.approx(0.0)


def test_single_candidate_best_idx_frac_is_zero():
    row = build_search_telemetry_row(
        artifact="t", artifact_type="tool", val_scores=[0.7], best_idx=0,
    )
    assert row.n_candidates == 1
    assert row.best_idx_frac == 0.0
    assert row.distinct_val_frac == pytest.approx(1.0)


def test_build_returns_none_on_empty_candidate_pool():
    # MIPROv2 fallback path: no candidates / empty val list.
    assert build_search_telemetry_row(
        artifact="x", artifact_type="skill", val_scores=[], best_idx=0,
    ) is None
    assert build_search_telemetry_row(
        artifact="x", artifact_type="skill", val_scores=None, best_idx=0,
    ) is None


def test_append_writes_one_jsonl_row(tmp_path):
    path = append_search_telemetry(
        tmp_path, artifact="s", artifact_type="skill",
        val_scores=[0.1, 0.4], best_idx=1, decision="pass",
    )
    assert path == tmp_path / LEDGER_NAME
    rows = read_ledger(path)
    assert len(rows) == 1
    assert rows[0]["artifact"] == "s"
    assert rows[0]["n_candidates"] == 2
    assert rows[0]["decision"] == "pass"


def test_append_is_additive(tmp_path):
    append_search_telemetry(
        tmp_path, artifact="a", artifact_type="skill", val_scores=[0.5], best_idx=0
    )
    append_search_telemetry(
        tmp_path, artifact="b", artifact_type="tool", val_scores=[0.3, 0.6], best_idx=1
    )
    rows = read_ledger(tmp_path / LEDGER_NAME)
    assert [r["artifact"] for r in rows] == ["a", "b"]


def test_append_skips_when_no_candidates(tmp_path):
    result = append_search_telemetry(
        tmp_path, artifact="x", artifact_type="skill", val_scores=[], best_idx=0,
    )
    assert result is None
    assert not (tmp_path / LEDGER_NAME).exists()


def test_summarize_empty_ledger(tmp_path):
    out = summarize_ledger(tmp_path / LEDGER_NAME)
    assert "No search telemetry" in out


def test_backfill_reports_infeasible_and_counts_legacy_runs(tmp_path, capsys):
    # Two legacy gate_decision.json files with no val distribution.
    for name in ("run_a", "run_b"):
        d = tmp_path / name
        d.mkdir()
        (d / "gate_decision.json").write_text(json.dumps({"decision": "pass"}))
    rc = main(["--backfill", "--output-root", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "infeasible" in out.lower()
    assert "2 run(s) skipped" in out


def test_main_prints_summary(tmp_path, capsys):
    append_search_telemetry(
        tmp_path, artifact="s", artifact_type="skill", val_scores=[0.5, 0.5], best_idx=0
    )
    rc = main(["--ledger", str(tmp_path / LEDGER_NAME)])
    assert rc == 0
    assert "Search telemetry" in capsys.readouterr().out


def test_summarize_groups_by_artifact_type(tmp_path):
    append_search_telemetry(
        tmp_path, artifact="s1", artifact_type="skill", val_scores=[0.5, 0.5], best_idx=0
    )
    append_search_telemetry(
        tmp_path, artifact="s2", artifact_type="skill", val_scores=[0.2, 0.8], best_idx=1
    )
    append_search_telemetry(
        tmp_path, artifact="t1", artifact_type="tool", val_scores=[0.4], best_idx=0
    )
    out = summarize_ledger(tmp_path / LEDGER_NAME)
    assert "skill" in out and "tool" in out
    assert "2" in out  # 2 skill runs

