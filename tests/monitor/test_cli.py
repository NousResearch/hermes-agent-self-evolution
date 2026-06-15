"""CLI guards for the monitor sentinel."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from evolution.monitor.__main__ import main


def test_attempt_top_requires_cost_cap(tmp_path: Path):
    # --attempt-top is the only step that spends; the CLI must refuse to run it
    # uncapped. The guard fires before any scan, so an empty dir suffices for --repo.
    res = CliRunner().invoke(main, ["--repo", str(tmp_path), "--attempt-top", "3"])
    assert res.exit_code != 0
    assert "max-cost-usd" in res.output
