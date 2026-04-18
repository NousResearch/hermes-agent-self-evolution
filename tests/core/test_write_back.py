"""Tests for the auto-mode write-back path."""
from __future__ import annotations

from pathlib import Path

import pytest

from evolution.core.write_back import WriteBackResult, write_back_skill


BASELINE = """---
name: foo
description: bar
---

Original body.
"""

EVOLVED = """---
name: foo
description: bar
---

Evolved body — much clearer.
"""


def _prep_live(tmp_path: Path, name: str = "SKILL.md", content: str = BASELINE) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


# ───────────────────────── guard rails ─────────────────────────
def test_no_writeback_when_mode_is_propose(tmp_path: Path):
    live = _prep_live(tmp_path)
    result = write_back_skill(live, EVOLVED, mode="propose", auto_merge=True)
    assert result.merged is False
    assert result.backup_path is None
    assert "not 'auto'" in result.reason
    assert live.read_text() == BASELINE  # untouched


def test_no_writeback_when_gate_rejects(tmp_path: Path):
    live = _prep_live(tmp_path)
    result = write_back_skill(live, EVOLVED, mode="auto", auto_merge=False)
    assert result.merged is False
    assert result.backup_path is None
    assert "gate rejected" in result.reason
    assert live.read_text() == BASELINE


def test_raises_when_live_path_missing(tmp_path: Path):
    missing = tmp_path / "does_not_exist.md"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        write_back_skill(missing, EVOLVED, mode="auto", auto_merge=True)


# ───────────────────────── happy path ─────────────────────────
def test_auto_merge_writes_evolved_and_creates_backup(tmp_path: Path):
    live = _prep_live(tmp_path)
    result = write_back_skill(
        live, EVOLVED, mode="auto", auto_merge=True, timestamp="20260418_060000"
    )
    assert result.merged is True
    assert live.read_text() == EVOLVED
    assert result.backup_path is not None
    assert result.backup_path.exists()
    assert result.backup_path.read_text() == BASELINE
    assert result.backup_path.parent.name == ".backups"
    assert "20260418_060000" in result.backup_path.name


def test_backup_filename_includes_timestamp_and_original_name(tmp_path: Path):
    live = _prep_live(tmp_path, name="custom-skill.md")
    result = write_back_skill(
        live, EVOLVED, mode="auto", auto_merge=True, timestamp="20260418_120000"
    )
    assert result.backup_path.name == "custom-skill.md.20260418_120000.bak"


def test_custom_backup_dir(tmp_path: Path):
    live = _prep_live(tmp_path)
    backup_dir = tmp_path / "my-backups"
    result = write_back_skill(
        live,
        EVOLVED,
        mode="auto",
        auto_merge=True,
        timestamp="20260418_060001",
        backup_dir=backup_dir,
    )
    assert result.backup_path.parent == backup_dir
    assert backup_dir.exists()


def test_backup_dir_created_if_missing(tmp_path: Path):
    live = _prep_live(tmp_path)
    backup_dir = tmp_path / "nested" / "a" / "b" / "backups"
    assert not backup_dir.exists()
    result = write_back_skill(
        live, EVOLVED, mode="auto", auto_merge=True,
        timestamp="20260418_060002", backup_dir=backup_dir,
    )
    assert backup_dir.is_dir()
    assert result.backup_path.exists()


def test_no_tmp_file_left_behind(tmp_path: Path):
    live = _prep_live(tmp_path)
    write_back_skill(live, EVOLVED, mode="auto", auto_merge=True, timestamp="20260418_060003")
    # No .tmp file should remain after atomic rename
    tmp_file = live.with_suffix(live.suffix + ".tmp")
    assert not tmp_file.exists()


def test_multiple_writebacks_create_multiple_backups(tmp_path: Path):
    live = _prep_live(tmp_path)
    r1 = write_back_skill(live, EVOLVED, mode="auto", auto_merge=True, timestamp="20260418_060010")
    # Second evolution with different content
    EVOLVED_V2 = EVOLVED.replace("much clearer.", "even better now.")
    r2 = write_back_skill(live, EVOLVED_V2, mode="auto", auto_merge=True, timestamp="20260418_060011")
    assert r1.backup_path.exists() and r2.backup_path.exists()
    assert r1.backup_path != r2.backup_path
    # r1's backup = original BASELINE; r2's backup = EVOLVED (what was there before r2)
    assert r1.backup_path.read_text() == BASELINE
    assert r2.backup_path.read_text() == EVOLVED
    assert live.read_text() == EVOLVED_V2


def test_auto_generates_timestamp_if_not_provided(tmp_path: Path):
    live = _prep_live(tmp_path)
    result = write_back_skill(live, EVOLVED, mode="auto", auto_merge=True)
    # YYYYMMDD_HHMMSS embedded in backup filename
    assert result.backup_path is not None
    # shape: "SKILL.md.YYYYMMDD_HHMMSS.bak"
    parts = result.backup_path.name.split(".")
    assert parts[-1] == "bak"
    ts = parts[-2]
    assert len(ts) == 15
    assert "_" in ts


def test_result_to_dict_is_json_safe(tmp_path: Path):
    live = _prep_live(tmp_path)
    result = write_back_skill(
        live, EVOLVED, mode="auto", auto_merge=True, timestamp="20260418_060020"
    )
    d = result.to_dict()
    import json
    # Must serialize cleanly (paths converted to strings)
    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed["merged"] is True
    assert parsed["live_path"] == str(live)
    assert parsed["backup_path"] == str(result.backup_path)


def test_result_to_dict_handles_no_backup(tmp_path: Path):
    live = _prep_live(tmp_path)
    result = write_back_skill(live, EVOLVED, mode="propose", auto_merge=True)
    d = result.to_dict()
    assert d["backup_path"] is None
    assert d["merged"] is False
