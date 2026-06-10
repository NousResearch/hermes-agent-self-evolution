"""ClaudeAppendPromptInstaller — candidate append-prompt for closed-loop validation."""
import pytest

from evolution.validation.artifact_installer import ClaudeAppendPromptInstaller


def test_baseline_written_at_construction(tmp_path):
    inst = ClaudeAppendPromptInstaller(workdir=tmp_path, baseline_text="seed conventions")
    assert inst.target_path.read_text() == "seed conventions"
    assert inst.target_path.parent == tmp_path


def test_install_overwrites_with_candidate(tmp_path):
    inst = ClaudeAppendPromptInstaller(workdir=tmp_path, baseline_text="seed")
    src = tmp_path / "cand.txt"
    src.write_text("evolved conventions")
    sha = inst.install(src)
    assert inst.target_path.read_text() == "evolved conventions"
    assert len(sha) == 64


def test_install_text_helper(tmp_path):
    inst = ClaudeAppendPromptInstaller(workdir=tmp_path, baseline_text="seed")
    inst.install_text("direct text")
    assert inst.target_path.read_text() == "direct text"


def test_verify_backup_rejects_empty(tmp_path):
    inst = ClaudeAppendPromptInstaller(workdir=tmp_path, baseline_text="x")
    bad = tmp_path / "empty"
    bad.write_bytes(b"")
    with pytest.raises(ValueError):
        inst.verify_backup(bad)


def test_verify_backup_accepts_valid_utf8(tmp_path):
    inst = ClaudeAppendPromptInstaller(workdir=tmp_path, baseline_text="x")
    ok = tmp_path / "ok"
    ok.write_text("some conventions")
    inst.verify_backup(ok)  # no raise


def test_missing_workdir_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        ClaudeAppendPromptInstaller(workdir=tmp_path / "nope", baseline_text="x")
