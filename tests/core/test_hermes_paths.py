"""Tests for Hermes install discovery.

Resolution used to run through ``Path.home()``. Evolution executes inside the
``hermes`` container, where ``HOME`` is ``/root`` but the data directory is
bind-mounted elsewhere, so every importer resolved to a path that did not
exist and mined nothing. These tests pin the behaviour that replaced it.
"""

from __future__ import annotations

import pytest

from evolution.core.hermes_paths import (
    HermesInstall,
    HermesInstallNotFound,
    find_hermes_install,
    try_find_hermes_install,
)


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for var in ("HERMES_DATA_DIR", "HERMES_HOME"):
        monkeypatch.delenv(var, raising=False)


def make_root(path, marker="state.db"):
    path.mkdir(parents=True, exist_ok=True)
    (path / marker).write_text("")
    return path


class TestDiscovery:
    def test_explicit_path_wins(self, tmp_path, monkeypatch):
        explicit = make_root(tmp_path / "explicit")
        make_root(tmp_path / "env")
        monkeypatch.setenv("HERMES_DATA_DIR", str(tmp_path / "env"))

        assert find_hermes_install(explicit).root == explicit

    def test_data_dir_env_beats_hermes_home(self, tmp_path, monkeypatch):
        primary = make_root(tmp_path / "primary")
        secondary = make_root(tmp_path / "secondary")
        monkeypatch.setenv("HERMES_DATA_DIR", str(primary))
        monkeypatch.setenv("HERMES_HOME", str(secondary))

        install = find_hermes_install()
        assert install.root == primary
        assert install.source == "$HERMES_DATA_DIR"

    def test_hermes_home_is_honored(self, tmp_path, monkeypatch):
        root = make_root(tmp_path / "hh")
        monkeypatch.setenv("HERMES_HOME", str(root))
        assert find_hermes_install().source == "$HERMES_HOME"

    def test_falls_back_to_home_dot_hermes(self, tmp_path, monkeypatch):
        from pathlib import Path

        fake_home = tmp_path / "home"
        make_root(fake_home / ".hermes")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

        assert find_hermes_install().source == "~/.hermes"

    def test_any_marker_qualifies_a_root(self, tmp_path):
        for marker in ("state.db", "config.yaml", "skills", "profiles"):
            root = tmp_path / marker
            root.mkdir()
            (root / marker).write_text("") if "." in marker else (root / marker).mkdir()
            assert find_hermes_install(root).root == root


class TestFailureModes:
    def test_explicit_missing_path_raises(self, tmp_path):
        with pytest.raises(HermesInstallNotFound, match="does not exist"):
            find_hermes_install(tmp_path / "nope")

    def test_explicit_non_hermes_directory_raises(self, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        with pytest.raises(HermesInstallNotFound, match="does not look like"):
            find_hermes_install(plain)

    def test_nothing_found_lists_what_was_tried(self, tmp_path, monkeypatch):
        from pathlib import Path

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "empty-home"))
        monkeypatch.setenv("HERMES_DATA_DIR", str(tmp_path / "ghost"))

        with pytest.raises(HermesInstallNotFound) as exc:
            find_hermes_install()

        message = str(exc.value)
        assert "$HERMES_DATA_DIR" in message
        assert "HERMES_DATA_DIR to the directory containing state.db" in message

    def test_try_variant_returns_none(self, tmp_path, monkeypatch):
        from pathlib import Path

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "empty"))
        assert try_find_hermes_install() is None


class TestProfiles:
    def test_profiles_are_listed_alphabetically(self, install):
        assert [p.name for p in install.profiles()] == ["ali", "musa"]

    def test_root_without_profiles_dir_yields_a_default_profile(self, tmp_path):
        root = make_root(tmp_path / "single")
        install = HermesInstall(root=root, source="test")
        profiles = install.profiles()
        assert len(profiles) == 1
        assert profiles[0].name == "default"
        assert profiles[0].root == root

    def test_profiles_with_state_filters_to_real_databases(self, install, hermes_root):
        (hermes_root / "profiles" / "empty").mkdir()
        names = [p.name for p in install.profiles_with_state()]
        assert "empty" not in names
        assert "ali" in names

    def test_named_profile_lookup(self, install):
        assert install.profile("ali").name == "ali"

    def test_unknown_profile_names_the_available_ones(self, install):
        with pytest.raises(HermesInstallNotFound, match="Available: ali, musa"):
            install.profile("nobody")

    def test_store_paths_hang_off_the_profile(self, install):
        prof = install.profile("ali")
        assert prof.state_db.name == "state.db"
        assert prof.verification_db.name == "verification_evidence.db"
        assert prof.has_state_db()

    def test_shared_store_paths(self, install):
        assert install.cron_executions_db.exists()
        assert install.cron_jobs_json.exists()
