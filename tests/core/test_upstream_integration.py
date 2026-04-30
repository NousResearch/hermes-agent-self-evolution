"""Tests for upstream PR integrations.

Covers:
  pr/26 — CodexLM provider, LLM keyword expansion in RelevanceFilter
  pr/25 — SQLite progress tracker (start_run/log_event/complete_run/fail_run)
  pr/19 — api_base/api_key wiring for local models
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from evolution.core.codex_lm import CodexLM
from evolution.core.config import EvolutionConfig
from evolution.core.external_importers import RelevanceFilter
from evolution.monitor.progress import (
    start_run,
    log_event,
    complete_run,
    fail_run,
    get_active_run,
    get_run_history,
    get_run_events,
    _db_path,
)


# ── pr/26: CodexLM ────────────────────────────────────────────────────────


class TestCodexLM:
    def test_init_defaults(self):
        lm = CodexLM()
        assert lm.timeout == 300
        assert lm.model == "codex/gpt-5.4"
        assert lm.model_type == "chat"
        assert lm.history == []
        assert lm.provider == "codex"

    def test_inspect_history_empty(self):
        lm = CodexLM()
        assert lm.inspect_history(1) == []

    def test_call_with_neither_prompt_nor_messages(self):
        lm = CodexLM()
        result = lm(prompt=None, messages=None)
        assert result == [""]

    def test_call_subprocess_failure_returns_error(self):
        lm = CodexLM()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="codex: command not found", stdout="")
            result = lm(prompt="hello")
            assert result[0].startswith("Error:")

    def test_call_subprocess_timeout(self):
        import subprocess
        lm = CodexLM(timeout=1)
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="codex", timeout=1)
            result = lm(prompt="hello")
            assert "timed out" in result[0]

    def test_call_parses_agent_message(self):
        lm = CodexLM()
        ndjson = (
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "hello world"}})
            + "\n"
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=ndjson, stderr="")
            result = lm(prompt="say hi")
            assert result == ["hello world"]
            assert lm.history  # was recorded

    def test_call_with_messages_concatenates(self):
        lm = CodexLM()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            lm(messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
            # Check that the args contain both messages joined
            call_args = mock_run.call_args[0][0]
            text_arg = call_args[3]  # ["codex", "exec", "--json", text]
            assert "hi" in text_arg
            assert "hello" in text_arg


# ── pr/26: LLM keyword expansion ──────────────────────────────────────────


def _stub_expander(keywords_payload: str):
    """Create a callable that mimics dspy.ChainOfThought returning .keywords."""
    def stub(**kwargs):
        return MagicMock(keywords=keywords_payload)
    return stub


def _failing_expander(exc: Exception):
    def stub(**kwargs):
        raise exc
    return stub


class TestKeywordExpansion:
    def test_expand_keywords_seeds_with_skill_name_tokens(self):
        rf = RelevanceFilter(model="openai/gpt-4.1-mini")
        rf.expander = _stub_expander("")
        with patch("dspy.LM"):
            keywords = rf._expand_keywords("github-code-review", "skill body")
        # Skill name tokens must always be present as fallback
        assert "github" in keywords
        assert "code" in keywords
        assert "review" in keywords

    def test_expand_keywords_parses_json_array(self):
        rf = RelevanceFilter(model="openai/gpt-4.1-mini")
        rf.expander = _stub_expander('["pull request", "git diff", "code audit"]')
        with patch("dspy.LM"):
            keywords = rf._expand_keywords("github-code-review", "skill body")
        assert "pull request" in keywords
        assert "git diff" in keywords
        assert "code audit" in keywords

    def test_expand_keywords_parses_with_surrounding_text(self):
        rf = RelevanceFilter(model="openai/gpt-4.1-mini")
        rf.expander = _stub_expander(
            'Here are the keywords: ["alpha", "beta", "gamma"] hope it helps'
        )
        with patch("dspy.LM"):
            keywords = rf._expand_keywords("test", "body")
        assert "alpha" in keywords
        assert "beta" in keywords
        assert "gamma" in keywords

    def test_expand_keywords_drops_short_strings(self):
        rf = RelevanceFilter(model="openai/gpt-4.1-mini")
        rf.expander = _stub_expander('["a", "ab", "abc", "abcd"]')
        with patch("dspy.LM"):
            keywords = rf._expand_keywords("test", "body")
        # Length > 2 filter
        assert "a" not in keywords
        assert "ab" not in keywords
        assert "abc" in keywords
        assert "abcd" in keywords

    def test_expand_keywords_dedupes(self):
        rf = RelevanceFilter(model="openai/gpt-4.1-mini")
        rf.expander = _stub_expander('["alpha", "ALPHA", "alpha", "beta"]')
        with patch("dspy.LM"):
            keywords = rf._expand_keywords("test", "body")
        # Lowercased + de-duped
        assert keywords.count("alpha") == 1
        assert "beta" in keywords

    def test_expand_keywords_handles_llm_exception(self):
        rf = RelevanceFilter(model="openai/gpt-4.1-mini")
        rf.expander = _failing_expander(RuntimeError("model unavailable"))
        with patch("dspy.LM"):
            keywords = rf._expand_keywords("code-review", "body")
        # Falls back to skill name tokens
        assert "code" in keywords
        assert "review" in keywords


# ── pr/25: SQLite progress tracker ────────────────────────────────────────


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Redirect HERMES_HOME so each test gets its own SQLite file."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path / "evolution_progress.db"


def test_db_path_uses_hermes_home(isolated_db, monkeypatch):
    assert _db_path() == isolated_db


def test_start_run_creates_row(isolated_db):
    cfg = EvolutionConfig(iterations=5, optimizer_model="openai/gpt-4.1")
    meta = start_run("alpha-skill", cfg)
    assert "run_id" in meta
    assert meta["skill_name"] == "alpha-skill"
    assert meta["status"] == "running"
    assert meta["iterations"] == 5

    # Verify it was written to the DB
    conn = sqlite3.connect(str(isolated_db))
    rows = conn.execute("SELECT skill_name, status FROM runs").fetchall()
    conn.close()
    assert rows == [("alpha-skill", "running")]


def test_log_event_appends_to_run(isolated_db):
    cfg = EvolutionConfig()
    meta = start_run("beta-skill", cfg)
    log_event(meta["run_id"], "step1", "did the thing")
    log_event(meta["run_id"], "step2", "did another thing")

    events = get_run_events(meta["run_id"])
    assert len(events) == 2
    assert events[0]["step"] == "step1"
    assert events[1]["step"] == "step2"


def test_complete_run_persists_metrics(isolated_db):
    cfg = EvolutionConfig()
    meta = start_run("gamma-skill", cfg)
    result = complete_run(meta["run_id"], {
        "baseline_score": 0.5,
        "evolved_score": 0.7,
        "improvement": 0.2,
        "constraints_passed": 1,
    })
    assert result["status"] == "completed"
    assert result["evolved_score"] == 0.7


def test_complete_run_ignores_unknown_keys(isolated_db):
    """Unknown keys must be silently dropped to avoid SQL injection vectors."""
    cfg = EvolutionConfig()
    meta = start_run("delta", cfg)
    # Try to inject an unknown column name
    complete_run(meta["run_id"], {
        "baseline_score": 0.1,
        "DROP TABLE runs": "not allowed",
        "; DELETE FROM runs;": "evil",
    })
    # Table must still exist and contain the row
    history = get_run_history()
    delta_runs = [r for r in history if r.get("skill_name") == "delta"]
    assert len(delta_runs) == 1


def test_fail_run_marks_failed(isolated_db):
    cfg = EvolutionConfig()
    meta = start_run("eps-skill", cfg)
    fail_run(meta["run_id"], "constraint violation")

    conn = sqlite3.connect(str(isolated_db))
    row = conn.execute(
        "SELECT status FROM runs WHERE id = ?", (meta["run_id"],)
    ).fetchone()
    conn.close()
    assert row[0] == "failed"


def test_get_active_run_finds_running(isolated_db):
    cfg = EvolutionConfig()
    start_run("active-skill", cfg)
    active = get_active_run()
    assert active is not None
    assert active["skill_name"] == "active-skill"
    assert active["status"] == "running"


def test_get_active_run_skips_completed(isolated_db):
    cfg = EvolutionConfig()
    meta = start_run("done-skill", cfg)
    complete_run(meta["run_id"], {})
    assert get_active_run() is None


# ── pr/19: api_base/api_key wiring ─────────────────────────────────────────


class TestApiBaseApiKey:
    def test_config_has_api_base_and_api_key_fields(self):
        cfg = EvolutionConfig(api_base="http://localhost:8000/v1", api_key="sk-test")
        assert cfg.api_base == "http://localhost:8000/v1"
        assert cfg.api_key == "sk-test"

    def test_api_key_does_not_leak_in_repr(self):
        cfg = EvolutionConfig(api_base="http://localhost:8000/v1", api_key="leak-me-please")
        assert "leak-me-please" not in repr(cfg)

    def test_make_lm_forwards_api_base_and_key(self):
        cfg = EvolutionConfig(api_base="http://localhost:8000/v1", api_key="sk-local")
        with patch("dspy.LM") as mock_lm:
            cfg.make_lm("openai/gpt-4.1")
            call_kwargs = mock_lm.call_args.kwargs
            assert call_kwargs.get("api_base") == "http://localhost:8000/v1"
            assert call_kwargs.get("api_key") == "sk-local"

    def test_make_lm_no_kwargs_without_api_base(self):
        cfg = EvolutionConfig()
        with patch("dspy.LM") as mock_lm:
            cfg.make_lm("openai/gpt-4.1")
            # When no api_base/api_key, kwargs are empty
            assert "api_base" not in mock_lm.call_args.kwargs
            assert "api_key" not in mock_lm.call_args.kwargs

    def test_make_lm_minimax_path_unchanged_by_api_base(self, monkeypatch):
        """MiniMax routing must use minimax_base_url, not the custom api_base."""
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-key")
        cfg = EvolutionConfig(api_base="http://wrong-endpoint/v1", api_key="wrong-key")
        with patch("dspy.LM") as mock_lm:
            cfg.make_lm("minimax/MiniMax-M2.7")
            call_kwargs = mock_lm.call_args.kwargs
            # Must use MiniMax's URL+key, NOT the api_base override
            assert "minimax" in call_kwargs.get("base_url", "")
            assert call_kwargs.get("api_key") == "mm-key"
