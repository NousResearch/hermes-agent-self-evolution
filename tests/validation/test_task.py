"""Tests for evolution.validation.task — Task + TaskSuite + JSONL loader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evolution.validation.task import Task, TaskSuite


class TestTaskRendering:
    def test_render_substitutes_fixture_dir(self, tmp_path):
        task = Task(
            task_id="t1",
            user_message="open {fixture_dir}/foo.py and add a line",
        )
        rendered = task.render_message(tmp_path)
        assert str(tmp_path) in rendered
        assert "{fixture_dir}" not in rendered

    def test_render_leaves_message_unchanged_when_no_placeholder(self, tmp_path):
        task = Task(task_id="t1", user_message="run the agent")
        assert task.render_message(tmp_path) == "run the agent"

    def test_render_preserves_literal_braces_in_message(self, tmp_path):
        # Task content often contains code with `{` and `}` (Python dict
        # literals, JSON snippets, format specifiers). These must survive
        # the placeholder substitution without being interpreted.
        task = Task(
            task_id="t1",
            user_message="write {fixture_dir}/x.py: data = {'id': 1, 'value': [2, 3]}",
        )
        rendered = task.render_message(tmp_path)
        assert "{'id': 1, 'value': [2, 3]}" in rendered
        assert str(tmp_path) in rendered


class TestTaskSuiteLoader:
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def test_loads_jsonl(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        self._write_jsonl(p, [
            {"task_id": "a", "user_message": "do a",
             "expected_tools": ["patch"], "forbidden_tools": ["write_file"]},
            {"task_id": "b", "user_message": "do b",
             "expected_tools": ["write_file"], "forbidden_tools": ["patch"]},
        ])
        suite = TaskSuite.from_jsonl(p)
        assert len(suite.tasks) == 2
        assert suite.tasks[0].task_id == "a"
        assert suite.tasks[0].expected_tools == ("patch",)
        assert suite.tasks[0].forbidden_tools == ("write_file",)
        assert suite.tasks[1].task_id == "b"

    def test_sha256_matches_file_contents(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        self._write_jsonl(p, [{"task_id": "t", "user_message": "m"}])
        suite = TaskSuite.from_jsonl(p)
        expected = hashlib.sha256(p.read_bytes()).hexdigest()
        assert suite.sha256 == expected

    def test_sha256_changes_when_file_changes(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        self._write_jsonl(p, [{"task_id": "t", "user_message": "m"}])
        sha_a = TaskSuite.from_jsonl(p).sha256
        self._write_jsonl(p, [{"task_id": "t", "user_message": "different"}])
        sha_b = TaskSuite.from_jsonl(p).sha256
        assert sha_a != sha_b

    def test_skips_blank_lines_and_comments(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        p.write_text(
            "# a comment\n"
            "\n"
            '{"task_id": "t1", "user_message": "m1"}\n'
            "  # another comment\n"
            '{"task_id": "t2", "user_message": "m2"}\n'
        )
        suite = TaskSuite.from_jsonl(p)
        assert [t.task_id for t in suite.tasks] == ["t1", "t2"]

    def test_fixture_setup_parsed_as_dict(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        self._write_jsonl(p, [{
            "task_id": "t",
            "user_message": "edit {fixture_dir}/foo.py",
            "expected_tools": ["patch"],
            "fixture_setup": {"foo.py": "def hello(): pass\n"},
        }])
        suite = TaskSuite.from_jsonl(p)
        assert suite.tasks[0].fixture_setup == {"foo.py": "def hello(): pass\n"}

    def test_missing_task_id_raises(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        self._write_jsonl(p, [{"user_message": "m"}])
        with pytest.raises(ValueError, match="task_id"):
            TaskSuite.from_jsonl(p)

    def test_missing_user_message_raises(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        self._write_jsonl(p, [{"task_id": "t"}])
        with pytest.raises(ValueError, match="user_message"):
            TaskSuite.from_jsonl(p)

    def test_malformed_json_names_lineno(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        p.write_text(
            '{"task_id": "t1", "user_message": "m1"}\n'
            "this is not json\n"
        )
        with pytest.raises(ValueError, match=":2:"):
            TaskSuite.from_jsonl(p)

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        p.write_text("# only comments\n\n")
        with pytest.raises(ValueError, match="no tasks parsed"):
            TaskSuite.from_jsonl(p)

    def test_non_dict_fixture_setup_raises(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        self._write_jsonl(p, [{
            "task_id": "t", "user_message": "m",
            "fixture_setup": ["not", "a", "dict"],
        }])
        with pytest.raises(ValueError, match="fixture_setup must be a dict"):
            TaskSuite.from_jsonl(p)

    def test_test_command_parsed_when_present(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        self._write_jsonl(p, [{
            "task_id": "t1",
            "user_message": "debug it",
            "test_command": "python test_solution.py",
        }])
        suite = TaskSuite.from_jsonl(p)
        assert suite.tasks[0].test_command == "python test_solution.py"

    def test_test_command_defaults_to_none(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        self._write_jsonl(p, [{"task_id": "t1", "user_message": "do X"}])
        suite = TaskSuite.from_jsonl(p)
        assert suite.tasks[0].test_command is None

    def test_test_command_non_string_raises(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        self._write_jsonl(p, [{
            "task_id": "t", "user_message": "m",
            "test_command": ["python", "x.py"],
        }])
        with pytest.raises(ValueError, match="test_command must be a string"):
            TaskSuite.from_jsonl(p)


class TestActionLevelFields:
    """skills_src, expected_action, target_skill, stale_token — new optional fields."""

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def test_all_four_fields_round_trip(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        self._write_jsonl(p, [{
            "task_id": "t1",
            "user_message": "update skill X",
            "skills_src": "/path/to/skills",
            "expected_action": "patch",
            "target_skill": "SKILLS_GUIDANCE",
            "stale_token": "old text",
        }])
        suite = TaskSuite.from_jsonl(p)
        t = suite.tasks[0]
        assert t.skills_src == "/path/to/skills"
        assert t.expected_action == "patch"
        assert t.target_skill == "SKILLS_GUIDANCE"
        assert t.stale_token == "old text"

    def test_all_four_default_to_none_when_absent(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        self._write_jsonl(p, [{"task_id": "t1", "user_message": "hi"}])
        suite = TaskSuite.from_jsonl(p)
        t = suite.tasks[0]
        assert t.skills_src is None
        assert t.expected_action is None
        assert t.target_skill is None
        assert t.stale_token is None

    def test_null_values_are_none(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        self._write_jsonl(p, [{
            "task_id": "t1",
            "user_message": "hi",
            "skills_src": None,
            "expected_action": None,
            "target_skill": None,
            "stale_token": None,
        }])
        suite = TaskSuite.from_jsonl(p)
        t = suite.tasks[0]
        assert t.skills_src is None
        assert t.expected_action is None
        assert t.target_skill is None
        assert t.stale_token is None

    def test_non_string_skills_src_raises(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        self._write_jsonl(p, [{"task_id": "t", "user_message": "m", "skills_src": 42}])
        with pytest.raises(ValueError, match="skills_src must be a string"):
            TaskSuite.from_jsonl(p)

    def test_non_string_expected_action_raises(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        self._write_jsonl(p, [{"task_id": "t", "user_message": "m", "expected_action": 42}])
        with pytest.raises(ValueError, match="expected_action must be a string"):
            TaskSuite.from_jsonl(p)

    def test_non_string_target_skill_raises(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        self._write_jsonl(p, [{"task_id": "t", "user_message": "m", "target_skill": 42}])
        with pytest.raises(ValueError, match="target_skill must be a string"):
            TaskSuite.from_jsonl(p)

    def test_non_string_stale_token_raises(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        self._write_jsonl(p, [{"task_id": "t", "user_message": "m", "stale_token": 42}])
        with pytest.raises(ValueError, match="stale_token must be a string"):
            TaskSuite.from_jsonl(p)

    def test_existing_task_construction_unchanged(self):
        # Backward-compat: old Task() calls without the new fields still work.
        t = Task(
            task_id="old",
            user_message="do something",
            expected_tools=("patch",),
            forbidden_tools=(),
        )
        assert t.skills_src is None
        assert t.expected_action is None
        assert t.target_skill is None
        assert t.stale_token is None


class TestExpectedSaveContent:
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def test_round_trips_from_jsonl(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        self._write_jsonl(p, [{
            "task_id": "save-pref-001",
            "user_message": "I prefer uv over pip for Python projects.",
            "expected_tools": ["memory"],
            "expected_save_content": "user prefers uv over pip",
        }])
        suite = TaskSuite.from_jsonl(p)
        assert suite.tasks[0].expected_save_content == "user prefers uv over pip"

    def test_defaults_to_none(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        self._write_jsonl(p, [{
            "task_id": "t1", "user_message": "hi", "expected_tools": ["memory"],
        }])
        suite = TaskSuite.from_jsonl(p)
        assert suite.tasks[0].expected_save_content is None

    def test_non_string_raises(self, tmp_path):
        p = tmp_path / "suite.jsonl"
        self._write_jsonl(p, [{
            "task_id": "t", "user_message": "m", "expected_save_content": 42,
        }])
        with pytest.raises(ValueError, match="expected_save_content must be a string"):
            TaskSuite.from_jsonl(p)
