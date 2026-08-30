"""Tests for the Phase 4 guardrails.

Pure source analysis, so these are fast and completely offline. The fixture
below imitates the shape of a real hermes-agent tool module: guard helpers that
return an error string instead of raising, try/except around every I/O call,
and module-level registry.register(...) statements at the bottom.
"""

from evolution.code.safety import (
    census_error_handling,
    census_guards,
    check_error_handling_not_reduced,
    check_parses,
    check_registry_calls_unchanged,
    check_safety_checks_not_removed,
    check_signatures_unchanged,
    collect_registry_calls,
    collect_signatures,
    quality_signals,
    run_safety_checks,
)

TOOL = '''"""Toy file tools shaped like the real ones."""

from tools import registry


def _is_blocked_device(path):
    """Refuse device files that would hang the process."""
    return path.startswith("/dev/")


def _check_sensitive_path(path):
    """Return an error string for a path we must not read."""
    if "secrets" in path:
        return "Error: refusing to read a sensitive path"
    return None


def read_file_tool(path, offset=1, limit=500, task_id="default"):
    """Read a file with pagination."""
    if _is_blocked_device(path):
        return tool_error("Cannot read a device file")
    blocked = _check_sensitive_path(path)
    if blocked:
        return blocked
    try:
        with open(path) as handle:
            return handle.read()
    except OSError as exc:
        return tool_error(f"Could not read {path}: {exc}")


def write_file_tool(path, content, task_id="default"):
    """Write a file."""
    try:
        with open(path, "w") as handle:
            handle.write(content)
    except OSError as exc:
        return tool_error(f"Could not write {path}: {exc}")
    return "ok"


def tool_error(message):
    """Wrap an error message the way the agent expects."""
    return f"Error: {message}"


registry.register(name="read_file", toolset="file", schema=READ_FILE_SCHEMA,
                  handler=read_file_tool, check_fn=_check_file_reqs, emoji="📖")
registry.register(name="write_file", toolset="file", schema=WRITE_FILE_SCHEMA,
                  handler=write_file_tool, check_fn=_check_file_reqs, emoji="✍️")
'''


class TestParseGate:
    def test_valid_source_parses(self):
        assert check_parses(TOOL, TOOL).passed

    def test_broken_candidate_is_reported_with_a_line(self):
        result = check_parses(TOOL, "def broken(:\n    pass\n")
        assert not result.passed
        assert "does not parse" in result.message
        assert result.violations

    def test_unparseable_candidate_short_circuits_the_ladder(self):
        report = run_safety_checks(TOOL, "def broken(:\n")
        assert not report.passed
        assert [r.name for r in report.results] == ["parses"]


class TestSignatures:
    def test_identical_source_passes(self):
        result = check_signatures_unchanged(TOOL, TOOL)
        assert result.passed
        assert "intact" in result.message

    def test_collect_finds_every_function(self):
        names = set(collect_signatures(TOOL))
        assert {"read_file_tool", "write_file_tool", "tool_error"} <= names

    def test_removed_function_is_a_violation(self):
        after = TOOL.replace(
            'def tool_error(message):\n    """Wrap an error message the way the agent expects."""\n    return f"Error: {message}"\n',
            "",
        )
        result = check_signatures_unchanged(TOOL, after)
        assert not result.passed
        assert any("removed function: def tool_error" in v for v in result.violations)

    def test_renamed_parameter_is_a_violation(self):
        after = TOOL.replace(
            "def read_file_tool(path, offset=1, limit=500, task_id=\"default\"):",
            "def read_file_tool(filepath, offset=1, limit=500, task_id=\"default\"):",
        )
        result = check_signatures_unchanged(TOOL, after)
        assert not result.passed
        assert any("signature changed" in v for v in result.violations)

    def test_changed_default_is_a_violation(self):
        after = TOOL.replace("limit=500", "limit=1000")
        result = check_signatures_unchanged(TOOL, after)
        assert not result.passed
        assert any("limit=1000" in v for v in result.violations)

    def test_added_parameter_is_a_violation(self):
        after = TOOL.replace(
            "def write_file_tool(path, content, task_id=\"default\"):",
            "def write_file_tool(path, content, task_id=\"default\", encoding=\"utf-8\"):",
        )
        assert not check_signatures_unchanged(TOOL, after).passed

    def test_added_helper_function_is_allowed(self):
        after = TOOL + '\n\ndef _new_helper(value):\n    """Added by evolution."""\n    return value\n'
        result = check_signatures_unchanged(TOOL, after)
        assert result.passed
        assert any("added function" in note for note in result.info)

    def test_becoming_async_is_a_violation(self):
        after = TOOL.replace("def tool_error(message):", "async def tool_error(message):")
        assert not check_signatures_unchanged(TOOL, after).passed

    def test_methods_are_tracked_by_qualified_name(self):
        before = "class A:\n    def run(self, x):\n        return x\n"
        after = "class A:\n    def run(self, x, y=1):\n        return x\n"
        assert "A.run" in collect_signatures(before)
        assert not check_signatures_unchanged(before, after).passed

    def test_annotation_only_change_is_tolerated(self):
        before = "def f(x):\n    return x\n"
        after = "def f(x: int) -> int:\n    return x\n"
        assert check_signatures_unchanged(before, after).passed


class TestRegistryCalls:
    def test_identical_registrations_pass(self):
        result = check_registry_calls_unchanged(TOOL, TOOL)
        assert result.passed
        assert "2 registration call(s)" in result.message

    def test_collect_finds_each_registration(self):
        assert len(collect_registry_calls(TOOL)) == 2

    def test_removed_registration_is_a_violation(self):
        after = TOOL.replace(
            'registry.register(name="write_file", toolset="file", schema=WRITE_FILE_SCHEMA,\n'
            '                  handler=write_file_tool, check_fn=_check_file_reqs, emoji="✍️")\n',
            "",
        )
        result = check_registry_calls_unchanged(TOOL, after)
        assert not result.passed
        assert any("removed registration" in v for v in result.violations)

    def test_dropped_keyword_is_a_violation(self):
        after = TOOL.replace(', check_fn=_check_file_reqs, emoji="📖"', ', emoji="📖"')
        result = check_registry_calls_unchanged(TOOL, after)
        assert not result.passed
        assert len(result.violations) == 2  # one removed, one added

    def test_changed_toolset_is_a_violation(self):
        after = TOOL.replace('toolset="file", schema=READ_FILE_SCHEMA', 'toolset="fs", schema=READ_FILE_SCHEMA')
        assert not check_registry_calls_unchanged(TOOL, after).passed

    def test_added_registration_is_a_violation(self):
        after = TOOL + '\nregistry.register(name="extra", toolset="file", schema=X, handler=y)\n'
        assert not check_registry_calls_unchanged(TOOL, after).passed

    def test_reformatting_a_registration_is_not_a_change(self):
        after = TOOL.replace(
            'registry.register(name="read_file", toolset="file", schema=READ_FILE_SCHEMA,\n'
            '                  handler=read_file_tool, check_fn=_check_file_reqs, emoji="📖")',
            'registry.register(\n'
            '    name="read_file",\n'
            '    toolset="file",\n'
            '    schema=READ_FILE_SCHEMA,\n'
            '    handler=read_file_tool,\n'
            '    check_fn=_check_file_reqs,\n'
            '    emoji="📖",\n'
            ')',
        )
        assert check_registry_calls_unchanged(TOOL, after).passed


class TestErrorHandling:
    def test_census_counts_the_real_shapes(self):
        census = census_error_handling(TOOL)
        assert census.try_blocks == 2
        assert census.handlers == 2
        assert census.bare_handlers == 0
        assert census.raises == 0

    def test_identical_source_passes(self):
        assert check_error_handling_not_reduced(TOOL, TOOL).passed

    def test_removing_a_try_block_is_a_violation(self):
        after = TOOL.replace(
            '    try:\n'
            '        with open(path, "w") as handle:\n'
            '            handle.write(content)\n'
            '    except OSError as exc:\n'
            '        return tool_error(f"Could not write {path}: {exc}")\n',
            '    with open(path, "w") as handle:\n'
            '        handle.write(content)\n',
        )
        result = check_error_handling_not_reduced(TOOL, after)
        assert not result.passed
        assert any("try_blocks dropped 2 -> 1" in v for v in result.violations)
        assert any("write_file_tool" in v for v in result.violations)

    def test_adding_error_handling_is_allowed_and_noted(self):
        after = TOOL.replace(
            "def tool_error(message):\n"
            '    """Wrap an error message the way the agent expects."""\n'
            '    return f"Error: {message}"\n',
            "def tool_error(message):\n"
            '    """Wrap an error message the way the agent expects."""\n'
            "    try:\n"
            '        return f"Error: {message}"\n'
            "    except TypeError:\n"
            '        return "Error"\n',
        )
        result = check_error_handling_not_reduced(TOOL, after)
        assert result.passed
        assert any("grew" in note for note in result.info)

    def test_a_removed_raise_is_a_violation(self):
        before = "def f(x):\n    if x < 0:\n        raise ValueError('negative')\n    return x\n"
        after = "def f(x):\n    if x < 0:\n        return None\n    return x\n"
        result = check_error_handling_not_reduced(before, after)
        assert not result.passed
        assert any("raises dropped" in v for v in result.violations)

    def test_a_removed_finally_is_a_violation(self):
        before = (
            "def f():\n    try:\n        g()\n    except OSError:\n        pass\n"
            "    finally:\n        cleanup()\n"
        )
        after = "def f():\n    try:\n        g()\n    except OSError:\n        pass\n"
        result = check_error_handling_not_reduced(before, after)
        assert not result.passed
        assert any("finallys dropped" in v for v in result.violations)

    def test_moving_handling_between_functions_is_still_a_violation(self):
        before = (
            "def a():\n    try:\n        x()\n    except OSError:\n        pass\n\n"
            "def b():\n    return 1\n"
        )
        after = (
            "def a():\n    return x()\n\n"
            "def b():\n    try:\n        return 1\n    except OSError:\n        pass\n"
        )
        result = check_error_handling_not_reduced(before, after)
        assert not result.passed
        assert any(v.startswith("a: error handling dropped") for v in result.violations)

    def test_error_handling_lost_with_a_removed_function_is_not_double_reported(self):
        before = "def a():\n    try:\n        x()\n    except OSError:\n        pass\n"
        after = ""
        result = check_error_handling_not_reduced(before, after)
        assert not result.passed
        assert not any(v.startswith("a: ") for v in result.violations)


class TestSafetyChecks:
    def test_census_counts_guards_the_way_hermes_writes_them(self):
        census = census_guards(TOOL)
        assert census.asserts == 0
        assert census.guard_calls  # _is_blocked_device, _check_sensitive_path
        assert census.guard_returns >= 2

    def test_identical_source_passes(self):
        assert check_safety_checks_not_removed(TOOL, TOOL).passed

    def test_removing_a_guard_call_is_a_violation(self):
        after = TOOL.replace(
            '    if _is_blocked_device(path):\n        return tool_error("Cannot read a device file")\n',
            "",
        )
        result = check_safety_checks_not_removed(TOOL, after)
        assert not result.passed
        assert any("_is_blocked_device" in v for v in result.violations)

    def test_removing_a_guard_return_is_a_violation(self):
        after = TOOL.replace(
            "    blocked = _check_sensitive_path(path)\n    if blocked:\n        return blocked\n",
            "    blocked = _check_sensitive_path(path)\n",
        )
        result = check_safety_checks_not_removed(TOOL, after)
        assert not result.passed

    def test_removing_an_assert_is_a_violation(self):
        before = "def f(x):\n    assert x is not None\n    return x\n"
        after = "def f(x):\n    return x\n"
        result = check_safety_checks_not_removed(before, after)
        assert not result.passed
        assert any("assert statements dropped 1 -> 0" in v for v in result.violations)

    def test_adding_a_guard_is_allowed_and_noted(self):
        after = TOOL.replace(
            "def write_file_tool(path, content, task_id=\"default\"):\n"
            '    """Write a file."""\n',
            "def write_file_tool(path, content, task_id=\"default\"):\n"
            '    """Write a file."""\n'
            "    if _is_blocked_device(path):\n"
            '        return tool_error("Cannot write a device file")\n',
        )
        result = check_safety_checks_not_removed(TOOL, after)
        assert result.passed

    def test_extra_guard_names_are_honoured(self):
        before = "def f(x):\n    y = house_rules(x)\n    return y\n"
        after = "def f(x):\n    return x\n"
        assert check_safety_checks_not_removed(before, after).passed
        strict = check_safety_checks_not_removed(
            before, after, extra_guard_names=["house_rules"]
        )
        assert not strict.passed


class TestReport:
    def test_clean_candidate_passes_the_whole_ladder(self):
        after = TOOL.replace('return handle.read()', 'return handle.read() or ""')
        report = run_safety_checks(TOOL, after)
        assert report.passed
        assert report.failures == []
        assert len(report.results) == 5

    def test_report_collects_every_violation(self):
        after = TOOL.replace("limit=500", "limit=1000").replace(
            ', check_fn=_check_file_reqs, emoji="📖"', ', emoji="📖"'
        )
        report = run_safety_checks(TOOL, after)
        assert not report.passed
        assert len(report.failures) == 2
        assert len(report.violations) >= 3
        assert report.first_failure().name == "signatures_frozen"

    def test_summary_marks_each_check(self):
        summary = run_safety_checks(TOOL, TOOL).summary()
        assert "✓ signatures_frozen" in summary
        assert "✓ registry_frozen" in summary

    def test_to_dict_is_json_serialisable(self):
        import json

        blob = json.loads(json.dumps(run_safety_checks(TOOL, TOOL).to_dict()))
        assert blob["passed"] is True
        assert len(blob["results"]) == 5

    def test_custom_check_list_is_respected(self):
        report = run_safety_checks(TOOL, TOOL, checks=[check_signatures_unchanged])
        assert [r.name for r in report.results] == ["parses", "signatures_frozen"]


class TestQualitySignals:
    def test_identical_source_scores_perfectly(self):
        signals = quality_signals(TOOL, TOOL)
        assert signals.score == 1.0
        assert signals.notes == ["no quality regressions detected"]

    def test_new_bare_except_costs_points(self):
        after = TOOL.replace("    except OSError as exc:\n        return tool_error(f\"Could not write {path}: {exc}\")",
                             "    except:\n        return tool_error(\"failed\")")
        signals = quality_signals(TOOL, after)
        assert signals.score < 1.0
        assert any("bare" in note for note in signals.notes)

    def test_silently_swallowed_exception_costs_points(self):
        before = "def f():\n    try:\n        g()\n    except OSError as exc:\n        log(exc)\n"
        after = "def f():\n    try:\n        g()\n    except OSError:\n        pass\n"
        signals = quality_signals(before, after)
        assert signals.score < 1.0
        assert any("swallow" in note for note in signals.notes)

    def test_doubling_the_file_costs_points(self):
        signals = quality_signals(TOOL, TOOL + TOOL + TOOL)
        assert signals.score < 1.0
        assert any("doubled" in note for note in signals.notes)

    def test_new_todo_markers_cost_points(self):
        after = TOOL.replace('    """Write a file."""', '    """Write a file."""\n    # TODO: handle unicode')
        signals = quality_signals(TOOL, after)
        assert signals.score < 1.0
        assert any("TODO" in note for note in signals.notes)

    def test_removing_a_docstring_costs_points(self):
        after = TOOL.replace('    """Write a file."""\n', "")
        signals = quality_signals(TOOL, after)
        assert signals.score < 1.0
        assert any("docstring" in note for note in signals.notes)
        assert "write_file_tool" in signals.details["dropped_docstrings"]

    def test_added_handling_earns_a_small_bonus(self):
        after = TOOL.replace(
            "def tool_error(message):\n"
            '    """Wrap an error message the way the agent expects."""\n'
            '    return f"Error: {message}"\n',
            "def tool_error(message):\n"
            '    """Wrap an error message the way the agent expects."""\n'
            "    try:\n"
            '        return f"Error: {message}"\n'
            "    except TypeError:\n"
            '        return "Error"\n',
        )
        signals = quality_signals(TOOL, after)
        assert signals.score == 1.0
        assert any("added exception handler" in note for note in signals.notes)

    def test_unparseable_candidate_scores_zero(self):
        signals = quality_signals(TOOL, "def broken(:\n")
        assert signals.score == 0.0

    def test_score_stays_within_bounds(self):
        after = "def broken_but_parsing():\n    try:\n        pass\n    except:\n        pass\n" * 5
        signals = quality_signals(TOOL, after)
        assert 0.0 <= signals.score <= 1.0

    def test_to_dict_is_json_serialisable(self):
        import json

        blob = json.loads(json.dumps(quality_signals(TOOL, TOOL).to_dict()))
        assert blob["score"] == 1.0
