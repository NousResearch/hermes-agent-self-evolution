"""Convention-adherence verdict: did the agent use the repo wrapper, not the default tool."""
from evolution.validation.agent_runner import AgentRunResult
from evolution.validation.report import score_task


def _run(cmds, error=None):
    return AgentRunResult(
        tool_calls_seq=["Bash"] * len(cmds),
        final_text_tail="",
        duration_seconds=1.0,
        error=error,
        tool_calls_with_args=[
            {"name": "Bash", "arguments": {"command": c}} for c in cmds
        ],
    )


def test_convention_adherent_when_wrapper_used_and_default_avoided():
    passed, abstained = score_task(
        expected_tools=(), forbidden_tools=(), run=_run(["./bin/check"]),
        expected_action="convention",
        required_cmd_substr=("bin/check",), forbidden_cmd_substr=("pytest",),
    )
    assert (passed, abstained) == (True, False)


def test_convention_fails_when_default_used():
    passed, _ = score_task(
        expected_tools=(), forbidden_tools=(), run=_run(["python -m pytest"]),
        expected_action="convention",
        required_cmd_substr=("bin/check",), forbidden_cmd_substr=("pytest",),
    )
    assert passed is False


def test_convention_fails_when_wrapper_used_but_default_also_used():
    passed, _ = score_task(
        expected_tools=(), forbidden_tools=(), run=_run(["./bin/check", "pytest -q"]),
        expected_action="convention",
        required_cmd_substr=("bin/check",), forbidden_cmd_substr=("pytest",),
    )
    assert passed is False


def test_convention_fails_when_no_bash_at_all():
    passed, abstained = score_task(
        expected_tools=(), forbidden_tools=(), run=_run([]),
        expected_action="convention",
        required_cmd_substr=("bin/check",), forbidden_cmd_substr=("pytest",),
    )
    assert (passed, abstained) == (False, False)


def test_convention_abstains_on_runner_error():
    passed, abstained = score_task(
        expected_tools=(), forbidden_tools=(), run=_run([], error="timeout"),
        expected_action="convention",
        required_cmd_substr=("bin/check",), forbidden_cmd_substr=("pytest",),
    )
    assert abstained is True


def test_convention_boundary_aware_no_false_positive_on_filename():
    # `cat pytest.ini` must NOT count as using pytest; `./bin/check` must count.
    passed, _ = score_task(
        expected_tools=(), forbidden_tools=(),
        run=_run(["./bin/check", "cat pytest.ini"]),
        expected_action="convention",
        required_cmd_substr=("bin/check",), forbidden_cmd_substr=("pytest",),
    )
    assert passed is True


def test_convention_boundary_matches_default_at_token_end():
    # `python -m pytest` (pytest at end) IS a forbidden use.
    passed, _ = score_task(
        expected_tools=(), forbidden_tools=(),
        run=_run(["./bin/check", "python -m pytest"]),
        expected_action="convention",
        required_cmd_substr=("bin/check",), forbidden_cmd_substr=("pytest",),
    )
    assert passed is False


def _run_named(tool, cmds):
    return AgentRunResult(
        tool_calls_seq=[tool] * len(cmds), final_text_tail="", duration_seconds=1.0,
        tool_calls_with_args=[{"name": tool, "arguments": {"command": c}} for c in cmds],
    )


def test_convention_default_tool_is_bash():
    # A Shell call is ignored by default (command_tool defaults to "Bash").
    passed, _ = score_task(
        expected_tools=(), forbidden_tools=(), run=_run_named("Shell", ["./bin/check"]),
        expected_action="convention",
        required_cmd_substr=("bin/check",), forbidden_cmd_substr=("pytest",),
    )
    assert passed is False


def test_convention_honors_command_tool_field():
    # With command_tool="Shell", the verdict scores Shell calls instead of Bash.
    passed, _ = score_task(
        expected_tools=(), forbidden_tools=(), run=_run_named("Shell", ["./bin/check"]),
        expected_action="convention", command_tool="Shell",
        required_cmd_substr=("bin/check",), forbidden_cmd_substr=("pytest",),
    )
    assert passed is True


def test_convention_task_requires_required_cmd_substr():
    import pytest
    from evolution.validation.task import TaskSuite
    bad = (
        '{"task_id":"x","expected_action":"convention","forbidden_cmd_substr":["pytest"],'
        '"user_message":"run tests","fixture_setup":{}}\n'
    )
    p = __import__("pathlib").Path(__import__("tempfile").mkstemp(suffix=".jsonl")[1])
    p.write_text(bad)
    with pytest.raises(ValueError, match="required_cmd_substr"):
        TaskSuite.from_jsonl(p)


# --- execution-aware matching: the wrapper must be INVOKED, not merely mentioned ---

def _conv(cmds):
    """convention verdict for required=bin/check, forbidden=pytest over Bash cmds."""
    passed, _ = score_task(
        expected_tools=(), forbidden_tools=(), run=_run(cmds),
        expected_action="convention",
        required_cmd_substr=("bin/check",), forbidden_cmd_substr=("pytest",),
    )
    return passed


import pytest as _pytest


@_pytest.mark.parametrize("cmd", [
    "cat bin/check",                 # reading the wrapper, not running it
    "echo 'bin/check passed'",       # mentioning it in echoed text
    "grep bin/check Makefile",       # searching for it
    "echo bin/check > run.log",      # writing its name to a file
    "./bin/check --help",            # invoked, but help mode — didn't run the suite
    "bash -c \"echo bin/check\"",    # mentioned inside a -c script body
])
def test_convention_required_mention_does_not_count_as_used(cmd):
    assert _conv([cmd]) is False


@_pytest.mark.parametrize("cmd", [
    "./bin/check",
    "bash bin/check",
    "sudo ./bin/check",
    "bin/check && echo done",
    "env CI=1 ./bin/check",
    "cat input.txt && ./bin/check",  # explore-then-run still counts
])
def test_convention_required_real_invocation_counts_as_used(cmd):
    assert _conv([cmd]) is True


@_pytest.mark.parametrize("cmd", [
    "pytest -q",
    "python -m pytest",
    "python3 -m pytest tests/",
])
def test_convention_forbidden_real_invocation_is_bypass(cmd):
    # used the wrapper AND ran the forbidden default -> bypass -> fail.
    assert _conv(["./bin/check", cmd]) is False


@_pytest.mark.parametrize("cmd", [
    "cat pytest.ini",                # inspecting a config file, not running pytest
    "echo \"do not run pytest\"",    # mentioning the forbidden tool in text
    "grep pytest Makefile",          # searching for it
])
def test_convention_forbidden_mention_is_not_a_bypass(cmd):
    # compliant run that only MENTIONS the forbidden tool must still pass.
    assert _conv(["./bin/check", cmd]) is True


def test_convention_forbidden_multitoken_app_invocation():
    # the suite uses multi-token forbidden substrings like "python app.py".
    passed, _ = score_task(
        expected_tools=(), forbidden_tools=(), run=_run(["./bin/run", "python app.py"]),
        expected_action="convention",
        required_cmd_substr=("bin/run",), forbidden_cmd_substr=("python app.py",),
    )
    assert passed is False
    passed2, _ = score_task(
        expected_tools=(), forbidden_tools=(), run=_run(["./bin/run", "echo 'python app.py is banned'"]),
        expected_action="convention",
        required_cmd_substr=("bin/run",), forbidden_cmd_substr=("python app.py",),
    )
    assert passed2 is True  # mention in echo is not a bypass
