"""Executor tests. Offline: no network, no API keys, no Hermes install required.

The one test that reads a real hermes-agent checkout parses its source with
``ast`` (never imports it) and is skipped when no checkout is found.
"""
import ast
import json
import math
import subprocess
import sys
import textwrap
import types

import pytest

from evolution.adaptive import hermes_bridge
from evolution.adaptive.core import (
    VIOLATION_REWARD, ContextualBandit, ExperienceStore, choose_strategy, score_outcome,
)
from evolution.adaptive.executor import (
    ALLOWED_TOOLSETS, DENIED_TOOLSETS, Budget, ExecutionOutcome, Executor, FakeExecutor,
    FakeResponse, HermesExecutor, LiveExecutionDisabled, Strategy, Task, audit_violations,
    loop_executor, to_reward,
)

TASK = Task("t1", "Review this function for defects.")
PLAIN = Strategy("plain", system_prompt="Be concise.")
PLANNED = Strategy("planned", toolsets=("todo",))


# ----------------------------------------------------------------- value types

@pytest.mark.parametrize("toolset", ["terminal", "code_execution", "file", "skills",
                                     "memory", "web", "browser", "delegation"])
def test_strategy_refuses_non_allowlisted_toolsets(toolset):
    with pytest.raises(ValueError, match="not permitted"):
        Strategy("s", toolsets=(toolset,))


def test_strategy_validation():
    with pytest.raises(ValueError):
        Strategy("")
    with pytest.raises(ValueError):
        Strategy("s", toolsets=["todo"])          # list, not tuple
    with pytest.raises(ValueError):
        Strategy("s", toolsets=("todo", "todo"))
    assert PLAIN.allowed_tools == frozenset()
    assert PLANNED.allowed_tools == frozenset({"todo_list"})
    assert not set(ALLOWED_TOOLSETS) & set(DENIED_TOOLSETS)


def test_task_and_budget_validation():
    with pytest.raises(ValueError):
        Task("", "x")
    with pytest.raises(ValueError):
        Task("t", "   ")
    for bad in ({"max_iterations": 0}, {"max_iterations": True}, {"timeout_s": 0},
                {"timeout_s": math.inf}, {"max_tokens": 0}, {"max_cost_usd": -1.0},
                {"max_cost_usd": math.nan}):
        with pytest.raises(ValueError):
            Budget(**bad)


def test_outcome_validation():
    base = dict(task_id="t", strategy_id="s", backend="fake", output="",
                completed=True, latency_s=1.0, cost_usd=0.1, cost_status="fake")
    ExecutionOutcome(**base)
    for bad in ({"latency_s": math.nan}, {"latency_s": -1.0}, {"cost_usd": -0.1},
                {"cost_usd": math.inf}, {"completed": 1}, {"violation_reasons": ("",)}):
        with pytest.raises(ValueError):
            ExecutionOutcome(**{**base, **bad})
    outcome = ExecutionOutcome(**{**base, "violation_reasons": ("x",)})
    assert outcome.violation and outcome.to_dict()["violation"] is True
    json.dumps(outcome.to_dict(), allow_nan=False)


def test_audit_violations():
    budget = Budget(max_cost_usd=0.5)
    assert audit_violations(PLAIN, budget) == ()
    assert audit_violations(PLAIN, budget, tool_calls=("terminal", "terminal")) == (
        "disallowed_tool:terminal",)
    assert audit_violations(PLANNED, budget, tool_calls=("todo_list",)) == ()
    assert audit_violations(PLAIN, budget, tool_calls=("todo_list",)) == (
        "disallowed_tool:todo_list",)
    assert audit_violations(PLAIN, budget, guardrail={"action": "warn"}) == ()
    assert audit_violations(PLAIN, budget, guardrail={"action": "halt", "code": "loop"}) == (
        "guardrail_halt:loop",)
    assert audit_violations(PLAIN, budget, cost_usd=0.51) == ("cost_budget_exceeded",)
    assert audit_violations(PLAIN, budget, cost_usd=None) == ()
    assert audit_violations(PLAIN, Budget(), cost_usd=99.0) == ()


# ----------------------------------------------------------------------- fake

def test_fake_executor_is_deterministic_and_audited():
    responses = {("t1", "plain"): FakeResponse("ok", cost_usd=0.2, latency_s=3.0),
                 ("t1", "planned"): FakeResponse("x", tool_calls=("terminal",))}
    a, b = FakeExecutor(responses), FakeExecutor(responses)
    assert isinstance(a, Executor)
    first = [a.execute(TASK, s) for s in (PLAIN, PLANNED, PLAIN)]
    second = [b.execute(TASK, s) for s in (PLAIN, PLANNED, PLAIN)]
    assert first == second
    assert a.calls == [("t1", "plain"), ("t1", "planned"), ("t1", "plain")]
    assert first[0].output == "ok" and first[0].cost_usd == 0.2 and not first[0].violation
    assert first[1].violation_reasons == ("disallowed_tool:terminal",)
    with pytest.raises(KeyError):
        a.execute(Task("other", "p"), PLAIN)
    fallback = FakeExecutor({}, default=FakeResponse("d"))
    assert fallback.execute(Task("other", "p"), PLAIN).output == "d"


def test_to_reward_maps_outcome_without_judging_quality():
    outcome = FakeExecutor({}, default=FakeResponse(cost_usd=0.2, latency_s=6.0)).execute(TASK, PLAIN)
    reward = to_reward(outcome, quality=0.8, reference_cost_usd=0.1, reference_latency_s=3.0)
    assert (reward.quality, reward.cost_ratio, reward.time_ratio, reward.violation) == (
        0.8, 2.0, 2.0, False)
    bad = FakeExecutor({}, default=FakeResponse(tool_calls=("terminal",))).execute(TASK, PLAIN)
    assert score_outcome(to_reward(bad, quality=1.0, reference_cost_usd=1,
                                   reference_latency_s=1)) == VIOLATION_REWARD
    unknown = FakeExecutor({}, default=FakeResponse(cost_usd=None)).execute(TASK, PLAIN)
    with pytest.raises(ValueError, match="cost unknown"):
        to_reward(unknown, quality=1.0, reference_cost_usd=1, reference_latency_s=1)
    with pytest.raises(ValueError):
        to_reward(outcome, quality=1.0, reference_cost_usd=0, reference_latency_s=1)


def test_loop_executor_drives_choose_strategy(tmp_path):
    catalogue = {"plain": PLAIN, "planned": PLANNED}
    executor = FakeExecutor({
        ("t1", "plain"): FakeResponse("correct", cost_usd=0.1, latency_s=1.0),
        ("t1", "planned"): FakeResponse("correct", cost_usd=0.1, latency_s=1.0,
                                        tool_calls=("terminal",)),
    })
    run = loop_executor(executor, catalogue, lambda context: TASK)

    def verifier(outcome):  # stand-in for the independent verifier gate
        return to_reward(outcome, quality=float(outcome.output == "correct"),
                         reference_cost_usd=0.1, reference_latency_s=1.0)

    policy = ContextualBandit(tuple(catalogue), epsilon=0.0, seed=0)
    with ExperienceStore(tmp_path / "exp.db") as store:
        records = [choose_strategy(policy, "review", run, verifier, store,
                                   skill_version="test/1") for _ in range(4)]
        rows = store.rows()
    assert [r["strategy"] for r in records[:2]] == ["plain", "planned"]  # initial sweep
    assert records[1]["reward"] == VIOLATION_REWARD
    assert all(r["strategy"] == "plain" for r in records[2:])
    assert len(rows) == 4 and json.loads(rows[1]["feedback_json"])["violation"] is True
    with pytest.raises(ValueError):
        loop_executor(executor, {"wrong": PLAIN}, lambda c: TASK)


# -------------------------------------------------------------- hermes bridge

def _request(**overrides):
    request = {"hermes_repo": "/repo", "task_id": "t1", "prompt": "p",
               "system_prompt": "sys", "enabled_toolsets": [],
               "disabled_toolsets": list(DENIED_TOOLSETS), "max_iterations": 5,
               "run_budget_seconds": 60.0, "max_tokens": None, "model": None,
               "provider": None}
    request.update(overrides)
    return request


def test_bridge_kwargs_fix_isolation_switches():
    kwargs = hermes_bridge.build_agent_kwargs(_request(skip_memory=False, quiet_mode=False))
    assert kwargs["enabled_toolsets"] == []     # [] == no tools in Hermes; None would be all
    assert "terminal" in kwargs["disabled_toolsets"]
    assert kwargs["quiet_mode"] is True and kwargs["skip_memory"] is True
    assert kwargs["skip_context_files"] is True and kwargs["skip_background_review"] is True
    assert kwargs["save_trajectories"] is False and kwargs["checkpoints_enabled"] is False
    assert kwargs["ephemeral_system_prompt"] == "sys" and kwargs["max_iterations"] == 5
    assert "provider" not in kwargs and "max_tokens" not in kwargs
    full = hermes_bridge.build_agent_kwargs(_request(provider="openrouter", max_tokens=100))
    assert full["provider"] == "openrouter" and full["max_tokens"] == 100
    with pytest.raises(ValueError):
        hermes_bridge.build_agent_kwargs(_request(enabled_toolsets=None))


def test_bridge_summarise_result():
    result = {
        "final_response": "done", "completed": True, "failed": False, "partial": "no",
        "api_calls": 2, "input_tokens": 10, "output_tokens": True, "cache_read_tokens": 900,
        "estimated_cost_usd": math.nan, "cost_status": "estimated",
        "guardrail": {"action": "halt", "code": "loop", "tool_name": "x", "count": 3},
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "todo_list"}},
                                                 {"name": "terminal"}, "junk"]},
            {"role": "tool", "content": "r"},
        ],
    }
    summary = hermes_bridge.summarise_result(result, 1.5)
    assert summary["tool_calls"] == ["todo_list", "terminal", "<unparseable>"]
    assert summary["estimated_cost_usd"] is None
    assert "partial" not in summary and "output_tokens" not in summary
    assert summary["cache_read_tokens"] == 900
    assert summary["guardrail"] == {"action": "halt", "code": "loop", "tool_name": "x"}
    json.dumps(summary, allow_nan=False)
    with pytest.raises(TypeError):
        hermes_bridge.summarise_result("text", 0.0)


def _stub_run_agent(monkeypatch, result=None, init_error=None):
    seen = {}

    class AIAgent:
        def __init__(self, **kwargs):
            if init_error:
                raise init_error
            seen["kwargs"] = kwargs

        def run_conversation(self, user_message, task_id=None):
            seen["call"] = (user_message, task_id)
            print("noise on stdout")
            return result

        def close(self):
            seen["closed"] = True

    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=AIAgent))
    return seen


def test_bridge_main_calls_documented_api(monkeypatch, tmp_path, capsys):
    seen = _stub_run_agent(monkeypatch, {"final_response": "hi", "completed": True,
                                         "estimated_cost_usd": 0.01, "cost_status": "estimated"})
    req, res = tmp_path / "req.json", tmp_path / "res.json"
    req.write_text(json.dumps(_request(hermes_repo=str(tmp_path))))
    assert hermes_bridge.main([str(req), str(res)]) == 0
    payload = json.loads(res.read_text())
    assert payload["ok"] is True and payload["final_response"] == "hi"
    assert seen["call"] == ("p", "t1") and seen["closed"] is True
    assert seen["kwargs"]["enabled_toolsets"] == []
    assert capsys.readouterr().out == ""   # agent noise redirected to stderr


def test_bridge_main_reports_errors(monkeypatch, tmp_path):
    _stub_run_agent(monkeypatch, init_error=RuntimeError("no provider configured"))
    req, res = tmp_path / "req.json", tmp_path / "res.json"
    req.write_text(json.dumps(_request()))
    assert hermes_bridge.main([str(req), str(res)]) == 0
    payload = json.loads(res.read_text())
    assert payload == {"ok": False, "exception_type": "RuntimeError",
                       "error": "RuntimeError: no provider configured"}
    assert hermes_bridge.main([]) == 2


# ------------------------------------------------------------ hermes executor

@pytest.fixture
def fake_repo(tmp_path):
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    (repo / "run_agent.py").write_text("raise ImportError('not used by fake runner')\n")
    return repo


def _runner_writing(payload, calls, returncode=0, stderr=""):
    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        if payload is not None:
            with open(argv[-1], "w") as handle:
                json.dump(payload, handle)
        return subprocess.CompletedProcess(argv, returncode, "", stderr)
    return runner


def test_hermes_executor_is_off_by_default(fake_repo):
    calls = []
    executor = HermesExecutor(hermes_repo=fake_repo, runner=_runner_writing({}, calls))
    assert isinstance(executor, Executor)
    with pytest.raises(LiveExecutionDisabled):
        executor.execute(TASK, PLAIN)
    assert calls == []


def test_hermes_executor_requires_explicit_model(fake_repo):
    calls = []
    executor = HermesExecutor(hermes_repo=fake_repo, allow_live=True,
                              runner=_runner_writing({}, calls))
    with pytest.raises(ValueError, match="explicit model"):
        executor.execute(TASK, PLAIN)
    assert calls == []


def test_hermes_executor_request_and_outcome(fake_repo):
    calls = []
    payload = {"ok": True, "final_response": "found 2 defects", "completed": True,
               "failed": False, "partial": False, "interrupted": False, "latency_s": 4.0,
               "estimated_cost_usd": 0.03, "cost_status": "estimated", "api_calls": 1,
               "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
               "model": "m", "provider": "p", "tool_calls": [], "cache_read_tokens": 950}
    budget = Budget(max_iterations=3, timeout_s=50, startup_grace_s=10)
    executor = HermesExecutor(hermes_repo=fake_repo, python="/hermes/python", budget=budget,
                              model="m", allow_live=True,
                              runner=_runner_writing(payload, calls))
    outcome = executor.execute(TASK, PLAIN)
    argv, kwargs = calls[0]
    assert argv[0] == "/hermes/python" and argv[1] == "-c"
    assert "from run_agent import AIAgent" in argv[2]
    assert kwargs["cwd"] == str(fake_repo.resolve()) and kwargs["timeout"] == 60
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert (outcome.output, outcome.completed, outcome.cost_usd, outcome.latency_s) == (
        "found 2 defects", True, 0.03, 4.0)
    assert outcome.backend == "hermes" and not outcome.violation and outcome.error is None
    assert outcome.details["total_tokens"] == 120
    assert outcome.details["cache_read_tokens"] == 950


def test_hermes_executor_request_contents(fake_repo):
    request = HermesExecutor(hermes_repo=fake_repo).build_request(TASK, PLANNED, fake_repo)
    assert request["enabled_toolsets"] == ["todo"]
    assert request["disabled_toolsets"] == list(DENIED_TOOLSETS)
    assert request["prompt"] == TASK.prompt and request["task_id"] == "t1"


@pytest.mark.parametrize("payload,expect", [
    ({"ok": True, "completed": True, "cost_status": "unknown", "estimated_cost_usd": 0.0},
     {"cost_usd": None, "completed": True}),
    ({"ok": True, "completed": False, "failed": True, "error": "rate limited",
      "cost_status": "estimated", "estimated_cost_usd": 0.01},
     {"completed": False, "error": "rate limited"}),
    ({"ok": True, "completed": False, "turn_exit_reason": "max_iterations_reached(3/3)",
      "cost_status": "estimated", "estimated_cost_usd": 0.01},
     {"completed": False, "error": "max_iterations_reached(3/3)"}),
    ({"ok": True, "completed": True, "tool_calls": ["terminal"], "cost_status": "included",
      "estimated_cost_usd": 0.0},
     {"violation_reasons": ("disallowed_tool:terminal",), "cost_usd": 0.0}),
    ({"ok": True, "completed": True, "guardrail": {"action": "block", "code": "c"},
      "cost_status": "estimated", "estimated_cost_usd": 5.0},
     {"violation_reasons": ("guardrail_block:c", "cost_budget_exceeded")}),
    ({"ok": False, "error": "ModuleNotFoundError: No module named 'openai'"},
     {"completed": False, "cost_usd": None,
      "error": "hermes_error: ModuleNotFoundError: No module named 'openai'"}),
])
def test_hermes_executor_outcome_mapping(fake_repo, payload, expect):
    executor = HermesExecutor(hermes_repo=fake_repo, allow_live=True, model="m",
                              budget=Budget(max_cost_usd=1.0),
                              runner=_runner_writing(payload, []))
    outcome = executor.execute(TASK, PLAIN)
    for key, value in expect.items():
        assert getattr(outcome, key) == value, key


def test_hermes_executor_failures(fake_repo):
    no_result = HermesExecutor(hermes_repo=fake_repo, allow_live=True, model="m",
                               runner=_runner_writing(None, [], returncode=1, stderr="boom"))
    outcome = no_result.execute(TASK, PLAIN)
    assert outcome.error == "bridge_failed: exit 1: boom" and outcome.cost_usd is None

    def timeout_runner(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
    timed_out = HermesExecutor(hermes_repo=fake_repo, allow_live=True, model="m", runner=timeout_runner,
                               budget=Budget(timeout_s=1, startup_grace_s=1))
    outcome = timed_out.execute(TASK, PLAIN)
    assert outcome.error == "timeout: killed after 2s" and not outcome.completed

    with pytest.raises(FileNotFoundError):
        HermesExecutor(hermes_repo=fake_repo.parent / "missing", allow_live=True,
                       model="m").execute(TASK, PLAIN)


def test_hermes_executor_real_subprocess_against_stub_checkout(tmp_path):
    """Real subprocess + real bridge, stub ``run_agent``: proves the import path
    resolves inside the Hermes checkout (not this package's ``cli.py``)."""
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    (repo / "cli.py").write_text("ORIGIN = 'hermes-checkout'\n")
    (repo / "run_agent.py").write_text(textwrap.dedent("""
        import cli
        assert cli.ORIGIN == 'hermes-checkout', cli.__file__

        class AIAgent:
            def __init__(self, **kwargs):
                assert kwargs['enabled_toolsets'] == [] and kwargs['skip_memory'] is True
                self.kwargs = kwargs

            def run_conversation(self, user_message, task_id=None):
                print('hermes chatter on stdout')
                return {'final_response': 'echo:' + user_message, 'completed': True,
                        'estimated_cost_usd': 0.002, 'cost_status': 'estimated',
                        'api_calls': 1, 'messages': []}

            def close(self):
                pass
    """))
    executor = HermesExecutor(hermes_repo=repo, python=sys.executable, allow_live=True, model="m",
                              budget=Budget(timeout_s=30, startup_grace_s=30))
    outcome = executor.execute(TASK, PLAIN)
    assert outcome.error is None, outcome.error
    assert outcome.output == "echo:" + TASK.prompt
    assert outcome.completed and outcome.cost_usd == 0.002 and outcome.wall_time_s > 0


# ------------------------------------------- contract with a real Hermes checkout

def _hermes_checkout():
    from evolution.core.config import get_hermes_agent_path
    try:
        repo = get_hermes_agent_path()
    except FileNotFoundError:
        return None
    return repo if (repo / "run_agent.py").is_file() else None


def _params(tree, class_name, method):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    args = item.args
                    return {a.arg for a in args.args + args.kwonlyargs}
    return None


@pytest.mark.skipif(_hermes_checkout() is None, reason="no hermes-agent checkout found")
def test_contract_with_hermes_checkout_source():
    """Parse (never import) hermes-agent to confirm every name the bridge uses exists."""
    repo = _hermes_checkout()
    run_agent = ast.parse((repo / "run_agent.py").read_text(encoding="utf-8"))
    init = _params(run_agent, "AIAgent", "__init__")
    assert init is not None and _params(run_agent, "AIAgent", "close") is not None
    kwargs = hermes_bridge.build_agent_kwargs(_request(provider="x", max_tokens=1))
    assert set(kwargs) <= init, set(kwargs) - init

    facade = ast.parse((repo / "agent" / "turn_facade.py").read_text(encoding="utf-8"))
    run_conv = _params(facade, "TurnFacadeMixin", "run_conversation")
    assert run_conv is not None and {"user_message", "task_id"} <= run_conv

    toolsets = (repo / "toolsets.py").read_text(encoding="utf-8")
    assert '"todo": _ts(' in toolsets and '"todo_list"' in toolsets
    for name in DENIED_TOOLSETS:
        assert f'"{name}": _ts(' in toolsets, name
    # [] must mean "no tools": only None selects every toolset.
    model_tools = (repo / "model_tools.py").read_text(encoding="utf-8")
    assert "if enabled_toolsets is not None:" in model_tools
