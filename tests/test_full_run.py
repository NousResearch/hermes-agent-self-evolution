"""Tests for the full-run driver. Offline: no Hermes, no network, no keys."""
import json
import subprocess

import pytest

from evolution.adaptive import full_run as fr
from evolution.adaptive.executor import Budget, HermesExecutor, Strategy, Task


@pytest.fixture
def hermes_repo(tmp_path):
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    (repo / "run_agent.py").write_text("# placeholder; the stub runner never starts it\n")
    return repo


def make_executor(hermes_repo, runner=fr.stub_runner, model="stub-model"):
    return HermesExecutor(hermes_repo=hermes_repo, model=model, provider="stub",
                          budget=Budget(), allow_live=True, runner=runner)


def make_run(tmp_path, hermes_repo, *, stop_usd=18.0, runner=fr.stub_runner, seeds=2,
             skill="# Review skill\nLook for defects.\n"):
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    ledger = fr.SpendLedger(run_dir / "ledger.jsonl", stop_usd=stop_usd)
    return fr.FullRun(run_dir=run_dir, skill_text=skill, skill_source="test",
                      executor=make_executor(hermes_repo, runner), ledger=ledger, live=False,
                      seeds=seeds, train_episodes=20)


# ------------------------------------------------------------------ ledger

def test_ledger_refuses_before_crossing_the_stop(tmp_path):
    ledger = fr.SpendLedger(tmp_path / "l.jsonl", stop_usd=0.05)
    ledger.record({"stage": "x", "cost_usd": 0.03})
    with pytest.raises(fr.BudgetExhausted):
        ledger.reserve(0.03)
    assert ledger.exhausted


def test_ledger_charges_unknown_cost_and_persists(tmp_path):
    path = tmp_path / "l.jsonl"
    ledger = fr.SpendLedger(path, stop_usd=1.0)
    ledger.record({"stage": "x", "cost_usd": None})
    ledger.record({"stage": "y", "cost_usd": None, "charge_if_unknown": 0.1})
    assert ledger.spent == pytest.approx(fr.UNKNOWN_COST_CHARGE_USD + 0.1)
    assert fr.SpendLedger(path, stop_usd=1.0).spent == pytest.approx(ledger.spent)


def test_stop_may_not_exceed_the_approved_cap(tmp_path):
    with pytest.raises(ValueError):
        fr.SpendLedger(tmp_path / "l.jsonl", stop_usd=fr.HARD_CAP_USD + 1)


def test_replay_pays_once_and_keys_on_request_content(tmp_path, hermes_repo):
    ledger = fr.SpendLedger(tmp_path / "l.jsonl", stop_usd=1.0)
    inner = make_executor(hermes_repo)
    ex = fr.LedgerExecutor(inner, ledger, stage="t")
    task = Task("case-1", "review this")
    first = ex.execute(task, Strategy("a", system_prompt="X"))
    again = ex.execute(task, Strategy("a", system_prompt="X"))
    assert (ex.fresh, ex.replayed) == (1, 1) and again.output == first.output
    ex.execute(task, Strategy("b", system_prompt="X"))            # other strategy ID
    ex.execute(task, Strategy("a", system_prompt="Y"))            # other system prompt
    fr.LedgerExecutor(inner, ledger, stage="t", sample=1).execute(task, Strategy("a", system_prompt="X"))
    assert len(ledger.rows) == 4
    # A new process replays from disk.
    resumed = fr.LedgerExecutor(inner, fr.SpendLedger(tmp_path / "l.jsonl", stop_usd=1.0), stage="t")
    resumed.execute(task, Strategy("a", system_prompt="X"))
    assert (resumed.fresh, resumed.replayed) == (0, 1)


def test_failed_calls_are_charged_retried_and_not_replayed(tmp_path, hermes_repo):
    def failing(argv, cwd=None, **kw):
        return subprocess.CompletedProcess(argv, 1, "", "boom")   # no result file
    ledger = fr.SpendLedger(tmp_path / "l.jsonl", stop_usd=1.0)
    ex = fr.LedgerExecutor(make_executor(hermes_repo, failing), ledger, stage="t", retries=1)
    out = ex.execute(Task("c", "p"), Strategy("a"))
    assert out.cost_usd is None and ex.fresh == 2
    assert ledger.spent == pytest.approx(2 * fr.UNKNOWN_COST_CHARGE_USD)
    assert ledger.lookup(fr.request_key(ex.inner, Task("c", "p"), Strategy("a"), 0)) is None


# ------------------------------------------------------------------ stages

def test_stub_dry_run_canary_and_compare(tmp_path, hermes_repo):
    run = make_run(tmp_path, hermes_repo)
    report = run.run(["canary", "compare"])
    assert report["mode"] == "stub"
    assert report["stages"]["canary"]["status"] == "done"
    comparison = report["stages"]["compare"]["result"]
    n_cases = len(run.train_cases) + len(run.validation_cases)
    executions = comparison["report"]["executions"]                     # one per visited pair
    assert 2 * len(run.validation_cases) <= executions <= 2 * n_cases
    assert executions == comparison["fresh_calls"] + comparison["replayed_calls"]
    assert report["spend"]["by_stage"]["compare"]["calls"] == comparison["fresh_calls"]
    assert report["sealed_split_open_attempts"] == 0
    assert set(comparison["report"]["policies"]) == {
        "fixed:direct", "fixed:skill", "random", "epsilon_greedy", "linucb"}
    assert (run.run_dir / "report.json").exists() and (run.run_dir / "comparison.md").exists()


def test_canary_stops_the_run_on_unparsable_replies(tmp_path, hermes_repo):
    def prose(argv, cwd=None, **kw):
        fr.stub_runner(argv, cwd=cwd, **kw)
        result = json.load(open(argv[-1]))
        result["final_response"] = "Here are the defects I found: ..."
        json.dump(result, open(argv[-1], "w"))
        return subprocess.CompletedProcess(argv, 0, "", "")
    report = make_run(tmp_path, hermes_repo, runner=prose).run(["canary", "compare"])
    assert report["stages"]["canary"]["status"] == "stopped"
    assert "compare" not in report["stages"]


def test_budget_stop_ends_the_run_with_a_report(tmp_path, hermes_repo):
    report = make_run(tmp_path, hermes_repo, stop_usd=0.2).run(["canary", "compare"])
    assert report["stages"]["compare"]["status"] == "stopped"
    assert "BudgetExhausted" in report["stages"]["compare"]["reason"]
    assert report["spend"]["spent_usd"] <= 0.2


def test_skipped_stage_is_not_marked_done(tmp_path, hermes_repo):
    run = make_run(tmp_path, hermes_repo)
    report = run.run(["gepa"], gepa_factory=None)
    assert report["stages"]["gepa"]["status"] == "skipped"


def _require_gepa():
    dspy = pytest.importorskip("dspy")
    import inspect
    params = inspect.signature(dspy.GEPA.__init__).parameters
    if not {"reflection_lm", "max_metric_calls"} <= set(params):
        pytest.skip("installed dspy.GEPA lacks reflection_lm/max_metric_calls")


def test_gepa_stage_with_stub_lms(tmp_path, hermes_repo):
    _require_gepa()
    run = make_run(tmp_path, hermes_repo)
    report = run.run(["gepa"], gepa_factory=fr.stub_gepa_factory(20), gepa_max_metric_calls=20)
    gepa = report["stages"]["gepa"]
    assert gepa["status"] == "done", gepa
    metrics = gepa["result"]["metrics"]
    assert metrics["optimizer"] == "dspy.GEPA" and metrics["data"].endswith("train split only")
    assert metrics["identical_to_baseline"]          # DummyLM cannot improve the score
    spend = report["spend"]["by_stage"]
    assert spend["gepa:task"]["calls"] > 0 and spend["gepa:task"]["cost_usd"] == 0


def test_review_prompt_matches_admission_when_available():
    try:
        from evolution.adaptive import admission
    except ImportError:
        pytest.skip("gate 6 admission module not on this branch")
    assert admission.REVIEW_PROMPT == fr.REVIEW_PROMPT
    assert admission.review_prompt("x = 1\n") == fr.review_prompt("x = 1\n")


# --------------------------------------------------------------------- CLI

@pytest.mark.parametrize("extra, message", [
    (["--live"], "--live needs explicit --model and --stop-usd"),
    (["--live", "--model", "m", "--stop-usd", "5", "--stages", "gepa"], "--gepa-via-hermes"),
    (["--gepa-via-hermes", "--gepa-allow-metered"], "mutually exclusive"),
])
def test_cli_refuses_unsafe_live_invocations(tmp_path, capsys, extra, message):
    skill = tmp_path / "skill.md"
    skill.write_text("skill")
    with pytest.raises(SystemExit):
        fr.main(["--run-dir", str(tmp_path / "run"), "--skill-file", str(skill), *extra])
    assert message in capsys.readouterr().err


def test_cli_refuses_a_run_dir_inside_the_repo(tmp_path, capsys):
    skill = tmp_path / "skill.md"
    skill.write_text("skill")
    with pytest.raises(SystemExit):
        fr.main(["--run-dir", "output/full-run", "--skill-file", str(skill)])
    assert "outside the repository" in capsys.readouterr().err


# ------------------------------------------------- persistent run-dir default

def test_evolution_home_defaults_and_env_override(tmp_path):
    assert fr.evolution_home({}) == fr.Path.home() / ".local" / "share" / "hermes-evolution"
    env = {fr.EVOLUTION_HOME_ENV: str(tmp_path / "evo")}
    assert fr.evolution_home(env) == tmp_path / "evo"
    assert fr.default_run_dir("kafo-evo-001/live-2", env) == tmp_path / "evo" / "runs" / "kafo-evo-001" / "live-2"


@pytest.mark.parametrize("name", ["", "/abs/run", "../escape", "a/../../b"])
def test_default_run_dir_rejects_unsafe_names(name):
    with pytest.raises(ValueError):
        fr.default_run_dir(name, {})


def test_under_tmpdir(tmp_path):
    env = {"TMPDIR": str(tmp_path)}
    assert fr._under_tmpdir(tmp_path / "kafo-fullrun" / "live-1", env)
    assert fr._under_tmpdir(fr.Path("/tmp/x"), {})
    assert not fr._under_tmpdir(fr.Path("/nonexistent-persistent/hermes-evolution/runs/x"), {})


def _cli_err(capsys, argv):
    with pytest.raises(SystemExit):
        fr.main(argv)
    return capsys.readouterr().err


def test_cli_run_name_uses_the_persistent_root_without_warning(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(fr.EVOLUTION_HOME_ENV, "/nonexistent-persistent/hermes-evolution")
    skill = tmp_path / "skill.md"
    skill.write_text("skill")
    err = _cli_err(capsys, ["--run-name", "kafo-evo-001/live-2", "--skill-file", str(skill), "--live"])
    assert "--live needs explicit --model" in err           # got past run-dir resolution
    assert "WARNING" not in err


def test_cli_warns_when_the_run_dir_is_in_tmpdir(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    skill = tmp_path / "skill.md"
    skill.write_text("skill")
    err = _cli_err(capsys, ["--run-dir", str(tmp_path / "run"), "--skill-file", str(skill), "--live"])
    assert "WARNING" in err and "temporary directory" in err and fr.EVOLUTION_HOME_ENV in err


@pytest.mark.parametrize("args, message", [
    ([], "give --run-name"),
    (["--run-dir", "/nonexistent-persistent/r", "--run-name", "x"], "not both"),
    (["--run-name", "../x"], "without '..'"),
])
def test_cli_run_dir_selection_errors(tmp_path, capsys, args, message):
    skill = tmp_path / "skill.md"
    skill.write_text("skill")
    assert message in _cli_err(capsys, [*args, "--skill-file", str(skill)])


def test_gepa_budget_stop_terminates_the_stage(tmp_path, hermes_repo):
    _require_gepa()
    run = make_run(tmp_path, hermes_repo, stop_usd=0.05)
    report = run.run(["gepa"], gepa_factory=fr.stub_gepa_factory(40, stub_cost_usd=0.01),
                     gepa_max_metric_calls=40)
    assert report["stages"]["gepa"]["status"] == "stopped"
    assert report["spend"]["spent_usd"] <= 0.05 + 1e-9
    assert not list((run.run_dir / "gepa").glob("**/evolved_skill.md")) \
        if (run.run_dir / "gepa").exists() else True


def test_a_run_dir_cannot_be_reused_with_another_identity(tmp_path, hermes_repo):
    make_run(tmp_path, hermes_repo).run(["canary"])
    run_dir = tmp_path / "run"
    other = fr.FullRun(run_dir=run_dir, skill_text="# Review skill\nLook for defects.\n",
                       skill_source="test",
                       executor=make_executor(hermes_repo, model="another-model"),
                       ledger=fr.SpendLedger(run_dir / "ledger.jsonl", stop_usd=18.0),
                       live=False, seeds=2, train_episodes=20)
    with pytest.raises(fr.RunMismatch, match="model"):
        other.run(["canary"])


def test_frontmatter_is_kept_out_of_the_optimised_text():
    front, body = fr.split_frontmatter("---\nname: github-code-review\n---\n\nBody text.\n")
    assert front == "name: github-code-review" and body == "Body text.\n"
    assert fr.split_frontmatter("No frontmatter.\n") == (None, "No frontmatter.\n")


def test_secret_values_are_redacted(monkeypatch):
    monkeypatch.setenv("EXAMPLE_API_KEY", "sk-test-1234567890")
    assert fr.redact("auth failed for sk-test-1234567890") == \
        "auth failed for <redacted:EXAMPLE_API_KEY>"


def _oracle_runner(run, better_marker):
    truth = {fr.review_prompt(c.code): c for c in (*run.train_cases, *run.validation_cases)}

    def runner(argv, cwd=None, **kw):
        fr.stub_runner(argv, cwd=cwd, **kw)
        request, result = json.load(open(argv[-2])), json.load(open(argv[-1]))
        if better_marker in (request.get("system_prompt") or ""):
            case = truth[request["prompt"]]
            result["final_response"] = json.dumps([
                {"line_start": d.line_start, "line_end": d.line_end, "category": d.category,
                 "description": "stub"} for d in case.defects])
        json.dump(result, open(argv[-1], "w"))
        return subprocess.CompletedProcess(argv, 0, "", "")
    return runner


@pytest.mark.parametrize("better, status", [(True, "kept"), (False, "rejected")])
def test_admission_uses_review_with_rollback(tmp_path, hermes_repo, better, status):
    adm = pytest.importorskip("evolution.adaptive.admission")
    if not hasattr(adm, "review_with_rollback"):
        pytest.skip("gate 7 admission path (PR #12) not on this branch")
    skill = "# Review skill\nLook for defects.\n"
    run = make_run(tmp_path, hermes_repo, skill=skill)
    run.executor._runner = _oracle_runner(run, "BETTER" if better else "NEVER-MATCHES")
    gepa_dir = run.run_dir / "gepa" / fr.SKILL_NAME / "stub"
    gepa_dir.mkdir(parents=True)
    (gepa_dir / "baseline_skill.md").write_text(skill)
    (gepa_dir / "evolved_skill.md").write_text(skill + "BETTER: stub edit.\n")
    (gepa_dir / "metrics.json").write_text(json.dumps({"skill_name": fr.SKILL_NAME,
                                                       "constraints_passed": True}))
    run.state["stages"]["gepa"] = {"status": "done", "result": {
        "output_dir": str(gepa_dir),
        "metrics": {"identical_to_baseline": False, "constraints_passed": True}}}
    report = run.run(["compare", "admission"])
    result = report["stages"]["admission"]["result"]
    assert result["status"] == status, result
    events = [e["event"] for e in result["checkpoint_log"]]
    assert events[0] == "save"                       # checkpoint BEFORE gate 6 decides
    if better:
        assert result["registry_status"] == "admitted" and events[-1] == "promote"
        assert result["active_skill"] == result["candidate_id"]
    else:
        assert result["registry_status"] == "rejected" and events[-1] == "admission_rejected"
        assert result["active_skill"] == result["parent_id"] == report["baseline_id"]
    # gate 7's catalogue arm is measured with its own calls, not gate 6's replays
    assert report["spend"]["by_stage"]["admission"]["calls"] >= 60
    assert report["sealed_split_open_attempts"] == 0


# ------------------------------------------------------ GEPA through Hermes

def test_gepa_via_hermes_uses_the_bridge_and_the_ledger(tmp_path, hermes_repo, monkeypatch):
    _require_gepa()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)     # no metered key needed
    run = make_run(tmp_path, hermes_repo)
    report = run.run(["gepa"], gepa_factory=fr.hermes_gepa_factory(run.executor),
                     gepa_max_metric_calls=20)
    gepa = report["stages"]["gepa"]
    assert gepa["status"] == "done", gepa
    route = gepa["result"]["metrics"]["lm_route"]
    assert route == {"task": "hermes:stub:stub-model", "reflection": "hermes:stub:stub-model"}
    spend = report["spend"]["by_stage"]
    assert spend["gepa:task"]["calls"] > 0 and spend["gepa:reflection"]["calls"] > 0
    assert spend["gepa:task"]["cost_usd"] > 0                  # stub estimates are recorded
    rows = [r for r in run.ledger.rows if r["stage"].startswith("gepa:")]
    assert rows and all(r["kind"] == "hermes" and r["backend"] == "hermes" for r in rows)


def test_gepa_via_hermes_fails_closed(tmp_path, hermes_repo):
    _require_gepa()
    calls = {"gepa": 0}

    def flaky(argv, cwd=None, **kw):
        request = json.load(open(argv[-2]))
        if "Your output fields are:" in (request.get("system_prompt") or ""):   # any DSPy call
            calls["gepa"] += 1
            if calls["gepa"] > 3:                                 # fail after the preflight
                return subprocess.CompletedProcess(argv, 1, "", "bridge down")
        return fr.stub_runner(argv, cwd=cwd, **kw)

    run = make_run(tmp_path, hermes_repo, runner=flaky)
    report = run.run(["gepa"], gepa_factory=fr.hermes_gepa_factory(run.executor),
                     gepa_max_metric_calls=20)
    gepa = report["stages"]["gepa"]
    assert gepa["status"] == "stopped" and "GepaCallFailed" in gepa["reason"]
    assert calls["gepa"] == 4                                     # no calls after the failure
    assert not (run.run_dir / "gepa").exists()


def test_gepa_via_hermes_respects_the_stop(tmp_path, hermes_repo):
    _require_gepa()
    run = make_run(tmp_path, hermes_repo, stop_usd=0.25)
    report = run.run(["gepa"], gepa_factory=fr.hermes_gepa_factory(run.executor),
                     gepa_max_metric_calls=40)
    assert report["stages"]["gepa"]["status"] == "stopped"
    assert report["spend"]["spent_usd"] <= 0.25
