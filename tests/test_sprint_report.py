"""Offline tests for the gate 8 sprint report and the one-time final evaluation."""
import ast
import copy
import json
from pathlib import Path

import pytest

from evolution.adaptive import benchmark as bm
from evolution.adaptive import evaluation as ev
from evolution.adaptive import final_evaluation as fe
from evolution.adaptive import report as rp
from evolution.adaptive.registry import REJECTED, CandidateRegistry
from tests.test_benchmark_integration import _scan

REPO = Path(__file__).resolve().parents[1]
SMALL = dict(seeds=3, train_episodes=40, resamples=50, verifier_bootstrap=50)


@pytest.fixture(scope="module")
def small_report():
    return rp.generate(final_ledger_path=REPO / "does-not-exist.json", **SMALL)


# ── Report content ─────────────────────────────────────────────────────────

def test_report_is_labelled_synthetic(small_report):
    md = rp.format_markdown(small_report)
    assert small_report["synthetic"] is True
    assert md.splitlines()[2].startswith("> **SYNTHETIC RESULTS.")
    assert "Nothing in this report is evidence that Hermes improves" in small_report["notice"]
    assert "results from a hypothetical simulator, not from Hermes" in small_report["headline"][-1]
    assert "evidence of real-world improvement" in small_report["headline"][-1]
    bare = rp.generate(final_ledger_path=REPO / "does-not-exist.json", live_smoke_path=None,
                       live_run_path=None, **SMALL)
    assert bare["headline"][-1] == "These are results from a hypothetical simulator, not from Hermes."
    assert bare["live_smoke"]["status"] == "pending" and bare["notice"] == rp.NOTICE
    assert "smoke" not in bare["notice"]


def test_same_settings_reproduce_identical_output(small_report):
    again = rp.generate(final_ledger_path=REPO / "does-not-exist.json", **SMALL)
    assert rp.to_json(again) == rp.to_json(small_report)
    assert rp.format_markdown(again) == rp.format_markdown(small_report)
    assert "generated_at" in small_report and small_report["generated_at"] is None


def test_headline_follows_the_verdicts(small_report):
    cmp_ = copy.deepcopy(small_report["policy_comparison"])
    best = cmp_["best_fixed_baseline"]
    for c in cmp_["comparisons"]:
        c["verdict"] = "no improvement detected"
    assert rp.headline(cmp_)[0].startswith("No adaptive policy shows an improvement")
    for c in cmp_["comparisons"]:
        if c["baseline"] == best and c["phase"] == "validation" and c["policy"] == "linucb":
            c["verdict"] = "improvement"
    assert "linucb improved on the best fixed strategy" in rp.headline(cmp_)[0]


def test_full_report_states_no_improvement_over_best_fixed():
    """The committed report's headline comes from the gate 4 verdicts."""
    data = json.loads((rp.DEFAULT_OUT / "report.json").read_text(encoding="utf-8"))
    assert any(h.startswith("Simulator: No adaptive policy shows an improvement over the best "
                            "fixed strategy") for h in data["headline"])
    val = [c for c in data["policy_comparison"]["comparisons"]
           if c["baseline"] == data["policy_comparison"]["best_fixed_baseline"]
           and c["phase"] == "validation"]
    assert val and all(c["verdict"] != "improvement" for c in val)


def test_verifier_reference_scores(small_report):
    reviewers = small_report["verifier_scores"]["reviewers"]
    for split in bm.OPEN_SPLITS:
        assert reviewers["ground_truth"][split]["micro"]["f1"]["point"] == 1.0
        assert reviewers["empty"][split]["micro"]["f1"]["point"] == 0.0
        noisy = reviewers["noisy_reference"][split]["micro"]["f1"]
        assert 0.0 < noisy["point"] < 1.0 and noisy["low"] <= noisy["point"] <= noisy["high"]
    assert set(small_report["verifier_scores"]["split_sizes"]) == set(bm.OPEN_SPLITS)


def test_safety_section_and_pending_gates(small_report):
    safety = small_report["safety"]
    assert safety["sealed_access_blocked_during_generation"] == 0
    cv = safety["critical_violations"]
    assert cv["total_violations"] == sum(
        p[ph]["violations"] for p in cv["by_policy"].values() for ph in p)
    from evolution.adaptive.admission import ADMISSION_POLICY
    assert safety["admission"]["status"] == "merged"
    assert safety["admission"]["policy_sha256"] == ADMISSION_POLICY.fingerprint()
    # registry and checkpoints come from the committed live-run snapshot by default
    assert safety["rollbacks"]["status"] == safety["registry"]["status"] == "supplied"
    bare = rp.generate(final_ledger_path=REPO / "nope.json", live_run_path=None, **SMALL)
    assert bare["safety"]["rollbacks"]["status"] == bare["safety"]["registry"]["status"] \
        == "not supplied"
    assert "0 rollbacks is not a safety result" in bare["safety"]["rollbacks"]["note"]
    live = small_report["live_smoke"]
    assert live["label"] == "4 live cases; a smoke test, not a benchmark result"
    assert "4-case smoke test" in small_report["notice"]
    assert small_report["final_test_evaluation"]["status"] == "not run"


def test_registry_rejections_are_reported(tmp_path):
    db = tmp_path / "registry.db"
    with CandidateRegistry(db) as reg:
        base = reg.register_baseline("code-review", b"base\n")
        cand = reg.propose("code-review", b"evolved\n", parent_id=base.candidate_id)
        reg.transition(cand.candidate_id, REJECTED, reason="worse than baseline",
                       decided_by="test", evidence={"diff": -0.1})
    before = db.read_bytes()
    summary = rp._registry_summary(str(db))
    assert db.read_bytes() == before   # opened read-only
    assert {(c["source"], c["status"], c["count"]) for c in summary["candidates"]} == {
        ("baseline", "admitted", 1), ("gepa", "rejected", 1)}
    assert summary["rejected"][0]["reason"] == "worse than baseline"


def test_checkpoint_rollbacks_are_reported(tmp_path):
    from evolution.adaptive.checkpoint import CheckpointStore
    from evolution.adaptive.core import ContextualBandit
    db = tmp_path / "knowledge.db"
    policy = ContextualBandit(("a", "b"), epsilon=0.1, seed=1)
    with CheckpointStore(db) as store:
        good = store.save(policy, skill_version="skill/1", label="pre-admission")
        policy.update("ctx", "a", 0.9)
        cand = store.save(policy, skill_version="skill/2", label="post-admission",
                          status="candidate")
        store.rollback(policy, from_checkpoint_id=cand.checkpoint_id, reason="utility drop")
    before = db.read_bytes()
    section = rp.checkpoint_section(str(db))
    assert db.read_bytes() == before   # opened read-only
    assert section["checkpoints"] == {"good": 1, "rolled_back": 1}
    assert section["events"]["rollback"] == 1
    assert section["rollbacks"] == [{"to": good.checkpoint_id, "from": cand.checkpoint_id,
                                     "reason": "utility drop"}]
    md = rp.format_markdown(rp.generate(checkpoint_path=str(db),
                                        final_ledger_path=REPO / "nope.json", **SMALL))
    assert "Rollbacks: 1." in md and "rolled back to" in md


def test_transcribed_smoke_results_match_the_sprint_doc():
    """The live smoke JSON is a transcription of the merged sprint doc table."""
    doc = (REPO / "docs/sprints/KAFO-EVO-001.md").read_text(encoding="utf-8")
    data = json.loads(rp.DEFAULT_LIVE_SMOKE.read_text(encoding="utf-8"))
    assert len(data["cases"]) == 4 and data["split"] == "validation"
    for case in data["cases"]:
        row = next(line for line in doc.splitlines() if line.startswith(f"| {case['case_id']} |"))
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert cells[4] == f"{case['cost_usd']:.4f}"
        assert cells[5] == f"{case['latency_s']:.1f}"
        assert cells[6].split("/")[-1].strip() == f"{case['quality']:.2f}"
        assert case["completed"] is True and case["violation"] is False


def test_live_smoke_is_labelled_as_a_handful_of_cases(tmp_path):
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps({"source": "worker -3", "cases": [
        {"case_id": "a", "completed": True, "quality": 0.5, "cost_usd": 0.01,
         "latency_s": 9.0, "violation": False},
        {"case_id": "b", "completed": False, "quality": None, "cost_usd": None,
         "latency_s": 3.0, "violation": False}]}))
    live = rp.live_smoke_section(str(path))
    assert live["label"] == "2 live cases; a smoke test, not a benchmark result"
    assert (live["completed"], live["mean_quality"], live["mean_latency_s"]) == (1, 0.5, 6.0)
    path.write_text(json.dumps({"cases": [{"case_id": "a", "cost_usd": float("inf")}]}))
    with pytest.raises(ValueError):
        rp.live_smoke_section(str(path))


def test_report_reads_a_recorded_final_evaluation(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"status": "failed", "started_at": "t0", "finished_at": "t1",
                                  "reviewer_id": "r", "synthetic": True, "error": "boom",
                                  "unexpected": "dropped"}))
    section = rp.final_evaluation_section(ledger)
    assert section["status"] == "failed" and "unexpected" not in section
    assert "Status **failed**" in rp.format_markdown(
        {**rp.generate(final_ledger_path=ledger, **SMALL)})


def test_committed_report_matches_generator_output():
    """reports/kafo-evo-001 must be regenerated whenever merged code changes it."""
    fresh = rp.generate()
    assert (rp.DEFAULT_OUT / "report.json").read_text(encoding="utf-8") == rp.to_json(fresh)
    assert (rp.DEFAULT_OUT / "report.md").read_text(encoding="utf-8") == \
        rp.format_markdown(fresh)


def test_generator_never_references_the_final_evaluation():
    tree = ast.parse(Path(rp.__file__).read_text(encoding="utf-8"))
    names = {getattr(n, "id", getattr(n, "attr", None)) for n in ast.walk(tree)
             if isinstance(n, (ast.Name, ast.Attribute))}
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                for a in n.names}
    assert not {"final_evaluation", "run_final_evaluation", "evaluate_sealed_test"} & (
        names | imported)
    assert rp.DEFAULT_LEDGER == fe.DEFAULT_LEDGER


# ── One-time final evaluation ──────────────────────────────────────────────

def _reviewer(seen):
    def review(view):
        seen.append(view)
        return []
    return review


def test_final_evaluation_runs_once_and_records_result(tmp_path, caplog):
    ledger, seen = tmp_path / "final.json", []
    with caplog.at_level("WARNING"):
        record = fe.run_final_evaluation(_reviewer(seen), reviewer_id="empty-reference",
                                         synthetic=True, confirm=fe.CONFIRM_PHRASE,
                                         ledger_path=ledger, n_bootstrap=20)
    assert record["status"] == "completed" and record["n_cases"] == len(seen) == 10
    assert record["synthetic"] is True and record["micro"]["f1"]["point"] == 0.0
    assert record["started_at"] <= record["finished_at"]
    assert json.loads(ledger.read_text()) == record
    assert "FINAL TEST EVALUATION completed" in caplog.text
    with pytest.raises(fe.FinalEvaluationRefused, match="already completed"):
        fe.run_final_evaluation(_reviewer([]), reviewer_id="again", synthetic=True,
                                confirm=fe.CONFIRM_PHRASE, ledger_path=ledger)
    assert json.loads(ledger.read_text()) == record


def test_reviewer_never_sees_labels(tmp_path):
    seen = []
    fe.run_final_evaluation(_reviewer(seen), reviewer_id="probe", synthetic=True,
                            confirm=fe.CONFIRM_PHRASE, ledger_path=tmp_path / "l.json",
                            n_bootstrap=5)
    for view in seen:
        assert set(view) == {"case_id", "language", "code", "context"}
        assert set(view["context"]) == {"difficulty", "domain"}
        with pytest.raises(TypeError):
            view["defects"] = []


def test_failed_run_still_counts_as_the_one_run(tmp_path):
    ledger = tmp_path / "final.json"

    def broken(view):
        raise RuntimeError("model crashed")

    with pytest.raises(RuntimeError):
        fe.run_final_evaluation(broken, reviewer_id="broken", synthetic=True,
                                confirm=fe.CONFIRM_PHRASE, ledger_path=ledger)
    assert json.loads(ledger.read_text())["status"] == "failed"
    with pytest.raises(fe.FinalEvaluationRefused, match="already failed"):
        fe.run_final_evaluation(_reviewer([]), reviewer_id="retry", synthetic=True,
                                confirm=fe.CONFIRM_PHRASE, ledger_path=ledger)


@pytest.mark.parametrize("kwargs, error", [
    ({"confirm": "yes"}, fe.FinalEvaluationRefused),
    ({"reviewer_id": " "}, ValueError),
    ({"synthetic": "yes"}, ValueError),
])
def test_final_evaluation_needs_explicit_arguments(tmp_path, kwargs, error):
    args = dict(reviewer_id="r", synthetic=True, confirm=fe.CONFIRM_PHRASE,
                ledger_path=tmp_path / "l.json")
    args.update(kwargs)
    with pytest.raises(error):
        fe.run_final_evaluation(_reviewer([]), **args)
    assert not (tmp_path / "l.json").exists()


def test_final_evaluation_refused_inside_optimisation_phase(tmp_path):
    with bm.optimisation_phase(), pytest.raises(fe.FinalEvaluationRefused):
        fe.run_final_evaluation(_reviewer([]), reviewer_id="r", synthetic=True,
                                confirm=fe.CONFIRM_PHRASE, ledger_path=tmp_path / "l.json")
    assert not (tmp_path / "l.json").exists()


def test_evaluate_sealed_test_needs_exactly_one_review_source():
    with pytest.raises(ValueError):
        ev.evaluate_sealed_test(final_evaluation=True)
    with pytest.raises(ValueError):
        ev.evaluate_sealed_test({}, reviewer=lambda v: [], final_evaluation=True)


def test_only_the_final_step_may_call_evaluate_sealed_test(tmp_path):
    rel = Path("evolution/adaptive/final_evaluation.py")
    assert _scan(REPO / rel, rel) == []
    rogue = tmp_path / "rogue.py"
    rogue.write_text("from . import evaluation\n"
                     "def run_final_evaluation():\n"
                     "    return evaluation.evaluate_sealed_test({}, final_evaluation=True)\n"
                     "def helper():\n"
                     "    return evaluation.evaluate_sealed_test({}, final_evaluation=True)\n",
                     encoding="utf-8")
    problems = _scan(rogue, Path("evolution/adaptive/rogue.py"))
    assert len(problems) == 2 and all("evaluate_sealed_test" in p for p in problems)
    renamed = _scan(rogue, rel)   # right file, but helper() is not the allowed function
    assert len(renamed) == 1 and ":5:" in renamed[0]


# ── Owner-decided live reviewer (tested with a fake executor only) ─────────

from evolution.adaptive import admission as adm  # noqa: E402
from evolution.adaptive.executor import FakeExecutor, FakeResponse, Strategy  # noqa: E402

BASE = Strategy("reviewer")


def test_executor_reviewer_records_skill_version_and_cost(tmp_path):
    strategy = Strategy("reviewer:code-review@abc")
    executor = FakeExecutor({}, default=FakeResponse(output="[]", cost_usd=0.01,
                                                     latency_s=2.0))
    reviewer = fe.ExecutorReviewer(executor, strategy, max_total_cost_usd=1.0)
    record = fe.run_final_evaluation(reviewer, reviewer_id="fake:reviewer",
                                     skill_version="code-review@abc", synthetic=True,
                                     confirm=fe.CONFIRM_PHRASE,
                                     ledger_path=tmp_path / "l.json", n_bootstrap=5)
    rs = record["reviewer_summary"]
    assert record["skill_version"] == "code-review@abc"
    assert (rs["calls"], rs["total_cost_usd"], rs["stopped"]) == (10, 0.1, None)
    assert rs["strategy_id"] == strategy.strategy_id and rs["backend"] == "fake"
    assert all(task_id.startswith("pyrev-") for task_id, _ in executor.calls)
    section = rp.final_evaluation_section(tmp_path / "l.json")
    assert "cases" not in section["reviewer_summary"]
    assert "skill version `code-review@abc`" in rp.format_markdown(
        rp.generate(final_ledger_path=tmp_path / "l.json", **SMALL))


def test_executor_reviewer_stops_spending_at_the_cap_and_survives_errors(tmp_path):
    executor = FakeExecutor({}, default=FakeResponse(output="[]", cost_usd=0.3))
    calls = {"n": 0}
    original = executor.execute

    def flaky(task, strategy):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return original(task, strategy)
    executor.execute = flaky
    reviewer = fe.ExecutorReviewer(executor, Strategy("s"), max_total_cost_usd=0.5)
    record = fe.run_final_evaluation(reviewer, reviewer_id="r", synthetic=True,
                                     confirm=fe.CONFIRM_PHRASE,
                                     ledger_path=tmp_path / "l.json", n_bootstrap=5)
    rs = record["reviewer_summary"]
    assert record["status"] == "completed"
    assert (rs["exceptions"], rs["calls"], rs["skipped"]) == (1, 2, 7)
    assert rs["total_cost_usd"] == 0.6 and "cost cap" in rs["stopped"]


def test_unknown_cost_stops_spending(tmp_path):
    executor = FakeExecutor({}, default=FakeResponse(output="[]", cost_usd=None))
    reviewer = fe.ExecutorReviewer(executor, Strategy("s"), max_total_cost_usd=5.0)
    fe.run_final_evaluation(reviewer, reviewer_id="r", synthetic=True,
                            confirm=fe.CONFIRM_PHRASE, ledger_path=tmp_path / "l.json",
                            n_bootstrap=5)
    rs = reviewer.summary()
    assert (rs["calls"], rs["skipped"]) == (1, 9) and "cost unknown" in rs["stopped"]


def _admit_gepa(registry, validation, base, cand):
    def make_executor(seed):
        responses = {}
        for case in validation:
            truth = json.dumps(ev.ground_truth_findings(case))
            for cid, out in ((cand.candidate_id, truth), (base.candidate_id, "[]")):
                responses[(case.case_id, f"eval:reviewer:{cid}")] = FakeResponse(
                    output=out, cost_usd=0.01, latency_s=2.0)
        return FakeExecutor(responses)
    return adm.review_registered(registry, cand.candidate_id, base=BASE,
                                 make_executor=make_executor, apply=True)


def test_select_final_skill_prefers_admitted_gepa_else_baseline(tmp_path):
    validation = bm.load_split("validation")
    with CandidateRegistry(tmp_path / "r.db") as registry:
        base = registry.register_baseline("github-code-review", b"baseline skill\n")
        cand = registry.propose("github-code-review", b"evolved skill\n",
                                parent_id=base.candidate_id)
        chosen, strategy = fe.select_final_skill(registry, "github-code-review", BASE)
        assert chosen.candidate_id == base.candidate_id   # candidate only proposed
        assert _admit_gepa(registry, validation, base, cand).admitted
        chosen, strategy = fe.select_final_skill(registry, "github-code-review", BASE)
        assert chosen.candidate_id == cand.candidate_id
        assert strategy.strategy_id == f"reviewer:{cand.candidate_id}"
        assert strategy.system_prompt == "evolved skill\n" and strategy.toolsets == ()


# ── Live full run (live-1) snapshot ────────────────────────────────────────

def test_live_run_snapshot_drives_the_report(small_report):
    run = small_report["live_run"]
    assert run["status"] == "supplied" and run["file"] == rp.DEFAULT_LIVE_RUN.name
    assert run["sealed_split_open_attempts"] == 0 and run["splits_used"] == ["train",
                                                                           "validation"]
    reg = small_report["safety"]["registry"]
    assert reg == run["registry"] and {c["source"] for c in reg["lineage"]} == {"baseline",
                                                                                "gepa"}
    kept = run["admission"]
    assert kept["status"] == "kept" and kept["decision"]["admitted"] is True
    assert any(c["registered_as"] == kept["candidate_id"] for c in run["gepa"]["candidates"])
    assert set(run["source_files"]) >= {"state.json", "registry.sqlite", "checkpoints.sqlite"}
    md = rp.format_markdown(small_report)
    for text in ("## Live full run (live-1)", "### Live policy comparison",
                 "### GEPA (gate 5 input)", "### Admission and checkpoint (gates 6 and 7)",
                 "### Spend by stage", "not evidence of real-world improvement"):
        assert text in md
    head = small_report["headline"]
    assert head[0].startswith("Live full run live-1")   # no final ledger in this report


def test_committed_headline_leads_with_the_sealed_result():
    data = json.loads((rp.DEFAULT_OUT / "report.json").read_text(encoding="utf-8"))
    assert data["headline"][0].startswith("Pre-registered sealed final evaluation")
    assert "STRICT micro-F1 +0.302 [0.157, 0.462]" in data["headline"][0]
    assert "mostly precision" in data["headline"][0]
    assert "used live Hermes" in data["notice"] and "real-world improvement" in data["notice"]


def test_snapshot_is_aggregate_only_and_rebuildable(tmp_path):
    """snapshot_live_run on a tiny fake run dir: read-only, no model outputs copied."""
    from evolution.adaptive.checkpoint import CheckpointStore
    from evolution.adaptive.core import ContextualBandit
    run = tmp_path / "run-x"
    (run / "gepa_log").mkdir(parents=True)
    with CandidateRegistry(run / "registry.sqlite") as reg:
        base = reg.register_baseline("code-review", b"base text\n")
        cand = reg.propose("code-review", b"evolved text\n", parent_id=base.candidate_id)
    with CheckpointStore(run / "checkpoints.sqlite") as store:
        store.save(ContextualBandit(("a",), epsilon=0.1, seed=1), skill_version="s",
                   label="pre")
    secret = "MODEL-OUTPUT-SHOULD-NOT-BE-COPIED"
    (run / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"stage": "compare", "cost_usd": 0.01, "latency_s": 2.0, "attempt": 0,
         "output": secret},
        {"stage": "compare", "cost_usd": None, "latency_s": 4.0, "attempt": 1,
         "error": secret}]) + "\n", encoding="utf-8")
    (run / "gepa_log/candidates.json").write_text(json.dumps(
        [{"self": "base text"}, {"self": "evolved text"}]), encoding="utf-8")
    (run / "gepa_log/run_log.json").write_text(json.dumps(
        [{"i": 0, "selected_program_candidate": 0, "new_program_idx": 1}]), encoding="utf-8")
    pol = {"validation": {k: {"mean": 0.5, "ci_low": 0.5, "ci_high": 0.5, "n": 1}
                          for k in ("utility", "quality", "cost_usd", "latency_s")}}
    validation = {"utility": 0.5, "n_tasks": 1, "critical_violation": False,
                  "synthetic": True}
    stages = {
        "canary": {"status": "done", "seconds": 1.0, "result": {
            "fresh_calls": 1, "replayed_calls": 0,
            "cases": [{"case_id": "c1", "f1": 1.0, "cost_usd": 0.01, "parse_ok": True}]}},
        "compare": {"status": "done", "seconds": 2.0, "result": {
            "fresh_calls": 2, "replayed_calls": 0, "report": {
                "best_fixed_baseline": "fixed:a", "executor_backend": "hermes",
                "executions": 2, "policies": {"fixed:a": pol}, "comparisons": [],
                "config": {"strategies": ["a"], "train_episodes": 1, "train_items": 1,
                           "validation_items": 1, "validation_learning": False,
                           "bootstrap": {"confidence": 0.95, "resamples": 10},
                           "seeds": [0]}}}},
        "gepa": {"status": "done", "seconds": 3.0, "result": {"metrics": {"seed": 0}}},
        "admission": {"status": "done", "seconds": 4.0, "result": {
            "status": "kept", "reason": "admitted and kept",
            "candidate_id": cand.candidate_id, "parent_id": base.candidate_id,
            "active_skill": cand.candidate_id, "registry_status": "proposed",
            "decision": {"admitted": True, "case_ids": ["c1"]},
            "pre_validation": validation, "post_validation": validation}}}
    (run / "state.json").write_text(json.dumps({
        "identity": {"model": "m"}, "baseline_id": base.candidate_id,
        "sealed_split_open_attempts": 0, "stages": {}}), encoding="utf-8")
    (run / "report.json").write_text(json.dumps({
        "stages": stages, "skill_source": "s", "splits_used": ["train"]}), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    snap = rp.snapshot_live_run(run)
    assert {p.name: p.read_bytes() for p in run.rglob("*") if p.is_file()} == before
    text = json.dumps(snap)
    assert secret not in text and "case_ids" not in snap["admission"]["decision"]
    st = snap["spend"]["by_stage"]["compare"]
    assert (st["calls"], st["errors"], st["unknown_cost_calls"], st["retries"]) == (2, 1, 1, 1)
    assert snap["gepa"]["candidates"][1] | {} == {
        "index": 1, "chars": 12, "sha256": snap["gepa"]["candidates"][1]["sha256"],
        "parent": 0, "merge": False, "seed": False, "registered_as": cand.candidate_id}
    out = tmp_path / "snap.json"
    out.write_text(json.dumps(snap), encoding="utf-8")
    assert rp.live_run_section(str(out))["run_id"] == "run-x"
    out.write_text(json.dumps({"schema": "other"}), encoding="utf-8")
    with pytest.raises(ValueError):
        rp.live_run_section(str(out))
