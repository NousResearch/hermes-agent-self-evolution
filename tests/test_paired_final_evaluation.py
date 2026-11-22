"""Offline tests for the pre-registered paired final evaluation (owner option 1).

Fake reviewers only; no Hermes call. Nothing here prints or asserts sealed
case content: sealed cases are only passed to fake reviewers and scored.
"""
import itertools
import json
from pathlib import Path
import random
import shutil

import pytest

from evolution.adaptive import benchmark as bm
from evolution.adaptive import evaluation as ev
from evolution.adaptive import final_evaluation as fe
from evolution.adaptive import lenient
from evolution.adaptive import report as rp
from evolution.adaptive import verifier
from evolution.adaptive.executor import FakeExecutor, FakeResponse, Strategy
from tests.test_benchmark_integration import _scan

REPO = Path(__file__).resolve().parents[1]
SMALL = dict(seeds=3, train_episodes=40, resamples=50, verifier_bootstrap=50)
VERSIONS = {"baseline": "github-code-review@11ad06e6d49b",
            "candidate": "github-code-review@28dcec5ab528"}


@pytest.fixture(scope="module")
def open_cases():
    return [c for cases in ev.open_cases().values() for c in cases]


def _reviews(case, seed):
    """Varied reviews: exact, noisy, wrong category, shifted, duplicate, invalid."""
    rng = random.Random(f"{seed}:{case.case_id}")
    truth = ev.ground_truth_findings(case)
    cats = list(bm.DEFECT_CATEGORIES)
    wrong = [{**f, "category": rng.choice([c for c in cats if c != f["category"]])}
             for f in truth]
    shifted = [{**f, "line_start": f["line_start"] + 3, "line_end": f["line_end"] + 3}
               for f in truth]
    return [truth, rp.noisy_reference_review(case, seed), wrong, shifted,
            truth + truth, wrong + truth, [{"line_start": 0, "category": "x"}, "junk"], []]


# ── Lenient scoring ────────────────────────────────────────────────────────

def test_lenient_without_alternates_equals_the_verifier(open_cases):
    for case, seed in itertools.product(open_cases, range(3)):
        for review in _reviews(case, seed):
            assert lenient.score_case(case, review, {}).to_dict() == \
                verifier.score_case(case, review).to_dict()


def test_lenient_never_scores_below_strict(open_cases):
    table = lenient.load_alternates()
    for case, seed in itertools.product(open_cases, range(3)):
        for review in _reviews(case, seed):
            s, l_ = verifier.score_case(case, review), lenient.score_case(case, review, table)
            assert l_.true_positives >= s.true_positives
            assert l_.true_positives + l_.false_positives == \
                s.true_positives + s.false_positives


def _toy_case():
    return verifier.parse_case({
        "case_id": "toy-001", "split": "train", "language": "python",
        "code": "a = 1\nb = 2\nc = 3\nd = 4\n",
        "context": {"difficulty": "easy", "domain": "toy"},
        "defects": [{"defect_id": "toy-001-d1", "category": "logic_error", "line_start": 2,
                     "line_end": 2, "severity": "low", "description": ""},
                    {"defect_id": "toy-001-d2", "category": "api_misuse", "line_start": 4,
                     "line_end": 4, "severity": "low", "description": ""}]})


def test_lenient_accepts_only_the_listed_alternates_and_stays_one_to_one():
    case = _toy_case()
    table = {"toy-001-d1": frozenset({"API misuse"})}       # normalised like the verifier
    alt = [{"line_start": 2, "category": "api_misuse"}]
    assert verifier.score_case(case, alt).true_positives == 0
    assert lenient.score_case(case, alt, table).matches == ((0, "toy-001-d1"),)
    other = [{"line_start": 2, "category": "security"}]
    assert lenient.score_case(case, other, table).true_positives == 0
    # the alternate does not leak to d2, and one finding matches one defect
    assert lenient.score_case(case, [{"line_start": 4, "category": "logic_error"}],
                              table).true_positives == 0
    twice = alt + alt
    res = lenient.score_case(case, twice, table)
    assert (res.true_positives, res.false_positives, res.duplicate_findings) == (1, 1, 1)


def test_alternates_table_is_pinned_and_consistent_with_open_labels(open_cases, tmp_path):
    table = lenient.load_alternates()
    prov = lenient.alternates_provenance()
    assert prov["n_defects"] == len(table) == 15
    assert prov["benchmark"]["manifest_sha256"] == bm.PINNED_MANIFEST_SHA256["v1"]
    assert all(c in bm.DEFECT_CATEGORIES for cats in table.values() for c in cats)
    labels = {d.defect_id: d.category for c in open_cases for d in c.defects}
    open_ids = {c.case_id for c in open_cases}
    for defect_id, cats in table.items():
        if defect_id.rsplit("-d", 1)[0] in open_ids:
            assert labels[defect_id] not in cats
    tampered = tmp_path / "alt.json"
    data = json.loads(lenient.ALTERNATES_PATH.read_text(encoding="utf-8"))
    data["alternates"]["pyrev-015-d1"].append("security")
    tampered.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(lenient.AlternatesError, match="pinned"):
        lenient.load_alternates(tampered)
    shutil.copy(lenient.ALTERNATES_PATH, tmp_path / "copy.json")
    assert lenient.load_alternates(tmp_path / "copy.json") == table


def test_paired_bootstrap(open_cases):
    cases = open_cases[:10]
    empty = [verifier.score_case(c, []) for c in cases]
    truth = [verifier.score_case(c, ev.ground_truth_findings(c)) for c in cases]
    same = lenient.paired_bootstrap(empty, empty, seed=0, n_bootstrap=200)
    assert all(v == {"point": 0.0, "low": 0.0, "high": 0.0} for v in same.values())
    better = lenient.paired_bootstrap(empty, truth, seed=0, n_bootstrap=200)
    assert better["micro_f1"]["point"] > 0 and better["micro_f1"]["low"] > 0
    assert better == lenient.paired_bootstrap(empty, truth, seed=0, n_bootstrap=200)
    assert lenient.paired_bootstrap(truth, empty, seed=0, n_bootstrap=200)["micro_f1"][
        "point"] == -better["micro_f1"]["point"]
    with pytest.raises(ValueError, match="same nonempty"):
        lenient.paired_bootstrap(empty, truth[:-1], seed=0, n_bootstrap=10)


# ── The paired one-time run (fake reviewers) ───────────────────────────────

def _recorder(name, log, findings=()):
    def review(view):
        assert set(view) == {"case_id", "language", "code", "context"}
        log.append((name, view["case_id"]))
        return list(findings)
    return review


def _run(tmp_path, reviewers, **kw):
    return fe.run_paired_final_evaluation(
        reviewers, skill_versions=VERSIONS, reviewer_id="fake:paired", synthetic=True,
        confirm=fe.CONFIRM_PHRASE, ledger_path=tmp_path / "final.json", **kw)


def test_paired_run_is_recorded_once_with_the_preregistered_plan(tmp_path):
    log = []
    record = _run(tmp_path, {"baseline": _recorder("baseline", log),
                             "candidate": _recorder("candidate", log)})
    assert record["status"] == "completed" and record["schema"] == fe.PAIRED_LEDGER_SCHEMA
    assert record["n_cases"] == 10 and record["skill_versions"] == VERSIONS
    assert record["analysis_plan"] == fe.ANALYSIS_PLAN
    assert record["analysis_plan_sha256"] == fe.ANALYSIS_PLAN_SHA256
    assert (record["primary"], record["secondary"]) == ("strict.micro_f1", "lenient.micro_f1")
    assert record["alternates"]["sha256"] == lenient.ALTERNATES_SHA256
    assert record["verifier"]["verifier_sha256"] == verifier.verifier_fingerprint()
    # every case reviewed by both skills, interleaved, order alternating per position
    ids = sorted({cid for _, cid in log})
    assert len(log) == 20 and len(ids) == 10
    for i, cid in enumerate(ids):
        pair = log[2 * i: 2 * i + 2]
        assert {p[1] for p in pair} == {cid}
        assert pair[0][0] == ("baseline" if i % 2 == 0 else "candidate")
    assert record["first_call_by_position"] == [
        "baseline" if i % 2 == 0 else "candidate" for i in range(10)]
    for name in ("strict", "lenient"):
        assert record["paired"][name]["micro_f1"]["point"] == 0.0
        assert [c["case_id"] for c in record["scores"][name]["candidate"]["cases"]] == ids
    on_disk = json.loads((tmp_path / "final.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(record))
    with pytest.raises(fe.FinalEvaluationRefused, match="already completed"):
        _run(tmp_path, {"baseline": _recorder("b", []), "candidate": _recorder("c", [])})
    with pytest.raises(fe.FinalEvaluationRefused, match="already completed"):
        fe.run_final_evaluation(_recorder("x", []), reviewer_id="single", synthetic=True,
                                confirm=fe.CONFIRM_PHRASE, ledger_path=tmp_path / "final.json")


def test_paired_differences_follow_the_reviews(tmp_path):
    flood = [{"line_start": n, "category": c} for n in range(1, 6)
             for c in bm.DEFECT_CATEGORIES]
    record = _run(tmp_path, {"baseline": _recorder("b", []),
                             "candidate": _recorder("c", [], findings=flood)})
    strict, loose = record["scores"]["strict"], record["scores"]["lenient"]
    assert strict["baseline"]["totals"]["true_positives"] == 0
    assert loose["candidate"]["totals"]["true_positives"] >= \
        strict["candidate"]["totals"]["true_positives"]
    d = record["paired"]["strict"]["micro_f1"]
    assert d["point"] == pytest.approx(strict["candidate"]["micro"]["f1"]["point"]
                                       - strict["baseline"]["micro"]["f1"]["point"])
    assert d["low"] <= d["point"] <= d["high"]


@pytest.mark.parametrize("change, error", [
    ({"confirm": "yes"}, fe.FinalEvaluationRefused),
    ({"skill_versions": {"baseline": "same", "candidate": "same"}}, ValueError),
    ({"skill_versions": {"baseline": "a"}}, ValueError),
    ({"reviewers": {"baseline": lambda v: []}}, ValueError),
    ({"reviewer_id": " "}, ValueError),
])
def test_bad_arguments_are_refused_before_the_ledger_exists(tmp_path, change, error):
    args = dict(reviewers={"baseline": lambda v: [], "candidate": lambda v: []},
                skill_versions=VERSIONS, reviewer_id="r", synthetic=True,
                confirm=fe.CONFIRM_PHRASE, ledger_path=tmp_path / "final.json")
    with pytest.raises(error):
        fe.run_paired_final_evaluation(**{**args, **change})
    assert not (tmp_path / "final.json").exists()


def test_modified_alternates_table_is_refused_before_the_ledger_exists(tmp_path, monkeypatch):
    bad = tmp_path / "alt.json"
    bad.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(lenient, "ALTERNATES_PATH", bad)
    monkeypatch.setattr(lenient.load_alternates, "__defaults__",
                        (bad, lenient.ALTERNATES_SHA256))
    with pytest.raises(lenient.AlternatesError):
        _run(tmp_path, {"baseline": lambda v: [], "candidate": lambda v: []})
    assert not (tmp_path / "final.json").exists()


def test_paired_run_refused_inside_optimisation_phase(tmp_path):
    with bm.optimisation_phase(), pytest.raises(fe.FinalEvaluationRefused):
        _run(tmp_path, {"baseline": lambda v: [], "candidate": lambda v: []})
    assert not (tmp_path / "final.json").exists()


def test_failed_paired_run_still_counts_as_the_one_run(tmp_path):
    def broken(view):
        raise RuntimeError("reviewer crashed")
    with pytest.raises(RuntimeError):
        _run(tmp_path, {"baseline": lambda v: [], "candidate": broken})
    record = fe.read_ledger(tmp_path / "final.json")
    assert record["status"] == "failed" and "reviewer crashed" in record["error"]
    with pytest.raises(fe.FinalEvaluationRefused, match="already failed"):
        _run(tmp_path, {"baseline": lambda v: [], "candidate": lambda v: []})


def test_shared_cost_meter_caps_both_skills_together(tmp_path):
    meter = fe.CostMeter(0.5)
    exec_b = FakeExecutor({}, default=FakeResponse(output="[]", cost_usd=0.1))
    exec_c = FakeExecutor({}, default=FakeResponse(output="[]", cost_usd=0.1))
    rb = fe.ExecutorReviewer(exec_b, Strategy("review:b"), meter=meter)
    rc = fe.ExecutorReviewer(exec_c, Strategy("review:c"), meter=meter)
    record = _run(tmp_path, {"baseline": rb, "candidate": rc})
    sb, sc = (record["reviewer_summaries"][a] for a in ("baseline", "candidate"))
    assert sb["calls"] + sc["calls"] == 5 and sb["skipped"] + sc["skipped"] == 15
    assert sb["meter_total_cost_usd"] == sc["meter_total_cost_usd"] == pytest.approx(0.5)
    assert "cost cap" in sb["stopped"] and sb["stopped"] == sc["stopped"]
    with pytest.raises(ValueError, match="exactly one"):
        fe.ExecutorReviewer(exec_b, Strategy("s"), max_total_cost_usd=1.0, meter=meter)


def test_report_renders_the_paired_ledger_without_case_rows(tmp_path):
    _run(tmp_path, {"baseline": _recorder("b", []), "candidate": _recorder("c", [])})
    section = rp.final_evaluation_section(tmp_path / "final.json")
    assert section["design"] == "paired" and "cases" not in section["scores"]["strict"][
        "baseline"]
    md = rp.format_markdown(rp.generate(final_ledger_path=tmp_path / "final.json", **SMALL))
    for text in ("STRICT (primary)", "LENIENT (secondary)", "Paired difference",
                 VERSIONS["baseline"], VERSIONS["candidate"], "not evidence of a real Hermes"):
        assert text in md


# ── Pre-registration and sealed access ─────────────────────────────────────

def test_plan_is_preregistered_in_the_sprint_doc():
    doc = (REPO / "docs/sprints/KAFO-EVO-001.md").read_text(encoding="utf-8")
    section = doc.split("### Pre-registered analysis plan", 1)[1].split("\n## ", 1)[0]
    for text in (fe.ANALYSIS_PLAN["id"], "STRICT", "LENIENT", "Primary metric",
                 "Secondary metric", "10000 resamples", "random.Random(0)", "95% paired",
                 "category_alternates_v1.json", "USD 2", *VERSIONS.values()):
        assert text in section, text
    assert (fe.ANALYSIS_PLAN["seed"], fe.ANALYSIS_PLAN["n_bootstrap"],
            fe.ANALYSIS_PLAN["confidence"]) == (0, 10000, 0.95)


def test_sealed_test_cases_need_the_final_flag_and_no_phase():
    with pytest.raises(bm.SealedSplitError):
        ev.sealed_test_cases()
    with bm.optimisation_phase(), pytest.raises(bm.SealedSplitError):
        ev.sealed_test_cases(final_evaluation=True)


def test_only_the_paired_step_may_call_sealed_test_cases(tmp_path):
    rel = Path("evolution/adaptive/final_evaluation.py")
    assert _scan(REPO / rel, rel) == []
    rogue = tmp_path / "rogue.py"
    rogue.write_text("from . import evaluation\n"
                     "def run_paired_final_evaluation():\n"
                     "    return evaluation.sealed_test_cases(final_evaluation=True)\n"
                     "def helper():\n"
                     "    return evaluation.sealed_test_cases(final_evaluation=True)\n",
                     encoding="utf-8")
    assert len(_scan(rogue, Path("evolution/adaptive/rogue.py"))) == 2
    renamed = _scan(rogue, rel)
    assert len(renamed) == 1 and ":5:" in renamed[0] and "sealed_test_cases" in renamed[0]


def test_committed_ledger_keeps_no_free_text_from_the_reviewer(tmp_path):
    secret = "ECHOED-SEALED-TEXT"
    exec_b = FakeExecutor({}, default=FakeResponse(output="not json", cost_usd=0.01,
                                                   error=f"agent said {secret}"))
    exec_c = FakeExecutor({}, default=FakeResponse(output="[]", cost_usd=0.01))
    original = exec_c.execute

    def flaky(task, strategy):
        if len(exec_c.calls) == 0:
            exec_c.calls.append((task.task_id, strategy.strategy_id))
            raise RuntimeError(f"crash quoting {secret}")
        return original(task, strategy)
    exec_c.execute = flaky
    meter = fe.CostMeter(2.0)
    _run(tmp_path, {"baseline": fe.ExecutorReviewer(exec_b, Strategy("review:b"), meter=meter),
                    "candidate": fe.ExecutorReviewer(exec_c, Strategy("review:c"), meter=meter)})
    text = (tmp_path / "final.json").read_text(encoding="utf-8")
    assert secret not in text
    record = json.loads(text)
    rows_b = record["reviewer_summaries"]["baseline"]["cases"]
    rows_c = record["reviewer_summaries"]["candidate"]["cases"]
    assert all(r["error"] is True and r["parse_ok"] is False for r in rows_b)
    assert [r.get("exception_type") for r in rows_c].count("RuntimeError") == 1


def test_late_scoring_error_is_recorded_as_failed(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise ArithmeticError("bootstrap broke")
    monkeypatch.setattr(lenient, "paired_bootstrap", boom)
    with pytest.raises(ArithmeticError):
        _run(tmp_path, {"baseline": lambda v: [], "candidate": lambda v: []})
    record = fe.read_ledger(tmp_path / "final.json")
    assert record["status"] == "failed" and "scores" not in record
