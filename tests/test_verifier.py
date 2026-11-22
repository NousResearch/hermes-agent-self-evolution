"""Offline tests for the independent verifier; small inline fixtures only."""
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from evolution.adaptive import verifier as v
from evolution.adaptive.verifier import (
    MatchConfig, aggregate, evaluate, load_cases, load_reviews, parse_case,
    quality, score_case,
)

VERIFIER_PATH = Path(v.__file__)
CODE_20 = "\n".join(f"line_{i} = {i}" for i in range(1, 21)) + "\n"


def make_case(case_id="c1", defects=(), split="train", code=CODE_20):
    return {
        "case_id": case_id, "split": split, "language": "python", "code": code,
        "context": {"difficulty": "easy", "domain": "web"},
        "defects": [
            {"defect_id": did, "category": cat, "line_start": s, "line_end": e,
             "severity": "high", "description": f"defect {did}"}
            for did, cat, s, e in defects
        ],
    }


THREE = (("d1", "security", 3, 4), ("d2", "logic", 10, 10), ("d3", "resource_leak", 15, 17))


@pytest.fixture
def case3():
    return parse_case(make_case("c3", THREE))


def finding(start, end=None, category="security", description=None):
    f = {"line_start": start, "category": category}
    if end is not None:
        f["line_end"] = end
    if description is not None:
        f["description"] = description
    return f


# ── Per-case scoring ───────────────────────────────────────────────────────

def test_perfect_review(case3):
    findings = [finding(3, 4), finding(10, category="logic"),
                finding(15, 17, category="resource_leak")]
    r = score_case(case3, findings)
    assert (r.true_positives, r.false_positives, r.false_negatives) == (3, 0, 0)
    assert r.precision == r.recall == r.f1 == r.quality == 1.0
    assert r.matches == ((0, "d1"), (1, "d2"), (2, "d3"))
    assert r.missed_defects == () and r.false_positive_findings == ()


def test_empty_review_on_defective_case(case3):
    r = score_case(case3, [])
    assert (r.true_positives, r.false_positives, r.false_negatives) == (0, 0, 3)
    assert r.precision == r.recall == r.f1 == 0.0
    assert r.missed_defects == ("d1", "d2", "d3")


def test_clean_case_silent_is_perfect_and_false_alarm_is_zero():
    clean = parse_case(make_case("clean"))
    silent = score_case(clean, [])
    assert silent.precision == silent.recall == silent.f1 == 1.0
    noisy = score_case(clean, [finding(5)])
    assert (noisy.true_positives, noisy.false_positives, noisy.false_negatives) == (0, 1, 0)
    assert noisy.precision == noisy.recall == noisy.f1 == 0.0


def test_partial_review(case3):
    findings = [finding(3), finding(10, category="logic"), finding(19, category="style")]
    r = score_case(case3, findings)
    assert (r.true_positives, r.false_positives, r.false_negatives) == (2, 1, 1)
    assert r.precision == pytest.approx(2 / 3)
    assert r.recall == pytest.approx(2 / 3)
    assert r.f1 == pytest.approx(2 / 3)
    assert r.missed_defects == ("d3",)
    assert r.false_positive_findings == (2,)
    assert r.duplicate_findings == 0


def test_duplicate_findings_count_as_false_positives(case3):
    findings = [finding(3), finding(4), finding(3, 4, description="same bug again")]
    r = score_case(case3, findings)
    assert (r.true_positives, r.false_positives, r.false_negatives) == (1, 2, 2)
    assert r.duplicate_findings == 2
    assert r.matches == ((0, "d1"),)
    assert r.precision == pytest.approx(1 / 3)
    assert r.recall == pytest.approx(1 / 3)


def test_category_must_match(case3):
    r = score_case(case3, [finding(3, 4, category="logic")])
    assert r.true_positives == 0 and r.false_positives == 1


def test_category_normalisation_and_aliases(case3):
    assert score_case(case3, [finding(15, category=" Resource-Leak ")]).true_positives == 1
    cfg = MatchConfig(category_aliases=(("injection", "security"),))
    assert score_case(case3, [finding(3, category="Injection")], cfg).true_positives == 1
    assert score_case(case3, [finding(3, category="Injection")]).true_positives == 0


def test_line_tolerance_is_one_line_by_default(case3):
    logic = dict(category="logic")
    assert score_case(case3, [finding(11, **logic)]).true_positives == 1   # off by one
    assert score_case(case3, [finding(12, **logic)]).true_positives == 0   # off by two
    strict = MatchConfig(line_tolerance=0)
    assert score_case(case3, [finding(11, **logic)], strict).true_positives == 0


def test_whole_file_finding_cannot_match_small_defect(case3):
    """Blanket line ranges must not earn true positives (reward hacking)."""
    blanket = [finding(1, 20, category=c) for c in ("security", "logic", "resource_leak")]
    r = score_case(case3, blanket)
    assert r.true_positives == 0 and r.f1 == 0.0
    # A tight range around the defect still matches.
    assert score_case(case3, [finding(9, 11, category="logic")]).true_positives == 1


def test_maximum_matching_is_order_invariant():
    """`wide` fits both defects, `narrow` only `a`. Greedy first-come matching
    would give wide->a and leave narrow unmatched; augmenting paths do not."""
    case = parse_case(make_case("mm", (("a", "logic", 5, 6), ("b", "logic", 8, 9))))
    wide, narrow = finding(5, 9, category="logic"), finding(5, category="logic")
    for findings, expected in (([wide, narrow], ((0, "b"), (1, "a"))),
                               ([narrow, wide], ((0, "a"), (1, "b")))):
        r = score_case(case, findings)
        assert r.true_positives == 2 and r.false_positives == 0
        assert r.matches == expected


def test_description_does_not_affect_matching(case3):
    a = score_case(case3, [finding(3, description="SQL injection via f-string")])
    b = score_case(case3, [finding(3, description="totally unrelated words")])
    assert a.matches == b.matches == ((0, "d1"),)


@pytest.mark.parametrize("bad", [
    "not a dict", {"category": "security"}, {"line_start": 3},
    {"line_start": True, "category": "security"},
    {"line_start": 3.0, "category": "security"},
    {"line_start": 5, "line_end": 4, "category": "security"},
    {"line_start": 0, "category": "security"},
    {"line_start": 3, "category": "   "},
    {"line_start": 3, "category": "security", "description": 7},
])
def test_malformed_findings_are_false_positives_not_errors(case3, bad):
    r = score_case(case3, [bad, finding(3)])
    assert r.invalid_findings == 1
    assert (r.true_positives, r.false_positives) == (1, 1)
    assert r.false_positive_findings == (0,)


@pytest.mark.parametrize("bad", ["3:security", {"line_start": 3}, None, 42])
def test_non_list_review_rejected(case3, bad):
    with pytest.raises(TypeError):
        score_case(case3, bad)


def test_quality_is_in_unit_interval_and_feeds_reward(case3):
    from evolution.adaptive.core import Reward, score_outcome
    for findings in ([], [finding(3)], [finding(1, 20)] * 5):
        q = quality(case3, findings)
        assert 0.0 <= q <= 1.0
        assert score_outcome(Reward(q, 0.5, 0.5)) <= 1.0


# ── Case validation ────────────────────────────────────────────────────────

@pytest.mark.parametrize("mutate, message", [
    (lambda c: c.pop("case_id"), "case_id"),
    (lambda c: c["context"].pop("domain"), "context.domain"),
    (lambda c: c.update(defects="none"), "defects must be a list"),
    (lambda c: c["defects"][0].update(line_start=0), "line_start"),
    (lambda c: c["defects"][0].update(line_end=21), "<= 20"),
    (lambda c: c["defects"][0].update(line_start=5, line_end=4), "line_start"),
    (lambda c: c["defects"][0].update(line_start=True), "ints"),
    (lambda c: c["defects"].append(dict(c["defects"][0])), "duplicate defect_id"),
    (lambda c: c["defects"][0].pop("severity"), "severity"),
])
def test_malformed_cases_raise(mutate, message):
    raw = make_case("bad", THREE)
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        parse_case(raw)


def test_extra_case_fields_are_ignored():
    raw = make_case("x", THREE)
    raw["source"] = "synthetic"
    assert parse_case(raw).defects[0].defect_id == "d1"


@pytest.mark.parametrize("kwargs", [
    {"line_tolerance": -1}, {"line_tolerance": True}, {"min_overlap_ratio": 1.5},
    {"min_overlap_ratio": float("nan")}, {"category_aliases": [("a", "b")]},
    {"category_aliases": (("a", ""),)},
])
def test_invalid_match_config_rejected(kwargs):
    with pytest.raises(ValueError):
        MatchConfig(**kwargs)


# ── Aggregation, bootstrap and reproducibility ─────────────────────────────

def _suite():
    cases = [
        parse_case(make_case("a", (("d1", "security", 3, 3), ("d2", "logic", 8, 8)))),
        parse_case(make_case("b", (("d1", "logic", 5, 6),))),
        parse_case(make_case("c")),
        parse_case(make_case("d", (("d1", "security", 2, 2), ("d2", "security", 12, 12),
                                   ("d3", "logic", 18, 18)))),
    ]
    reviews = {
        "a": [finding(3), finding(8, category="logic")],         # TP2
        "b": [finding(14, category="logic")],                    # FP1 FN1
        "c": [],                                                 # clean, silent
        "d": [finding(2), finding(2), finding(18, category="logic")],  # TP2 FP1 FN1
    }
    return cases, reviews


def test_micro_and_macro_point_estimates():
    cases, reviews = _suite()
    report = evaluate(cases, reviews, n_bootstrap=50)
    # Totals: TP=4, FP=2, FN=2.
    assert report.micro.precision.point == pytest.approx(4 / 6)
    assert report.micro.recall.point == pytest.approx(4 / 6)
    assert report.micro.f1.point == pytest.approx(8 / 12)
    # Per case P: 1, 0, 1, 2/3; R: 1, 0, 1, 2/3; F1: 1, 0, 1, 2/3.
    assert report.macro.precision.point == pytest.approx((1 + 0 + 1 + 2 / 3) / 4)
    assert report.macro.recall.point == pytest.approx((1 + 0 + 1 + 2 / 3) / 4)
    assert report.macro.f1.point == pytest.approx((1 + 0 + 1 + 2 / 3) / 4)


def test_intervals_are_ordered_and_bounded():
    cases, reviews = _suite()
    report = evaluate(cases, reviews, n_bootstrap=200, seed=11)
    for scores in (report.micro, report.macro):
        for iv in (scores.precision, scores.recall, scores.f1):
            assert 0.0 <= iv.low <= iv.high <= 1.0
    assert report.macro.f1.low < report.macro.f1.high  # the suite has spread


def test_same_seed_reproduces_report_exactly():
    cases, reviews = _suite()
    a = evaluate(cases, reviews, seed=123, n_bootstrap=300).to_dict()
    b = evaluate(cases, reviews, seed=123, n_bootstrap=300).to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_report_independent_of_case_and_review_order():
    cases, reviews = _suite()
    shuffled = dict(reversed(list(reviews.items())))
    a = evaluate(cases, reviews, seed=5, n_bootstrap=100).to_dict()
    b = evaluate(list(reversed(cases)), shuffled, seed=5, n_bootstrap=100).to_dict()
    assert a == b


def test_different_seed_changes_intervals_not_points():
    cases, reviews = _suite()
    a = evaluate(cases, reviews, seed=1, n_bootstrap=100)
    b = evaluate(cases, reviews, seed=2, n_bootstrap=100)
    assert a.micro.f1.point == b.micro.f1.point
    assert a.to_dict()["micro"] != b.to_dict()["micro"]


def test_single_case_interval_is_degenerate():
    cases, reviews = _suite()
    report = evaluate(cases[:1], {"a": reviews["a"]}, n_bootstrap=20)
    assert report.micro.f1 == v.Interval(1.0, 1.0, 1.0)


@pytest.mark.parametrize("kwargs", [
    {"seed": 1.5}, {"seed": True}, {"n_bootstrap": 0}, {"confidence": 1.0},
    {"confidence": 0.0},
])
def test_invalid_bootstrap_settings_rejected(kwargs):
    cases, reviews = _suite()
    with pytest.raises(ValueError):
        evaluate(cases, reviews, **kwargs)


def test_aggregate_requires_cases():
    with pytest.raises(ValueError):
        aggregate([])


def test_missing_review_scored_empty_and_reported():
    cases, reviews = _suite()
    del reviews["a"]
    report = evaluate(cases, reviews, n_bootstrap=10)
    assert report.missing_reviews == ("a",)
    a = next(c for c in report.cases if c.case_id == "a")
    assert a.false_negatives == 2 and a.f1 == 0.0


def test_unknown_or_duplicate_ids_rejected():
    cases, reviews = _suite()
    with pytest.raises(ValueError, match="unknown"):
        evaluate(cases, {**reviews, "zzz": []})
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(cases + cases[:1], reviews)


# ── Independence and integrity ─────────────────────────────────────────────

STDLIB_ALLOWED = {"__future__", "argparse", "dataclasses", "hashlib", "json",
                  "math", "pathlib", "random", "sys", "typing"}


def test_verifier_imports_only_stdlib_and_no_agent_code():
    tree = ast.parse(VERIFIER_PATH.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports would reach agent code"
            imported.add(node.module.split(".")[0])
    assert imported <= STDLIB_ALLOWED, imported - STDLIB_ALLOWED
    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not calls & {"eval", "exec", "compile", "__import__"}


def test_verifier_loads_standalone_without_package():
    spec = importlib.util.spec_from_file_location("standalone_verifier", VERIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["standalone_verifier"] = module
    try:
        spec.loader.exec_module(module)
        case = module.parse_case(make_case("s", THREE))
        assert module.score_case(case, [finding(3)]).true_positives == 1
    finally:
        del sys.modules["standalone_verifier"]


def test_report_carries_fingerprint_and_rule_versions():
    cases, reviews = _suite()
    d = evaluate(cases, reviews, n_bootstrap=10).to_dict()
    assert d["verifier_sha256"] == hashlib.sha256(VERIFIER_PATH.read_bytes()).hexdigest()
    assert d["verifier_version"] == v.VERIFIER_VERSION
    assert d["match_rule"] == v.MATCH_RULE
    assert d["config"] == {"line_tolerance": 1, "min_overlap_ratio": 0.5,
                           "category_aliases": []}


def test_results_are_immutable(case3):
    r = score_case(case3, [finding(3)])
    with pytest.raises(AttributeError):
        r.f1 = 1.0
    with pytest.raises(AttributeError):
        MatchConfig().min_overlap_ratio = 0.0


# ── JSONL loading and CLI ──────────────────────────────────────────────────

def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + "\n", encoding="utf-8")


def test_jsonl_roundtrip_and_cli_is_read_only(tmp_path, capsys):
    cases_path, reviews_path = tmp_path / "cases.jsonl", tmp_path / "reviews.jsonl"
    _write_jsonl(cases_path, [make_case("a", THREE, split="train"),
                              make_case("b", THREE, split="validation")])
    _write_jsonl(reviews_path, [{"case_id": "a", "findings": [finding(3)]},
                                {"case_id": "b", "findings": []}])
    before = {p: p.read_bytes() for p in (cases_path, reviews_path)}
    assert [c.case_id for c in load_cases(cases_path)] == ["a", "b"]
    assert load_reviews(reviews_path)["a"] == [finding(3)]

    assert v.main(["--cases", str(cases_path), "--reviews", str(reviews_path),
                   "--bootstrap", "20", "--split", "train"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_cases"] == 1 and out["cases"][0]["case_id"] == "a"
    assert out["cases"][0]["true_positives"] == 1
    assert {p: p.read_bytes() for p in before} == before


def test_jsonl_errors_report_line_numbers(tmp_path):
    p = tmp_path / "cases.jsonl"
    p.write_text(json.dumps(make_case("a")) + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        load_cases(p)
    r = tmp_path / "reviews.jsonl"
    _write_jsonl(r, [{"case_id": "a", "findings": []}, {"case_id": "a", "findings": []}])
    with pytest.raises(ValueError, match="duplicate review"):
        load_reviews(r)
