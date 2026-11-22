"""Gate 3 verifier over the gate 2 benchmark, and sealed-split separation.

Offline: no Hermes, no LLM, no network.
"""
import ast
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from evolution.adaptive import benchmark as bm
from evolution.adaptive import cli, evaluation as ev, verifier as v
from evolution.adaptive.core import ContextualBandit, ExperienceStore, Reward, choose_strategy
from evolution.adaptive.executor import FakeExecutor, FakeResponse, Strategy, Task, loop_executor

REPO = Path(__file__).resolve().parents[1]
SEALED_FILE = bm.BENCHMARK_ROOT / "v1" / "sealed" / "test.jsonl"


@pytest.fixture(scope="module")
def open_benchmark():
    return bm.load_benchmark()


@pytest.fixture(scope="module")
def open_verifier_cases():
    return ev.open_cases()


# ── (a) perfect review, (b) empty review ─────────────────────────────────────

@pytest.mark.parametrize("split", bm.OPEN_SPLITS)
def test_ground_truth_review_scores_one(split, open_benchmark):
    cases = open_benchmark[split]
    reviews = {c.case_id: ev.ground_truth_findings(c) for c in cases}
    report = ev.evaluate_open(reviews, splits=(split,), n_bootstrap=50)[split]
    assert len(report.cases) == len(cases) and report.missing_reviews == ()
    assert all(r.quality == 1.0 for r in report.cases)
    for scores in (report.micro, report.macro):
        assert scores.precision.point == scores.recall.point == scores.f1.point == 1.0
    matched = {defect_id for r in report.cases for _, defect_id in r.matches}
    assert matched == {d.defect_id for c in cases for d in c.defects}


@pytest.mark.parametrize("split", bm.OPEN_SPLITS)
def test_empty_review_scores_zero(split, open_benchmark):
    cases = open_benchmark[split]
    for reviews in ({c.case_id: [] for c in cases}, {}):
        report = ev.evaluate_open(reviews, splits=(split,), n_bootstrap=50)[split]
        micro = report.micro
        assert micro.precision.point == micro.recall.point == micro.f1.point == 0.0
        by_id = {r.case_id: r for r in report.cases}
        for case in cases:
            # Silence is correct on a clean control (verifier convention), 0 elsewhere.
            assert by_id[case.case_id].quality == (1.0 if case.is_clean else 0.0)
        clean_share = sum(c.is_clean for c in cases) / len(cases)
        assert report.macro.f1.point == pytest.approx(clean_share)


def test_wrong_categories_score_zero(open_benchmark):
    other = {c: bm.DEFECT_CATEGORIES[(i + 1) % len(bm.DEFECT_CATEGORIES)]
             for i, c in enumerate(bm.DEFECT_CATEGORIES)}
    cases = open_benchmark["train"]
    reviews = {c.case_id: [dict(f, category=other[f["category"]])
                           for f in ev.ground_truth_findings(c)] for c in cases}
    report = ev.evaluate_open(reviews, splits=("train",), n_bootstrap=10)["train"]
    assert sum(r.true_positives for r in report.cases) == 0


def test_findings_on_clean_controls_are_false_positives(open_benchmark):
    clean = [c for c in open_benchmark["train"] if c.is_clean]
    reviews = {c.case_id: [{"line_start": 1, "line_end": 1, "category": "security"}]
               for c in clean}
    report = ev.evaluate_open(reviews, splits=("train",), n_bootstrap=10)["train"]
    by_id = {r.case_id: r for r in report.cases}
    assert all(by_id[c.case_id].false_positives == 1 and by_id[c.case_id].quality == 0.0
               for c in clean)


# ── (c) shared conventions ───────────────────────────────────────────────────

def test_cases_convert_without_loss(open_benchmark, open_verifier_cases):
    for split, cases in open_benchmark.items():
        converted = open_verifier_cases[split]
        assert [c.case_id for c in converted] == [c.case_id for c in cases]
        for ours, theirs in zip(cases, converted):
            assert theirs.split == ours.split == split
            assert (theirs.code, theirs.difficulty, theirs.domain) == (
                ours.code, ours.context.difficulty, ours.context.domain)
            assert [tuple(vars(d).values()) for d in theirs.defects] == \
                   [tuple(vars(d).values()) for d in ours.defects]


@pytest.mark.parametrize("code", ["", "a", "a\n", "a\nb", "a\n\n", "a\x0cb",
                                  "a\rb", "a\r\nb", "a b\n", "a\x1cb\x1dc"])
def test_line_counting_is_identical(code):
    assert v._line_count(code) == bm.code_line_count(code)


def test_line_counting_identical_on_every_open_case(open_benchmark):
    for cases in open_benchmark.values():
        for case in cases:
            assert v._line_count(case.code) == bm.code_line_count(case.code)


def test_benchmark_categories_are_canonical_for_the_verifier():
    config = v.MatchConfig()
    for category in bm.DEFECT_CATEGORIES:
        assert v._normalise(category) == category == config.canonical(category)


def test_defect_ids_follow_case_ids(open_verifier_cases):
    for cases in open_verifier_cases.values():
        for case in cases:
            assert [d.defect_id for d in case.defects] == [
                f"{case.case_id}-d{n}" for n in range(1, len(case.defects) + 1)]


def test_split_vocabularies_match():
    assert v.SEALED_SPLITS == bm.SEALED_SPLITS
    assert set(bm.OPEN_SPLITS) | set(bm.SEALED_SPLITS) == set(bm.SPLITS)


def test_to_verifier_case_requires_loader_output(open_verifier_cases):
    with pytest.raises(TypeError):
        ev.to_verifier_case(open_verifier_cases["train"][0])


def test_reviews_outside_open_splits_are_rejected():
    with pytest.raises(ValueError, match="outside"):
        ev.evaluate_open({"pyrev-999": []})


# ── Sealed split separation ──────────────────────────────────────────────────

_OPENED: list[str] = []
_RECORDING = False


def _recorder(event, args):
    if _RECORDING and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        _OPENED.append(os.path.realpath(os.fsdecode(args[0])))


sys.addaudithook(_recorder)


@pytest.fixture
def opened_files():
    global _RECORDING
    _OPENED.clear()
    _RECORDING = True
    try:
        yield _OPENED
    finally:
        _RECORDING = False


def _touches_sealed(paths):
    return [p for p in paths if bm._is_sealed_path(p)]


def test_sealed_split_unreachable_inside_optimisation_phase(tmp_path):
    before = bm.sealed_access_attempts()
    link = tmp_path / "alias.jsonl"
    link.symlink_to(SEALED_FILE)
    with bm.optimisation_phase():
        assert bm.in_optimisation_phase()
        with pytest.raises(bm.SealedSplitError):
            open(SEALED_FILE, encoding="utf-8")
        with pytest.raises(bm.SealedSplitError):
            SEALED_FILE.read_bytes()
        with pytest.raises(bm.SealedSplitError):
            open(link, "rb")                       # symlink alias
        with pytest.raises(bm.SealedSplitError):
            bm.load_split("test", allow_sealed_test=True)
        with pytest.raises(bm.SealedSplitError):
            ev.evaluate_sealed_test({}, final_evaluation=True)
        assert len(bm.load_benchmark()["train"]) == 28   # open splits still work
    assert not bm.in_optimisation_phase()
    assert bm.sealed_access_attempts() >= before + 3


def test_copied_sealed_file_is_also_guarded(tmp_path):
    copy_root = tmp_path / "python_review"
    shutil.copytree(bm.BENCHMARK_ROOT, copy_root)
    with bm.optimisation_phase(), pytest.raises(bm.SealedSplitError):
        (copy_root / "v1" / "sealed" / "test.jsonl").read_text(encoding="utf-8")


def test_final_evaluation_is_explicit_and_outside_optimisation():
    with pytest.raises(bm.SealedSplitError):
        ev.evaluate_sealed_test({})
    report = ev.evaluate_sealed_test({}, final_evaluation=True, n_bootstrap=10)
    assert len(report.cases) == 10 and {r.split for r in report.cases} == {"test"}


def test_verifier_never_reads_sealed_cases_from_files(tmp_path):
    path = tmp_path / "copy.jsonl"
    path.write_bytes(SEALED_FILE.read_bytes())
    with pytest.raises(PermissionError, match="sealed"):
        v.load_cases(path)


def _review_loop(cases, misbehave=None):
    """Adaptive loop scoring reviews with the verifier on open-split cases."""
    by_id = {c.case_id: c for c in cases}
    vcases = {c.case_id: ev.to_verifier_case(c) for c in cases}
    responses = {}
    for case in cases:
        responses[(case.case_id, "oracle")] = FakeResponse(
            output=json.dumps(ev.ground_truth_findings(case)), cost_usd=0.02, latency_s=2.0)
        responses[(case.case_id, "silent")] = FakeResponse(
            output="[]", cost_usd=0.01, latency_s=1.0)
    fake = FakeExecutor(responses)
    catalogue = {"oracle": Strategy("oracle"), "silent": Strategy("silent")}
    run = loop_executor(fake, catalogue, lambda cid: Task(cid, by_id[cid].code))

    def executor(context, strategy_id):
        if misbehave:
            misbehave()
        return run(context, strategy_id)

    def verifier(outcome):
        quality = v.quality(vcases[outcome.task_id], json.loads(outcome.output))
        return Reward(quality, outcome.cost_usd / 0.02, outcome.latency_s / 2.0,
                      violation=outcome.violation)

    policy = ContextualBandit(("oracle", "silent"), epsilon=0.1, seed=3)
    with ExperienceStore(":memory:") as store:
        return [choose_strategy(policy, cid, executor, verifier, store,
                                skill_version="benchmark-v1-test")
                for cid in sorted(by_id)]


def test_adaptive_loop_over_open_splits_never_opens_sealed_file(opened_files, open_benchmark):
    before = bm.sealed_access_attempts()
    records = _review_loop(open_benchmark["train"] + open_benchmark["validation"])
    assert len(records) == 38
    assert _touches_sealed(opened_files) == []
    assert bm.sealed_access_attempts() == before


def test_synthetic_cli_loop_never_opens_sealed_file(opened_files, capsys):
    cli_args = ["--episodes", "30", "--seed", "7"]
    sys_argv = sys.argv
    sys.argv = ["cli", *cli_args]
    try:
        cli.main()
    finally:
        sys.argv = sys_argv
    assert "average utility" in capsys.readouterr().out
    assert _touches_sealed(opened_files) == []


@pytest.mark.parametrize("attack", ["open_file", "unseal_loader", "final_evaluation"])
def test_loop_fails_if_optimisation_code_touches_sealed_split(open_benchmark, attack):
    def misbehave():
        if attack == "open_file":
            SEALED_FILE.read_text(encoding="utf-8")
        elif attack == "unseal_loader":
            bm.load_split("test", allow_sealed_test=True)
        else:
            ev.evaluate_sealed_test({}, final_evaluation=True)
    with pytest.raises(bm.SealedSplitError):
        _review_loop(open_benchmark["train"][:2], misbehave=misbehave)


# Only these may name benchmark internals or unseal the test split.
_LOADER = Path("evolution/adaptive/benchmark.py")
_UNSEAL_ALLOWED = {(Path("evolution/adaptive/evaluation.py"), "sealed_test_cases")}
# The one recorded, run-once final evaluation (gate 8) is the only package code
# that may reach the sealed cases: per name, the (file, function) allowed to use it.
_FINAL_CALL_ALLOWED = {
    "evaluate_sealed_test": {(Path("evolution/adaptive/final_evaluation.py"),
                              "run_final_evaluation")},
    "sealed_test_cases": {(Path("evolution/adaptive/evaluation.py"), "evaluate_sealed_test"),
                          (Path("evolution/adaptive/final_evaluation.py"),
                           "run_paired_final_evaluation")},
}
_INTERNAL_NAMES = {"BENCHMARK_ROOT", "SPLIT_FILES", "_read_inside", "_read_verified",
                   "_parse_jsonl", "_read_manifest", "_phase_depth"}
_PATH_FRAGMENTS = ("python_review/", "python_review\\", "sealed/", "sealed\\",
                   "/sealed", "\\sealed", "test.jsonl")


def _scan(path: Path, rel: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
    problems = []
    function_of = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                function_of.setdefault(id(child), node.name)
    for node in ast.walk(tree):
        where = f"{rel}:{getattr(node, 'lineno', '?')}"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
            if text == "sealed" or any(f in text for f in _PATH_FRAGMENTS):
                problems.append(f"{where}: benchmark path literal {text[:60]!r}")
        elif isinstance(node, ast.Name) and node.id in _INTERNAL_NAMES:
            problems.append(f"{where}: uses benchmark internal {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in _INTERNAL_NAMES:
            problems.append(f"{where}: uses benchmark internal {node.attr}")
        elif isinstance(node, ast.alias) and node.name in _INTERNAL_NAMES:
            problems.append(f"{rel}: imports benchmark internal {node.name}")
        elif isinstance(node, ast.keyword) and node.arg == "allow_sealed_test":
            unsealing = not (isinstance(node.value, ast.Constant) and node.value.value is False)
            if unsealing and (rel, function_of.get(id(node))) not in _UNSEAL_ALLOWED:
                problems.append(f"{where}: passes allow_sealed_test outside final evaluation")
        elif (isinstance(node, (ast.Name, ast.Attribute))
              and (name := getattr(node, "id", getattr(node, "attr", None)))
              in _FINAL_CALL_ALLOWED
              and isinstance(getattr(node, "ctx", None), ast.Load)
              and (rel, function_of.get(id(node))) not in _FINAL_CALL_ALLOWED[name]):
            problems.append(f"{where}: calls {name} from package code")
    return problems


def scan_package(root: Path) -> list[str]:
    problems = []
    for path in sorted((root / "evolution").rglob("*.py")):
        rel = path.relative_to(root)
        if rel != _LOADER:
            problems += _scan(path, rel)
    return problems


def test_no_package_code_reaches_sealed_data_except_through_the_loader():
    assert scan_package(REPO) == []


def test_scanner_catches_direct_reads(tmp_path):
    bad = tmp_path / "evolution" / "adaptive" / "sneaky.py"
    bad.parent.mkdir(parents=True)
    bad.write_text(
        "from evolution.adaptive.benchmark import BENCHMARK_ROOT, load_split\n"
        "def leak():\n"
        "    open('datasets/python_review/v1/sealed/test.jsonl').read()\n"
        "    (BENCHMARK_ROOT / 'v1').exists()\n"
        "    load_split('test', allow_sealed_test=True)\n", encoding="utf-8")
    problems = scan_package(tmp_path)
    assert any("path literal" in p for p in problems)
    assert any("imports benchmark internal BENCHMARK_ROOT" in p for p in problems)
    assert any("allow_sealed_test" in p for p in problems)


# ── Gate 4 harness: compare() runs inside the optimisation phase ─────────────

from evolution.adaptive.compare import CaseBenchmark, ContextEncoder, compare, default_policy_specs  # noqa: E402
from evolution.adaptive.synthetic import REFERENCE_SCORING, STRATEGIES  # noqa: E402


def _compare_over_open_splits(open_benchmark, misbehave=None, seen_phase=None):
    cases = list(open_benchmark["train"]) + list(open_benchmark["validation"])
    bench = CaseBenchmark(cases)
    responses = {}
    for case in cases:
        for strategy in STRATEGIES:
            findings = ev.ground_truth_findings(case) if strategy == "structured" else []
            responses[(case.case_id, strategy)] = FakeResponse(
                output=json.dumps(findings), cost_usd=0.02, latency_s=3.0)
    vcases = {c.case_id: ev.to_verifier_case(c) for c in cases}

    def quality_fn(item, outcome):
        if seen_phase is not None:
            seen_phase.append(bm.in_optimisation_phase())
        if misbehave:
            misbehave()
        return v.quality(vcases[item.task_id], json.loads(outcome.output))

    encoder = ContextEncoder.fit(bench.items("train"), key_fields=("difficulty",),
                                 categorical=("difficulty",), numeric={"n_lines": 100.0})
    return compare(bench, FakeExecutor(responses), quality_fn,
                   {s: Strategy(s) for s in STRATEGIES}, encoder,
                   default_policy_specs(STRATEGIES, encoder.dim),
                   seeds=[0], train_episodes=20, scoring=REFERENCE_SCORING, n_resamples=20)


def test_compare_runs_inside_optimisation_phase_and_never_opens_sealed(
        opened_files, open_benchmark):
    seen = []
    result = _compare_over_open_splits(open_benchmark, seen_phase=seen)
    assert result.report["policies"]
    assert seen and all(seen)
    assert not bm.in_optimisation_phase()
    assert _touches_sealed(opened_files) == []


@pytest.mark.parametrize("attack", ["open_file", "unseal_loader", "final_evaluation"])
def test_compare_fails_if_it_touches_sealed_split(open_benchmark, attack):
    def misbehave():
        if attack == "open_file":
            SEALED_FILE.read_bytes()
        elif attack == "unseal_loader":
            bm.load_split("test", allow_sealed_test=True)
        else:
            ev.evaluate_sealed_test({}, final_evaluation=True)
    with pytest.raises(bm.SealedSplitError):
        _compare_over_open_splits(open_benchmark, misbehave=misbehave)


# Optimisation entry points that must run their body inside optimisation_phase().
_GUARDED_ENTRY_POINTS = (
    (Path("evolution/adaptive/core.py"), "choose_strategy"),
    (Path("evolution/adaptive/cli.py"), "main"),
    (Path("evolution/adaptive/compare.py"), "compare"),
    (Path("evolution/adaptive/checkpoint.py"), "admit_with_rollback"),
    (Path("evolution/adaptive/checkpoint.py"), "validation_utility"),
    (Path("evolution/adaptive/admission.py"), "admit"),
    (Path("evolution/adaptive/admission.py"), "catalogue_validation"),
    (Path("evolution/adaptive/admission.py"), "review_with_rollback"),
    (Path("evolution/adaptive/report.py"), "generate"),
)


def _is_phase_call(expr) -> bool:
    func = expr.func if isinstance(expr, ast.Call) else None
    return (isinstance(func, ast.Name) and func.id == "optimisation_phase") or \
           (isinstance(func, ast.Attribute) and func.attr == "optimisation_phase")


def unguarded_entry_points(root: Path) -> list[str]:
    problems = []
    for rel, name in _GUARDED_ENTRY_POINTS:
        path = root / rel
        if not path.exists():
            problems.append(f"{rel}: missing")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(funcs) != 1:
            problems.append(f"{rel}: expected one top-level {name}()")
            continue
        guarded = any(isinstance(stmt, ast.With) and any(_is_phase_call(i.context_expr)
                                                         for i in stmt.items)
                      for stmt in funcs[0].body)
        if not guarded:
            problems.append(f"{rel}:{name}() does not run inside optimisation_phase()")
    return problems


def test_loop_cli_and_compare_entry_points_are_guarded():
    assert unguarded_entry_points(REPO) == []


def test_entry_point_check_catches_an_unguarded_compare(tmp_path):
    for rel in {rel for rel, _ in _GUARDED_ENTRY_POINTS}:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, target)
    (tmp_path / "evolution/adaptive/compare.py").write_text(
        "def compare(*args, **kwargs):\n    return _compare(*args, **kwargs)\n",
        encoding="utf-8")
    assert unguarded_entry_points(tmp_path) == [
        "evolution/adaptive/compare.py:compare() does not run inside optimisation_phase()"]


# ── Clean controls cannot inflate scores ─────────────────────────────────────

@pytest.mark.parametrize("split", bm.OPEN_SPLITS)
def test_false_finding_on_clean_control_scores_zero(split, open_benchmark):
    cases = open_benchmark[split]
    clean = [c for c in cases if c.is_clean]
    assert clean
    false_finding = [{"line_start": 1, "line_end": 1, "category": "logic_error"}]

    only_clean = ev.evaluate_open({c.case_id: false_finding for c in clean},
                                  splits=(split,), n_bootstrap=20)[split]
    by_id = {r.case_id: r for r in only_clean.cases}
    for case in clean:
        result = by_id[case.case_id]
        assert (result.true_positives, result.false_positives) == (0, 1)
        assert result.precision == result.recall == result.f1 == result.quality == 0.0

    perfect = {c.case_id: ev.ground_truth_findings(c) for c in cases}
    noisy = dict(perfect, **{c.case_id: false_finding for c in clean})
    best = ev.evaluate_open(perfect, splits=(split,), n_bootstrap=20)[split]
    worse = ev.evaluate_open(noisy, splits=(split,), n_bootstrap=20)[split]
    assert best.micro.precision.point == best.macro.f1.point == 1.0
    assert worse.micro.precision.point < 1.0
    assert worse.macro.f1.point == pytest.approx(1 - len(clean) / len(cases))
