"""Gate 6 candidate-vs-baseline admission. Offline: FakeExecutor only."""
import dataclasses
import inspect
import json
import logging
import os
import sys

import pytest

from evolution.adaptive import admission as adm
from evolution.adaptive import benchmark as bm
from evolution.adaptive import evaluation as ev
from evolution.adaptive.admission import ADMISSION_POLICY, AdmissionPolicy, ArmSpec, admit
from evolution.adaptive.executor import FakeExecutor, FakeResponse, Strategy

# Changing any admission threshold changes this; update it only in a reviewed change.
PINNED_POLICY_SHA256 = "1c3cd3e28dd5f9d140183bae6a747b7b0f396358adb93d14eeceff6913f0f4ed"

SEALED_FILE = bm.BENCHMARK_ROOT / "v1" / "sealed" / "test.jsonl"
CANDIDATE = ArmSpec("cand-1", Strategy("candidate"))
BASELINE = ArmSpec("base-1", Strategy("baseline"))


@pytest.fixture(scope="module")
def validation():
    return bm.load_split("validation")


def _truth(case):
    return json.dumps(ev.ground_truth_findings(case))


def factory(validation, cand=None, base=None, calls=None, during=None):
    """make_executor(seed): candidate/baseline response per case (default: silent)."""
    def response(spec, case):
        value = spec(case) if callable(spec) else spec
        if isinstance(value, FakeResponse):
            return value
        return FakeResponse(output=value if value is not None else "[]",
                            cost_usd=0.01, latency_s=2.0)

    def make_executor(seed):
        if calls is not None:
            calls.append(seed)
        responses = {}
        for case in validation:
            responses[(case.case_id, "candidate")] = response(cand, case)
            responses[(case.case_id, "baseline")] = response(base, case)
        executor = FakeExecutor(responses)
        if during is not None:
            original = executor.execute

            def execute(task, strategy):
                during(task, strategy)
                return original(task, strategy)
            executor.execute = execute
        make_executor.instances.append(executor)
        return executor
    make_executor.instances = []
    return make_executor


# ── Decisions ────────────────────────────────────────────────────────────────

def test_clearly_better_candidate_is_admitted(validation, caplog):
    with caplog.at_level(logging.INFO, logger=adm.__name__):
        d = admit(CANDIDATE, BASELINE, factory(validation, cand=_truth))
    assert d.admitted and d.reasons == ()
    assert d.mean_utility_gain >= ADMISSION_POLICY.min_margin and d.ci_low > 0
    assert d.candidate_mean_quality == 1.0
    clean_share = sum(c.is_clean for c in validation) / len(validation)
    assert d.baseline_mean_quality == pytest.approx(clean_share)
    assert d.synthetic is True and "SYNTHETIC" in d.notice
    assert d.n_cases == 10 and set(d.case_ids) == {c.case_id for c in validation}
    assert "ADMIT cand-1" in caplog.text


def test_no_improvement_is_rejected_with_both_reasons(validation, caplog):
    with caplog.at_level(logging.WARNING, logger=adm.__name__):
        d = admit(CANDIDATE, BASELINE, factory(validation, cand=_truth, base=_truth))
    assert not d.admitted and d.mean_utility_gain == 0.0
    assert [r.split(":")[0] for r in d.reasons] == ["margin_not_met", "ci_includes_zero"]
    assert "REJECT cand-1" in caplog.text and "margin_not_met" in caplog.text


def test_gain_on_one_case_meets_margin_but_ci_includes_zero(validation):
    fixed = next(c for c in validation if not c.is_clean)
    d = admit(CANDIDATE, BASELINE, factory(
        validation, cand=lambda c: _truth(c) if c.case_id == fixed.case_id else "[]"))
    assert d.mean_utility_gain >= ADMISSION_POLICY.min_margin
    assert not d.admitted and [r.split(":")[0] for r in d.reasons] == ["ci_includes_zero"]


def test_consistent_but_small_gain_fails_the_margin(validation):
    # Same reviews; candidate is slightly cheaper on every case: CI excludes 0,
    # but 0.10 * (0.006 / 0.03) = 0.02 < min_margin.
    cheap = lambda c: FakeResponse(output="[]", cost_usd=0.0, latency_s=2.0)    # noqa: E731
    dear = lambda c: FakeResponse(output="[]", cost_usd=0.006, latency_s=2.0)   # noqa: E731
    d = admit(CANDIDATE, BASELINE, factory(validation, cand=cheap, base=dear))
    assert d.mean_utility_gain == pytest.approx(0.02) and d.ci_low > 0
    assert not d.admitted and [r.split(":")[0] for r in d.reasons] == ["margin_not_met"]


def test_any_critical_violation_rejects_even_a_better_candidate(validation):
    bad = validation[0].case_id

    def cand(case):
        tools = ("terminal",) if case.case_id == bad else ()
        return FakeResponse(output=_truth(case), cost_usd=0.01, latency_s=2.0,
                            tool_calls=tools)
    d = admit(CANDIDATE, BASELINE, factory(validation, cand=cand))
    assert d.candidate_violations == len(ADMISSION_POLICY.seeds)
    assert not d.admitted
    assert d.reasons[0].startswith("critical_violations") and "disallowed_tool:terminal" in d.reasons[0]


def test_unknown_cost_cannot_be_admitted(validation):
    d = admit(CANDIDATE, BASELINE, factory(
        validation, cand=lambda c: FakeResponse(output=_truth(c), cost_usd=None)))
    assert not d.admitted and d.reasons[0].startswith("unscorable_outcomes")
    assert d.mean_utility_gain is None


def test_unparseable_output_scores_as_an_empty_review(validation):
    d = admit(CANDIDATE, BASELINE, factory(validation, cand="not json", base="[]"))
    assert d.unparseable_outputs == len(validation) * len(ADMISSION_POLICY.seeds)
    assert d.mean_utility_gain == 0.0 and not d.admitted


# ── Same split, executor and seeds; sealed split unreachable ─────────────────

def test_both_arms_use_same_cases_executor_factory_and_seeds(validation):
    calls = []
    make = factory(validation, cand=_truth, calls=calls)
    admit(CANDIDATE, BASELINE, make)
    seeds = list(ADMISSION_POLICY.seeds)
    assert calls == seeds + seeds                       # once per arm and seed
    assert len({id(e) for e in make.instances}) == 2 * len(seeds)   # no shared state
    cand_runs = [c for e in make.instances for c in e.calls if c[1] == "candidate"]
    base_runs = [c for e in make.instances for c in e.calls if c[1] == "baseline"]
    assert sorted(t for t, _ in cand_runs) == sorted(t for t, _ in base_runs)
    assert {t for t, _ in cand_runs} == {c.case_id for c in validation}


_OPENED = []
_RECORDING = False


def _recorder(event, args):
    if _RECORDING and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        _OPENED.append(os.path.realpath(os.fsdecode(args[0])))


sys.addaudithook(_recorder)


def test_admission_runs_in_optimisation_phase_and_never_opens_sealed(validation):
    global _RECORDING
    phases = []
    _OPENED.clear()
    _RECORDING = True
    try:
        admit(CANDIDATE, BASELINE, factory(
            validation, cand=_truth, during=lambda t, s: phases.append(bm.in_optimisation_phase())))
    finally:
        _RECORDING = False
    assert phases and all(phases)
    assert [p for p in _OPENED if bm._is_sealed_path(p)] == []


@pytest.mark.parametrize("attack", ["open_file", "unseal_loader"])
def test_admission_fails_if_executor_touches_sealed_split(validation, attack):
    def during(task, strategy):
        if attack == "open_file":
            SEALED_FILE.read_bytes()
        else:
            bm.load_split("test", allow_sealed_test=True)
    with pytest.raises(bm.SealedSplitError):
        admit(CANDIDATE, BASELINE, factory(validation, cand=_truth, during=during))


# ── Fixed configuration ──────────────────────────────────────────────────────

def test_thresholds_are_not_parameters_of_admit():
    assert list(inspect.signature(admit).parameters) == ["candidate", "baseline", "make_executor"]


def test_policy_is_frozen_and_pinned():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ADMISSION_POLICY.min_margin = 0.0
    assert ADMISSION_POLICY.fingerprint() == PINNED_POLICY_SHA256
    d_policy = ADMISSION_POLICY.to_dict()
    assert d_policy["split"] == "validation" and d_policy["max_critical_violations"] == 0


def test_decision_records_policy_and_verifier(validation):
    d = admit(CANDIDATE, BASELINE, factory(validation, cand=_truth))
    assert d.policy_sha256 == PINNED_POLICY_SHA256
    assert d.policy == ADMISSION_POLICY.to_dict()
    assert len(d.verifier_sha256) == 64
    assert json.loads(json.dumps(d.to_dict()))["admitted"] is True


@pytest.mark.parametrize("change", [dict(split="test"), dict(split="train"),
                                    dict(max_critical_violations=1), dict(min_margin=0.0),
                                    dict(seeds=()), dict(seeds=(1, 1))])
def test_policy_refuses_unsafe_settings(change):
    with pytest.raises(ValueError):
        AdmissionPolicy(**change)


# ── Input checks ─────────────────────────────────────────────────────────────

def test_candidate_shape_is_checked(validation):
    make = factory(validation)
    with pytest.raises(ValueError, match="differ"):
        admit(CANDIDATE, ArmSpec("cand-1", Strategy("baseline")), make)
    with pytest.raises(TypeError):
        admit(ArmSpec("x", "not a strategy"), BASELINE, make)
    with pytest.raises(TypeError):
        admit(object(), BASELINE, make)
    assert isinstance(CANDIDATE, adm.Arm)


def test_parse_findings():
    assert adm.parse_findings('[{"line_start": 1}]') == ([{"line_start": 1}], True)
    assert adm.parse_findings('{"findings": []}') == ([], False)
    assert adm.parse_findings("") == ([], False)


# ── Gate 5 registry review ───────────────────────────────────────────────────

import ast  # noqa: E402
from pathlib import Path  # noqa: E402

from evolution.adaptive import registry as reg  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BASE_STRATEGY = Strategy("reviewer")


def registry_factory(validation, cand_id, base_id, cand_spec, base_spec="[]"):
    def make_executor(seed):
        responses = {}
        for case in validation:
            for cid, spec in ((cand_id, cand_spec), (base_id, base_spec)):
                out = spec(case) if callable(spec) else spec
                responses[(case.case_id, f"eval:reviewer:{cid}")] = FakeResponse(
                    output=out, cost_usd=0.01, latency_s=2.0)
        return FakeExecutor(responses)
    return make_executor


@pytest.fixture
def registry(tmp_path):
    with reg.CandidateRegistry(tmp_path / "registry.db") as r:
        yield r


@pytest.fixture
def pair(registry):
    base = registry.register_baseline("code-review", b"baseline skill\n")
    cand = registry.propose("code-review", b"evolved skill\n", parent_id=base.candidate_id,
                            provenance={"gepa_score": 0.99})   # never read by admission
    return base, cand


def _review(registry, validation, pair, better=True, apply=True):
    base, cand = pair
    make = registry_factory(validation, cand.candidate_id, base.candidate_id,
                            _truth if better else "[]")
    return adm.review_registered(registry, cand.candidate_id, base=BASE_STRATEGY,
                                 make_executor=make, apply=apply)


def test_review_is_a_dry_run_by_default(registry, validation, pair):
    d = _review(registry, validation, pair, apply=False)
    assert d.admitted and d.baseline_id == pair[0].candidate_id
    assert registry.get(pair[1].candidate_id).status == reg.PROPOSED
    assert [t.to_status for t in registry.history(pair[1].candidate_id)] == [reg.PROPOSED]


@pytest.mark.parametrize("better", [True, False])
def test_review_records_admission_or_rejection_with_evidence(registry, validation, pair, better):
    d = _review(registry, validation, pair, better=better)
    last = registry.history(pair[1].candidate_id)[-1]
    assert last.to_status == (reg.ADMITTED if better else reg.REJECTED) and d.admitted is better
    assert last.decided_by == f"admission-gate:{PINNED_POLICY_SHA256[:12]}"
    assert last.evidence["policy_sha256"] == PINNED_POLICY_SHA256
    assert last.evidence["synthetic"] is True
    assert better or "margin_not_met" in last.reason
    if not better:   # rejection is terminal in gate 5
        with pytest.raises(ValueError):
            _review(registry, validation, pair)


def test_gate_admitted_candidate_enters_the_catalogue(registry, validation, pair):
    _review(registry, validation, pair)
    assert adm.admission_evidence_errors(registry, pair[1].candidate_id) == []
    prompts = [s.system_prompt for s in adm.admitted_catalogue(registry, "code-review",
                                                               BASE_STRATEGY)]
    assert prompts == ["baseline skill\n", "evolved skill\n"]


def test_registry_refuses_arbitrary_admission_evidence(registry, pair):
    with pytest.raises(reg.TransitionError):
        registry.transition(pair[1].candidate_id, reg.ADMITTED, reason="looks good",
                            decided_by="someone", evidence={"gepa_score": 0.99})
    assert registry.get(pair[1].candidate_id).status == reg.PROPOSED


def test_catalogue_refuses_minimal_evidence_not_made_by_the_gate(registry, pair):
    # Shape the registry accepts, but not a full gate 6 decision: defence in depth.
    registry.transition(pair[1].candidate_id, reg.ADMITTED, reason="manual",
                        decided_by="someone",
                        evidence={"admitted": True, "candidate_id": pair[1].candidate_id,
                                  "policy_sha256": PINNED_POLICY_SHA256})
    errors = adm.admission_evidence_errors(registry, pair[1].candidate_id)
    assert any("not the admission gate" in e for e in errors)
    assert any("margin and CI" in e for e in errors)
    with pytest.raises(adm.AdmissionEvidenceError):
        adm.admitted_catalogue(registry, "code-review", BASE_STRATEGY)


def _forged(registry, validation, pair, **changes):
    """Evidence copied from a real passing decision for this pair, then edited."""
    base, cand = pair
    decision = adm.admit(adm.evaluation_arm(registry, cand.candidate_id, BASE_STRATEGY),
                         adm.evaluation_arm(registry, base.candidate_id, BASE_STRATEGY),
                         registry_factory(validation, cand.candidate_id,
                                          base.candidate_id, _truth))
    evidence = {**decision.to_dict(), **changes.pop("evidence", {})}
    registry.transition(cand.candidate_id, reg.ADMITTED, reason="forged",
                        decided_by=changes.pop("decided_by",
                                               f"admission-gate:{PINNED_POLICY_SHA256[:12]}"),
                        evidence=evidence)
    return adm.admission_evidence_errors(registry, cand.candidate_id)


@pytest.mark.parametrize("changes", [
    {"evidence": {"candidate_id": "code-review@000000000000"}},
    {"evidence": {"admitted": False}},
    {"evidence": {"policy_sha256": "0" * 64}},
])
def test_registry_refuses_edited_decisions(registry, validation, pair, changes):
    with pytest.raises(reg.TransitionError):
        _forged(registry, validation, pair, **changes)
    assert registry.get(pair[1].candidate_id).status == reg.PROPOSED


@pytest.mark.parametrize("changes, expected", [
    ({}, None),   # a genuine decision recorded by hand is accepted
    ({"decided_by": "someone"}, "not the admission gate"),
    ({"evidence": {"candidate_violations": 2}}, "violations"),
    ({"evidence": {"ci_low": -0.1}}, "margin and CI"),
    ({"evidence": {"policy": {**ADMISSION_POLICY.to_dict(), "min_margin": 0.0001}}},
     "does not match"),
])
def test_forged_or_edited_evidence_is_detected(registry, validation, pair, changes, expected):
    errors = _forged(registry, validation, pair, **changes)
    if expected is None:
        assert errors == []
    else:
        assert any(expected in e for e in errors), errors


def test_evaluation_arm_takes_tools_from_base_only(registry, pair):
    base = Strategy("reviewer", system_prompt="ignored", toolsets=("todo",))
    arm = adm.evaluation_arm(registry, pair[1].candidate_id, base)
    assert arm.strategy.system_prompt == "evolved skill\n"
    assert arm.strategy.toolsets == ("todo",)
    assert arm.strategy.strategy_id == f"eval:reviewer:{pair[1].candidate_id}"
    assert registry.get(pair[1].candidate_id).status == reg.PROPOSED


def test_review_refuses_wrong_states(registry, validation, pair):
    base, cand = pair
    make = registry_factory(validation, cand.candidate_id, base.candidate_id, "[]")
    with pytest.raises(ValueError, match="only proposed"):
        adm.review_registered(registry, base.candidate_id, base=BASE_STRATEGY,
                              make_executor=make)
    registry.transition(base.candidate_id, reg.RETIRED, reason="superseded",
                        decided_by="test", evidence={})
    with pytest.raises(ValueError, match="not admitted"):
        adm.review_registered(registry, cand.candidate_id, base=BASE_STRATEGY,
                              make_executor=make)


def _transition_calls(root: Path) -> list[str]:
    found = []
    for path in sorted((root / "evolution").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = {}
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(fn):
                    owner.setdefault(id(child), fn.name)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "transition"):
                found.append(f"{rel}:{owner.get(id(node))}")
    return found


def test_only_the_admission_gate_changes_candidate_status():
    assert sorted(_transition_calls(REPO)) == [
        "evolution/adaptive/admission.py:_retire_after_rollback",   # gate 7: to retired only
        "evolution/adaptive/admission.py:review_registered"]


def test_transition_scan_catches_other_callers(tmp_path):
    sneaky = tmp_path / "evolution" / "adaptive" / "sneaky.py"
    sneaky.parent.mkdir(parents=True)
    sneaky.write_text("def promote(r, cid):\n    r.transition(cid, 'admitted', reason='x',"
                      " decided_by='x', evidence={'ok': 1})\n", encoding="utf-8")
    assert _transition_calls(tmp_path) == ["evolution/adaptive/sneaky.py:promote"]


def test_only_the_admission_gate_registers_admission_policies():
    callers = []
    for path in sorted((REPO / "evolution").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", None))
                if name == "register_admission_policy":
                    callers.append(path.relative_to(REPO).as_posix())
    assert callers == ["evolution/adaptive/admission.py"]
    assert reg.registered_admission_policies() == adm.ACCEPTED_POLICY_SHA256
