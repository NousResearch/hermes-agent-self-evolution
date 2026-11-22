"""Offline tests for the candidate registry and GEPA adapter (stubbed GEPA output)."""
import ast
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from evolution.adaptive import gepa_adapter, registry as reg
from evolution.adaptive.executor import Strategy
from evolution.adaptive.gepa_adapter import (
    GepaOutputError, read_gepa_output, register_gepa_output,
)
from evolution.adaptive.registry import (
    ADMITTED, ALLOWED_TRANSITIONS, PROPOSED, REJECTED, RETIRED, CandidateRegistry,
    IntegrityError, RegistryError, TransitionError, admitted_strategies,
    candidate_strategy, content_sha256,
)

BASE = b"---\nname: code-review\ndescription: Review code\n---\n\nCheck for bugs.\n"
EVOLVED = b"---\nname: code-review\ndescription: Review code\n---\n\nCheck bugs and leaks.\n"
EVIDENCE = {"validation": "candidate beat baseline", "split": "validation"}

# Importing the admission gate registers its policy fingerprint with the registry.
from evolution.adaptive.admission import ADMISSION_POLICY  # noqa: E402


def admitted_evidence(candidate_id):
    """Minimal evidence shape the registry accepts for ``admitted`` (a gate 6 decision)."""
    return {"admitted": True, "candidate_id": candidate_id,
            "policy_sha256": ADMISSION_POLICY.fingerprint(), "reasons": []}


def evidence_for(candidate_id, status):
    return admitted_evidence(candidate_id) if status == ADMITTED else EVIDENCE


@pytest.fixture
def registry():
    with CandidateRegistry(":memory:") as r:
        yield r


def gepa_dir(tmp_path, *, evolved=EVOLVED, baseline=BASE, metrics=None, failed=False,
             name="20261122_120000"):
    """Stub of what evolve_skill.evolve() writes; no GEPA or LLM involved."""
    d = tmp_path / "output" / "code-review" / name
    d.mkdir(parents=True)
    (d / ("evolved_FAILED.md" if failed else "evolved_skill.md")).write_bytes(evolved)
    if baseline is not None:
        (d / "baseline_skill.md").write_bytes(baseline)
    if metrics is not False:
        m = {"skill_name": "code-review", "timestamp": name, "iterations": 3,
             "optimizer_model": "stub", "eval_model": "stub", "baseline_score": 0.5,
             "evolved_score": 0.9, "improvement": 0.4, "train_examples": 4,
             "val_examples": 2, "holdout_examples": 2, "elapsed_seconds": 1.0,
             "constraints_passed": True}
        m.update(metrics or {})
        (d / "metrics.json").write_text(json.dumps(m))
    return d


# ── GEPA adapter ───────────────────────────────────────────────────────────

def test_gepa_output_registered_as_proposed_with_baseline_parent(registry, tmp_path):
    cand = register_gepa_output(registry, gepa_dir(tmp_path))
    base_id = f"code-review@{hashlib.sha256(BASE).hexdigest()[:12]}"
    assert cand.status == PROPOSED and cand.source == "gepa"
    assert cand.parent_id == base_id
    assert cand.content_sha256 == hashlib.sha256(EVOLVED).hexdigest()
    assert registry.content(cand.candidate_id) == EVOLVED
    parent = registry.get(base_id)
    assert (parent.status, parent.source, parent.parent_id) == (ADMITTED, "baseline", None)
    assert cand.provenance["gepa_reported"]["evolved_score"] == 0.9
    assert cand.provenance["gepa_reported_status"].startswith("unverified")
    assert cand.provenance["data_label"] == "synthetic"
    assert cand.created_at.endswith("+00:00")


def test_registration_is_idempotent(registry, tmp_path):
    a = register_gepa_output(registry, gepa_dir(tmp_path, name="run1"))
    b = register_gepa_output(registry, gepa_dir(tmp_path, name="run2"))
    assert a == b and len(registry.candidates()) == 2


def test_high_gepa_score_never_admits(registry, tmp_path):
    d = gepa_dir(tmp_path, metrics={"evolved_score": 1.0, "improvement": 0.99})
    cand = register_gepa_output(registry, d)
    assert cand.status == PROPOSED
    assert registry.candidates(status=ADMITTED) == (registry.get(cand.parent_id),)
    assert admitted_strategies(registry, "code-review", Strategy("structured")) == (
        candidate_strategy(registry, cand.parent_id, Strategy("structured")),)


def test_failed_constraints_still_proposed_and_flagged(registry, tmp_path):
    d = gepa_dir(tmp_path, failed=True, baseline=None, metrics=False)
    with pytest.raises(GepaOutputError):
        register_gepa_output(registry, d)   # no baseline next to evolved_FAILED.md
    cand = register_gepa_output(registry, d, baseline=BASE, skill_name="code-review")
    assert cand.status == PROPOSED and cand.provenance["constraints_passed"] is False


def test_gepa_dir_is_not_modified(registry, tmp_path):
    d = gepa_dir(tmp_path)
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    register_gepa_output(registry, d)
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before


@pytest.mark.parametrize("setup, message", [
    (lambda d: (d / "evolved_skill.md").unlink(), "exactly one"),
    (lambda d: (d / "evolved_FAILED.md").write_bytes(EVOLVED), "exactly one"),
    (lambda d: (d / "baseline_skill.md").unlink(), "missing"),
    (lambda d: (d / "metrics.json").write_text("{bad"), "invalid JSON"),
    (lambda d: (d / "metrics.json").write_text('{"skill_name": "other"}'), "inconsistent"),
    (lambda d: (d / "metrics.json").write_text('{"evolved_score": NaN}'), "not finite"),
    (lambda d: (d / "evolved_skill.md").write_bytes(BASE), "identical"),
    (lambda d: (d / "evolved_skill.md").write_bytes(b"\xff\xfe"), "UTF-8"),
])
def test_malformed_gepa_output_rejected(registry, tmp_path, setup, message):
    d = gepa_dir(tmp_path)
    setup(d)
    with pytest.raises(GepaOutputError, match=message):
        register_gepa_output(registry, d)
    assert registry.candidates() == ()


def test_symlinked_skill_file_refused(tmp_path):
    d = gepa_dir(tmp_path)
    target = tmp_path / "elsewhere.md"
    target.write_bytes(EVOLVED + b"x")
    (d / "evolved_skill.md").unlink()
    (d / "evolved_skill.md").symlink_to(target)
    with pytest.raises(GepaOutputError, match="symlink"):
        read_gepa_output(d)


def test_evolving_an_admitted_candidate_uses_it_as_parent(registry, tmp_path):
    first = register_gepa_output(registry, gepa_dir(tmp_path, name="r1"))
    registry.transition(first.candidate_id, ADMITTED, reason="test", decided_by="test",
                        evidence=admitted_evidence(first.candidate_id))
    second_body = EVOLVED + b"Also check types.\n"
    second = register_gepa_output(
        registry, gepa_dir(tmp_path, name="r2", baseline=EVOLVED, evolved=second_body))
    assert second.parent_id == first.candidate_id and second.status == PROPOSED


def test_proposed_baseline_cannot_be_a_parent(registry, tmp_path):
    first = register_gepa_output(registry, gepa_dir(tmp_path, name="r1"))
    d = gepa_dir(tmp_path, name="r2", baseline=EVOLVED, evolved=EVOLVED + b"more\n")
    with pytest.raises(RegistryError, match="admitted"):
        register_gepa_output(registry, d)
    assert registry.get(first.candidate_id).status == PROPOSED


# ── Registry rules ─────────────────────────────────────────────────────────

def _proposed(registry):
    base = registry.register_baseline("code-review", BASE)
    return base, registry.propose("code-review", EVOLVED, parent_id=base.candidate_id)


def test_transition_table_is_fixed():
    assert dict(ALLOWED_TRANSITIONS) == {
        PROPOSED: {ADMITTED, REJECTED}, ADMITTED: {RETIRED}, REJECTED: set(), RETIRED: set()}
    with pytest.raises(TypeError):
        ALLOWED_TRANSITIONS[REJECTED] = frozenset({ADMITTED})


@pytest.mark.parametrize("path", [
    (REJECTED, ADMITTED), (REJECTED, PROPOSED), (ADMITTED, PROPOSED),
    (ADMITTED, REJECTED), (RETIRED, ADMITTED),
])
def test_illegal_transitions_raise(registry, path):
    _, cand = _proposed(registry)
    steps = {REJECTED: [REJECTED], ADMITTED: [ADMITTED], RETIRED: [ADMITTED, RETIRED]}
    for status in steps[path[0]]:
        registry.transition(cand.candidate_id, status, reason="r", decided_by="t",
                            evidence=evidence_for(cand.candidate_id, status))
    with pytest.raises(TransitionError):
        registry.transition(cand.candidate_id, path[1], reason="r", decided_by="t",
                            evidence=EVIDENCE)


@pytest.mark.parametrize("kwargs", [
    {"reason": "", "decided_by": "t", "evidence": EVIDENCE},
    {"reason": "r", "decided_by": " ", "evidence": EVIDENCE},
    {"reason": "r", "decided_by": "t", "evidence": {}},
    {"reason": "r", "decided_by": "t", "evidence": {"x": float("nan")}},
    {"reason": "r", "decided_by": "t", "evidence": "passed"},
])
def test_admission_needs_reason_author_and_evidence(registry, kwargs):
    _, cand = _proposed(registry)
    with pytest.raises(ValueError):
        registry.transition(cand.candidate_id, ADMITTED, **kwargs)
    assert registry.get(cand.candidate_id).status == PROPOSED


def test_history_is_recorded_and_append_only(registry):
    _, cand = _proposed(registry)
    registry.transition(cand.candidate_id, REJECTED, reason="worse than baseline",
                        decided_by="gate6", evidence=EVIDENCE)
    hist = registry.history(cand.candidate_id)
    assert [(h.from_status, h.to_status) for h in hist] == [(None, PROPOSED),
                                                          (PROPOSED, REJECTED)]
    assert hist[1].evidence == EVIDENCE and hist[1].decided_by == "gate6"
    for sql in ("UPDATE transitions SET to_status='admitted'", "DELETE FROM transitions",
                "DELETE FROM candidates",
                "UPDATE candidates SET content=x'00'",
                "UPDATE candidates SET content_sha256='0'"):
        with pytest.raises(sqlite3.DatabaseError):
            registry.conn.execute(sql)


def test_rejected_content_cannot_be_reproposed(registry):
    base, cand = _proposed(registry)
    registry.transition(cand.candidate_id, REJECTED, reason="r", decided_by="t", evidence={})
    with pytest.raises(RegistryError, match="rejected"):
        registry.propose("code-review", EVOLVED, parent_id=base.candidate_id)


def test_propose_rules(registry):
    base = registry.register_baseline("code-review", BASE)
    with pytest.raises(KeyError):
        registry.propose("code-review", EVOLVED, parent_id="code-review@000000000000")
    other = registry.register_baseline("other-skill", b"other\n")
    with pytest.raises(ValueError, match="belongs to skill"):
        registry.propose("code-review", EVOLVED, parent_id=other.candidate_id)
    with pytest.raises(ValueError):
        registry.propose("code-review", EVOLVED, parent_id=base.candidate_id,
                         source="baseline")
    with pytest.raises(TypeError):
        registry.propose("code-review", "text not bytes", parent_id=base.candidate_id)
    with pytest.raises(ValueError):
        registry.register_baseline("../escape", BASE)


def test_tampered_content_detected(registry):
    _, cand = _proposed(registry)
    registry.conn.execute("DROP TRIGGER candidates_identity_fixed")  # simulate tampering
    registry.conn.execute("UPDATE candidates SET content=? WHERE candidate_id=?",
                          (b"evil", cand.candidate_id))
    with pytest.raises(IntegrityError):
        registry.content(cand.candidate_id)
    with pytest.raises(IntegrityError):
        registry.transition(cand.candidate_id, ADMITTED, reason="r", decided_by="t",
                            evidence=EVIDENCE)


def test_persistence_roundtrip(tmp_path):
    path = tmp_path / "registry.db"
    with CandidateRegistry(path) as r:
        _, cand = _proposed(r)
    with CandidateRegistry(path) as r:
        assert r.get(cand.candidate_id) == cand
        assert r.content(cand.candidate_id) == EVOLVED
        assert len(r.history()) == 2


def test_to_dict_matches_schema(registry):
    _, cand = _proposed(registry)
    d = cand.to_dict()
    assert set(d) == {"schema_version", "candidate_id", "skill_name", "parent_id",
                      "content_sha256", "source", "status", "created_at", "provenance"}
    assert d["schema_version"] == reg.SCHEMA_VERSION
    assert d["content_sha256"] == content_sha256(EVOLVED)


# ── Strategy catalogue ─────────────────────────────────────────────────────

def test_only_admitted_candidates_become_strategies(registry):
    base, cand = _proposed(registry)
    parent = Strategy("structured", system_prompt="ignored", toolsets=("todo",))
    with pytest.raises(RegistryError, match="only admitted"):
        candidate_strategy(registry, cand.candidate_id, parent)
    assert [s.strategy_id for s in admitted_strategies(registry, "code-review", parent)] == [
        f"structured:{base.candidate_id}"]
    registry.transition(cand.candidate_id, ADMITTED, reason="r", decided_by="t",
                        evidence=admitted_evidence(cand.candidate_id))
    strategies = admitted_strategies(registry, "code-review", parent)
    assert [s.system_prompt for s in strategies] == [BASE.decode(), EVOLVED.decode()]
    assert all(s.toolsets == ("todo",) for s in strategies)


# ── Isolation ──────────────────────────────────────────────────────────────

def _imports(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(("." * node.level) + (node.module or ""))
    return tree, names


@pytest.mark.parametrize("module", [reg, gepa_adapter])
def test_no_gepa_pipeline_llm_or_reward_imports(module):
    _, names = _imports(module)
    banned = ("dspy", "openai", "evolution.skills", "evolution.core", ".core",
              ".verifier", ".benchmark", ".evaluation", ".policies", "subprocess")
    assert not [n for n in names if n.startswith(banned)], names


def test_adapter_has_no_path_to_admission():
    tree, _ = _imports(gepa_adapter)
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "transition" not in attrs
    assert not {"ADMITTED"} & {n.id for n in ast.walk(tree)
                               if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    calls = {n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert {"propose", "register_baseline"} <= calls


# ── Admission requires a gate 6 decision (hardening, gate 6) ────────────────

@pytest.mark.parametrize("make_evidence, match", [
    (lambda cid: EVIDENCE, "admitted=true"),                       # arbitrary non-empty dict
    (lambda cid: {"gepa_score": 0.99, "split": "validation"}, "admitted=true"),
    (lambda cid: {**admitted_evidence(cid), "admitted": False}, "admitted=true"),
    (lambda cid: {**admitted_evidence(cid), "admitted": "yes"}, "admitted=true"),
    (lambda cid: {**admitted_evidence(cid), "policy_sha256": "0" * 64}, "registered"),
    (lambda cid: {k: v for k, v in admitted_evidence(cid).items() if k != "policy_sha256"},
     "registered"),
    (lambda cid: {**admitted_evidence(cid), "candidate_id": "code-review@000000000000"},
     "different candidate"),
])
def test_admission_rejects_evidence_that_is_not_a_gate6_decision(registry, make_evidence, match):
    _, cand = _proposed(registry)
    with pytest.raises(TransitionError, match=match):
        registry.transition(cand.candidate_id, ADMITTED, reason="r", decided_by="t",
                            evidence=make_evidence(cand.candidate_id))
    assert registry.get(cand.candidate_id).status == PROPOSED
    assert [t.to_status for t in registry.history(cand.candidate_id)] == [PROPOSED]


def test_admission_accepts_a_gate6_decision(registry):
    _, cand = _proposed(registry)
    registry.transition(cand.candidate_id, ADMITTED, reason="r", decided_by="t",
                        evidence=admitted_evidence(cand.candidate_id))
    assert registry.get(cand.candidate_id).status == ADMITTED


def test_nothing_can_be_admitted_without_a_registered_policy(registry, monkeypatch):
    monkeypatch.setattr(reg, "_ADMISSION_POLICIES", set())
    _, cand = _proposed(registry)
    with pytest.raises(TransitionError, match="no admission policy"):
        registry.transition(cand.candidate_id, ADMITTED, reason="r", decided_by="t",
                            evidence=admitted_evidence(cand.candidate_id))
    # Rejection needs no admission policy.
    registry.transition(cand.candidate_id, REJECTED, reason="r", decided_by="t",
                        evidence=EVIDENCE)


def test_registered_policy_is_the_admission_gate_policy():
    assert ADMISSION_POLICY.fingerprint() in reg.registered_admission_policies()
    with pytest.raises(ValueError):
        reg.register_admission_policy("not-a-sha")


def test_registry_does_not_import_the_admission_gate():
    _, names = _imports(reg)
    assert not [n for n in names if "admission" in n], names


# ── Evaluation-only strategies for proposed candidates ──────────────────────

def test_evaluation_strategy_for_proposed_candidate_never_admits(registry):
    base, cand = _proposed(registry)
    parent = Strategy("structured", system_prompt="ignored", toolsets=("todo",))
    strategy = reg.evaluation_strategy(registry, cand.candidate_id, parent)
    assert strategy.strategy_id == f"eval:structured:{cand.candidate_id}"
    assert strategy.system_prompt == EVOLVED.decode() and strategy.toolsets == ("todo",)
    assert registry.get(cand.candidate_id).status == PROPOSED
    assert len(registry.history(cand.candidate_id)) == 1
    with pytest.raises(RegistryError, match="only admitted"):
        candidate_strategy(registry, cand.candidate_id, parent)
    assert all(not s.strategy_id.startswith("eval:")
               for s in admitted_strategies(registry, "code-review", parent))
    assert reg.evaluation_strategy(registry, base.candidate_id, parent).strategy_id \
        == f"eval:structured:{base.candidate_id}"


@pytest.mark.parametrize("status", [REJECTED, RETIRED])
def test_evaluation_strategy_refuses_terminal_candidates(registry, status):
    _, cand = _proposed(registry)
    for step in ([REJECTED] if status == REJECTED else [ADMITTED, RETIRED]):
        registry.transition(cand.candidate_id, step, reason="r", decided_by="t",
                            evidence=evidence_for(cand.candidate_id, step))
    with pytest.raises(RegistryError, match="can be evaluated"):
        reg.evaluation_strategy(registry, cand.candidate_id, Strategy("structured"))
