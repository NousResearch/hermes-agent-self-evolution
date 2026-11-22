"""Offline tests for the labelled Python review benchmark (KAFO-EVO-001 gate 2)."""
import ast
import dataclasses
import hashlib
import json
import logging
import shutil

import pytest

from evolution.adaptive import benchmark as bm
from evolution.adaptive.benchmark import (
    BENCHMARK_ROOT, DEFECT_CATEGORIES, DIFFICULTIES, OPEN_SPLITS, SEVERITIES,
    SPLITS, IntegrityError, SchemaError, SealedSplitError, case_errors,
    code_line_count, compute_manifest, load_benchmark, load_split,
    render_manifest, schema_errors, verify_integrity,
)

VERSION_DIR = BENCHMARK_ROOT / "v1"


@pytest.fixture(scope="module")
def schema():
    return json.loads((BENCHMARK_ROOT / "schema.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def all_splits():
    """Every split, including the sealed one: integrity tests are not optimisation."""
    return load_benchmark(SPLITS, allow_sealed_test=True)


@pytest.fixture(scope="module")
def raw_cases():
    rows = []
    for split, rel in bm.SPLIT_FILES.items():
        text = (VERSION_DIR / rel).read_text(encoding="utf-8")
        rows += [json.loads(line) for line in text.splitlines()]
    return rows


@pytest.fixture
def copy_root(tmp_path):
    root = tmp_path / "python_review"
    shutil.copytree(BENCHMARK_ROOT, root)
    return root


def _repin(root):
    """Rewrite a copied manifest for its current files; return the new pin."""
    text = render_manifest(compute_manifest(root, "v1"))
    (root / "v1" / "MANIFEST.json").write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _valid_case():
    return {
        "case_id": "x-001", "split": "train", "language": "python",
        "code": "a = 1\nb = a / 0\n",
        "context": {"difficulty": "simple", "domain": "demo"},
        "defects": [{"defect_id": "x-001-d1", "category": "logic_error",
                     "line_start": 2, "line_end": 2, "severity": "low",
                     "description": "division by zero"}],
    }


# ── Integrity and sealing ────────────────────────────────────────────────────

def test_pinned_manifest_and_all_file_hashes_match():
    summary = verify_integrity()
    assert summary["splits"]["test"]["sealed"] is True
    assert {s: summary["splits"][s]["cases"] for s in SPLITS} == {
        "train": 28, "validation": 10, "test": 10}


def test_committed_manifest_is_reproducible():
    committed = (VERSION_DIR / "MANIFEST.json").read_bytes()
    assert render_manifest(compute_manifest()).encode("utf-8") == committed
    assert hashlib.sha256(committed).hexdigest() == bm.PINNED_MANIFEST_SHA256["v1"]


def test_test_split_refused_by_default():
    with pytest.raises(SealedSplitError):
        load_split("test")
    with pytest.raises(SealedSplitError):
        load_benchmark(("train", "test"))
    with pytest.raises(SealedSplitError):
        load_split("test", allow_sealed_test="yes")   # only literal True unseals
    assert issubclass(SealedSplitError, PermissionError)


def test_sealed_refusal_happens_before_any_file_access(tmp_path):
    with pytest.raises(SealedSplitError):
        load_split("test", root=tmp_path / "does-not-exist")


def test_open_splits_never_read_sealed_file(copy_root):
    (copy_root / "v1" / "sealed" / "test.jsonl").unlink()
    loaded = load_benchmark(root=copy_root)
    assert set(loaded) == set(OPEN_SPLITS)
    with pytest.raises(IntegrityError):
        verify_integrity(root=copy_root)   # the full check does notice


def test_unsealing_is_explicit_and_logged(caplog):
    with caplog.at_level(logging.WARNING, logger=bm.__name__):
        cases = load_split("test", allow_sealed_test=True)
    assert len(cases) == 10 and all(c.split == "test" for c in cases)
    assert "Unsealing" in caplog.text


@pytest.mark.parametrize("rel", ["v1/train.jsonl", "v1/validation.jsonl",
                                 "v1/sealed/test.jsonl", "schema.json"])
def test_tampering_with_any_file_is_detected(copy_root, rel):
    path = copy_root / rel
    data = path.read_bytes()
    path.write_bytes(data.replace(b'"high"', b'"low"', 1) if b'"high"' in data
                     else data.replace(b'"minimum": 1', b'"minimum": 0', 1))
    assert path.read_bytes() != data
    with pytest.raises(IntegrityError):
        verify_integrity(root=copy_root)
    split = next((s for s, f in bm.SPLIT_FILES.items() if rel.endswith(f)), "train")
    with pytest.raises(IntegrityError):
        load_split(split, root=copy_root, allow_sealed_test=True)


def test_rewriting_the_manifest_does_not_bypass_the_pin(copy_root):
    train = copy_root / "v1" / "train.jsonl"
    train.write_bytes(train.read_bytes().replace(b'"critical"', b'"low"', 1))
    _repin(copy_root)   # attacker updates the manifest to match
    with pytest.raises(IntegrityError, match="pinned"):
        load_split("train", root=copy_root)


def test_unknown_version_requires_explicit_pin(copy_root):
    shutil.copytree(copy_root / "v1", copy_root / "v9")
    with pytest.raises(IntegrityError, match="no pinned manifest"):
        load_split("train", root=copy_root, version="v9")


def test_manifest_paths_cannot_escape_root(copy_root):
    manifest = compute_manifest(copy_root, "v1")
    manifest["splits"]["train"]["path"] = "../outside.jsonl"
    text = render_manifest(manifest)
    (copy_root / "v1" / "MANIFEST.json").write_text(text, encoding="utf-8")
    pin = hashlib.sha256(text.encode("utf-8")).hexdigest()
    with pytest.raises(IntegrityError, match="escapes"):
        load_split("train", root=copy_root, expected_manifest_sha256=pin)


@pytest.mark.parametrize("bad_line", [
    '{"case_id": "a", "case_id": "b"}',   # duplicate key
    '{"value": NaN}',                     # non-standard constant
    "",                                   # blank line
])
def test_strict_jsonl_parsing(copy_root, bad_line):
    train = copy_root / "v1" / "train.jsonl"
    train.write_text(train.read_text(encoding="utf-8") + bad_line + "\n",
                     encoding="utf-8")
    pin = _repin(copy_root)
    with pytest.raises(SchemaError):
        load_split("train", root=copy_root, expected_manifest_sha256=pin)


def test_case_stored_in_wrong_split_file_is_rejected(copy_root):
    train = copy_root / "v1" / "train.jsonl"
    text = train.read_text(encoding="utf-8")
    train.write_text(text.replace('"split": "train"', '"split": "validation"', 1),
                     encoding="utf-8")
    pin = _repin(copy_root)
    with pytest.raises(SchemaError, match="stored in the 'train' split"):
        load_split("train", root=copy_root, expected_manifest_sha256=pin)


def test_case_leaking_across_splits_is_rejected(copy_root):
    first_train = (copy_root / "v1" / "train.jsonl").read_text(
        encoding="utf-8").splitlines()[0]
    leaked = json.loads(first_train)
    leaked["split"] = "validation"
    validation = copy_root / "v1" / "validation.jsonl"
    validation.write_text(validation.read_text(encoding="utf-8")
                          + json.dumps(leaked) + "\n", encoding="utf-8")
    pin = _repin(copy_root)
    with pytest.raises(IntegrityError, match="shared by"):
        load_benchmark(root=copy_root, expected_manifest_sha256=pin)


# ── Schema validity ──────────────────────────────────────────────────────────

def test_every_case_matches_schema(schema, raw_cases):
    assert len(raw_cases) == 48
    for obj in raw_cases:
        assert case_errors(obj, schema) == [], obj["case_id"]


def test_schema_enums_match_loader_constants(schema):
    props = schema["properties"]
    defect = props["defects"]["items"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"case_id", "split", "language", "code",
                                       "context", "defects"}
    assert tuple(props["split"]["enum"]) == SPLITS
    assert props["language"]["const"] == "python"
    assert tuple(props["context"]["properties"]["difficulty"]["enum"]) == DIFFICULTIES
    assert tuple(defect["properties"]["severity"]["enum"]) == SEVERITIES
    assert set(defect["required"]) == {"defect_id", "category", "line_start",
                                       "line_end", "severity", "description"}


def _mutate(path, value):
    def apply(obj):
        target = obj
        for key in path[:-1]:
            target = target[key]
        if value is KeyError:
            del target[path[-1]]
        else:
            target[path[-1]] = value
        return obj
    return apply


SCHEMA_VIOLATIONS = {
    "missing case_id": _mutate(["case_id"], KeyError),
    "extra property": _mutate(["notes"], "x"),
    "bad split": _mutate(["split"], "holdout"),
    "wrong language": _mutate(["language"], "javascript"),
    "empty code": _mutate(["code"], ""),
    "bad difficulty": _mutate(["context", "difficulty"], "hard"),
    "extra context key": _mutate(["context", "tier"], 1),
    "bad severity": _mutate(["defects", 0, "severity"], "blocker"),
    "bool line": _mutate(["defects", 0, "line_start"], True),
    "string line": _mutate(["defects", 0, "line_end"], "2"),
    "line zero": _mutate(["defects", 0, "line_start"], 0),
    "defects not list": _mutate(["defects"], {}),
}
CROSS_FIELD_VIOLATIONS = {
    "end before start": _mutate(["defects", 0, "line_start"], 3),
    "beyond snippet": _mutate(["defects", 0, "line_end"], 3),
    "duplicate defect id": lambda obj: (obj["defects"].append(
        dict(obj["defects"][0])), obj)[1],
}


@pytest.mark.parametrize("name", sorted(SCHEMA_VIOLATIONS) + sorted(CROSS_FIELD_VIOLATIONS))
def test_validator_rejects_invalid_cases(schema, name):
    assert case_errors(_valid_case(), schema) == []
    mutate = {**SCHEMA_VIOLATIONS, **CROSS_FIELD_VIOLATIONS}[name]
    assert case_errors(mutate(_valid_case()), schema), name
    with pytest.raises(SchemaError):
        bm.case_from_dict(mutate(_valid_case()), schema)


def test_validator_refuses_unsupported_schema_keywords():
    with pytest.raises(bm.BenchmarkError, match="unsupported"):
        schema_errors("x", {"type": "string", "pattern": "^x$"})


def test_jsonschema_library_agrees_when_installed(schema, raw_cases):
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    for obj in raw_cases:
        assert not list(validator.iter_errors(obj)), obj["case_id"]
    for name, mutate in SCHEMA_VIOLATIONS.items():
        assert list(validator.iter_errors(mutate(_valid_case()))), name


# ── Unique IDs and labels ────────────────────────────────────────────────────

def test_case_ids_unique_across_all_splits(all_splits):
    ids = [c.case_id for cases in all_splits.values() for c in cases]
    assert len(ids) == len(set(ids)) == 48


def test_defect_ids_unique_and_derived_from_case_id(all_splits):
    seen = set()
    for cases in all_splits.values():
        for case in cases:
            expected = [f"{case.case_id}-d{n}" for n in range(1, len(case.defects) + 1)]
            assert [d.defect_id for d in case.defects] == expected
            for defect in case.defects:
                assert defect.defect_id not in seen
                seen.add(defect.defect_id)
    assert len(seen) == 50


def test_labels_point_at_real_lines_of_parseable_code(all_splits):
    for cases in all_splits.values():
        for case in cases:
            ast.parse(case.code)
            assert "\r" not in case.code and "\t" not in case.code
            assert "# <D" not in case.code   # authoring markers were stripped
            lines = case.code.split("\n")
            for defect in case.defects:
                assert 1 <= defect.line_start <= defect.line_end <= code_line_count(case.code)
                span = lines[defect.line_start - 1:defect.line_end]
                assert any(line.strip() for line in span), defect.defect_id
            assert set(d.category for d in case.defects) <= set(DEFECT_CATEGORIES)


def test_code_line_count():
    assert code_line_count("") == 0
    assert code_line_count("a") == 1
    assert code_line_count("a\n") == 1
    assert code_line_count("a\nb") == 2
    assert code_line_count("a\n\n") == 2


# ── Split integrity ──────────────────────────────────────────────────────────

def test_split_membership_and_balance(all_splits):
    total = sum(len(cases) for cases in all_splits.values())
    assert 30 <= total <= 60
    for split, cases in all_splits.items():
        assert all(case.split == split for case in cases)
        assert any(case.is_clean for case in cases), split
        assert sum(not case.is_clean for case in cases) >= len(cases) // 2, split
        assert {case.context.difficulty for case in cases} == set(DIFFICULTIES), split
    assert 0.15 <= len(all_splits["test"]) / total <= 0.30


def test_no_case_or_snippet_shared_between_splits(all_splits):
    owner = {}
    for split, cases in all_splits.items():
        for case in cases:
            key = " ".join(case.code.split())
            assert owner.setdefault(key, split) == split, case.case_id
    assert len(owner) == 48


def test_open_load_returns_only_open_splits():
    loaded = load_benchmark()
    assert tuple(loaded) == OPEN_SPLITS
    assert {len(v) for v in loaded.values()} == {28, 10}


def test_cases_are_immutable_and_round_trip(raw_cases):
    train = load_split("train")
    assert isinstance(train, tuple) and isinstance(train[0].defects, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        train[0].code = "pass\n"
    by_id = {obj["case_id"]: obj for obj in raw_cases}
    for case in train:
        assert case.to_dict() == by_id[case.case_id]


# ── CLI ──────────────────────────────────────────────────────────────────────

def test_cli_verify_passes_without_printing_sealed_content(capsys, all_splits):
    assert bm.main(["verify"]) == 0
    out = capsys.readouterr().out
    assert "sealed (not parsed)" in out
    for case in all_splits["test"]:
        assert case.case_id not in out


def test_cli_verify_fails_on_tampering(copy_root, capsys):
    path = copy_root / "v1" / "validation.jsonl"
    path.write_bytes(path.read_bytes() + b" ")
    assert bm.main(["verify", "--root", str(copy_root)]) == 1
    assert "FAIL" in capsys.readouterr().err


def test_cli_manifest_writes_nothing(copy_root, capsys):
    before = {p: p.read_bytes() for p in copy_root.rglob("*") if p.is_file()}
    assert bm.main(["manifest", "--root", str(copy_root)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == compute_manifest(copy_root, "v1")
    assert bm.PINNED_MANIFEST_SHA256["v1"] in captured.err
    after = {p: p.read_bytes() for p in copy_root.rglob("*") if p.is_file()}
    assert before == after

