"""Read-only loader for the labelled Python code-review benchmark (KAFO-EVO-001).

Layout under ``datasets/python_review/``::

    schema.json               JSON Schema for one case (one JSON object per JSONL line)
    <version>/MANIFEST.json   SHA-256 of the schema and of every split file
    <version>/train.jsonl     open split: may be used during optimisation
    <version>/validation.jsonl  open split: model / policy selection
    <version>/sealed/test.jsonl sealed split: final evaluation only

Integrity: the SHA-256 of each version's MANIFEST.json is pinned in this module
and the manifest pins the schema and every split file. Any edit to cases,
labels or schema is therefore detected at load time unless the pin is changed
in reviewed code.

Sealing: ``load_split("test")`` raises :class:`SealedSplitError` unless the
caller passes ``allow_sealed_test=True``. Loading open splits never opens the
sealed file.

Separation during optimisation: code that optimises or selects (the adaptive
loop, policy comparison) runs inside :func:`optimisation_phase`. While any
optimisation phase is active in the process, the sealed split cannot be
loaded even with ``allow_sealed_test=True``, and a Python audit hook makes any
``open()`` of a file under a ``python_review/.../sealed/`` directory raise
:class:`SealedSplitError`. The hook covers this process only, not
subprocesses.

The snippets are hand-written synthetic examples with seeded defects. A score
on this benchmark is not evidence that Hermes improves on real code review.

Dependency-free (standard library only) and never writes to disk.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)

BENCHMARK_NAME = "python_review"
BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "datasets" / BENCHMARK_NAME
DEFAULT_VERSION = "v1"
SCHEMA_FILE = "schema.json"
MANIFEST_FILE = "MANIFEST.json"

SPLITS = ("train", "validation", "test")
OPEN_SPLITS = ("train", "validation")
SEALED_SPLITS = ("test",)
# Split file locations relative to the version directory.
SPLIT_FILES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "sealed/test.jsonl",
}
DIFFICULTIES = ("simple", "moderate", "complex")
SEVERITIES = ("low", "medium", "high", "critical")
# Category vocabulary used by v1. The schema keeps ``category`` a free string;
# this tuple documents the labels a verifier can expect.
DEFECT_CATEGORIES = (
    "security", "logic_error", "error_handling", "input_validation",
    "resource_management", "concurrency", "performance", "api_misuse",
)

# SHA-256 of datasets/python_review/<version>/MANIFEST.json. Changing any case,
# label or the schema changes the manifest, which then needs a new pin here.
PINNED_MANIFEST_SHA256 = {
    "v1": "fada5f14b7d89c0e081f820475ce52a51ecc585abfe2264de09f38c69cfb7282",
}


class BenchmarkError(Exception):
    """Base class for benchmark loading failures."""


class IntegrityError(BenchmarkError):
    """A file does not match its pinned SHA-256, or the manifest is invalid."""


class SchemaError(BenchmarkError, ValueError):
    """A case does not satisfy the case schema or its cross-field rules."""


class SealedSplitError(BenchmarkError, PermissionError):
    """The sealed test split was requested without explicit unsealing."""


@dataclass(frozen=True)
class Defect:
    defect_id: str
    category: str
    line_start: int   # 1-based, inclusive
    line_end: int     # 1-based, inclusive
    severity: str
    description: str


@dataclass(frozen=True)
class CaseContext:
    difficulty: str
    domain: str


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    split: str
    language: str
    code: str
    context: CaseContext
    defects: tuple[Defect, ...]

    @property
    def is_clean(self) -> bool:
        """True for clean control cases (no seeded defects)."""
        return not self.defects

    def to_dict(self) -> dict:
        """Return the case in the on-disk schema shape."""
        return {
            "case_id": self.case_id,
            "split": self.split,
            "language": self.language,
            "code": self.code,
            "context": {"difficulty": self.context.difficulty,
                        "domain": self.context.domain},
            "defects": [
                {"defect_id": d.defect_id, "category": d.category,
                 "line_start": d.line_start, "line_end": d.line_end,
                 "severity": d.severity, "description": d.description}
                for d in self.defects
            ],
        }


def code_line_count(code: str) -> int:
    """Number of lines in ``code``, counted on '\\n' (a final newline adds none)."""
    if not code:
        return 0
    return code.count("\n") + (0 if code.endswith("\n") else 1)


# ── Minimal JSON Schema subset ───────────────────────────────────────────────
# Enough of draft 2020-12 to enforce schema.json without third-party packages.
# Any keyword outside this subset raises, so the schema file cannot silently
# rely on a rule this validator ignores.

_ANNOTATION_KEYWORDS = frozenset({"$schema", "$id", "$comment", "title", "description"})
_ASSERTION_KEYWORDS = frozenset({
    "type", "required", "properties", "additionalProperties",
    "enum", "const", "minLength", "minimum", "items",
})
_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
}


def _json_equal(a, b) -> bool:
    return isinstance(a, bool) == isinstance(b, bool) and a == b


def schema_errors(instance, schema: dict, path: str = "$") -> list[str]:
    """Validate ``instance`` against the supported JSON Schema subset."""
    unsupported = set(schema) - _ANNOTATION_KEYWORDS - _ASSERTION_KEYWORDS
    if unsupported:
        raise BenchmarkError(
            f"schema uses unsupported keywords at {path}: {sorted(unsupported)}")
    expected = schema.get("type")
    if expected is not None:
        if expected not in _TYPE_CHECKS:
            raise BenchmarkError(f"schema uses unsupported type {expected!r} at {path}")
        if not _TYPE_CHECKS[expected](instance):
            return [f"{path}: expected {expected}, got {type(instance).__name__}"]
    errors = []
    if "const" in schema and not _json_equal(instance, schema["const"]):
        errors.append(f"{path}: must equal {schema['const']!r}")
    if "enum" in schema and not any(_json_equal(instance, e) for e in schema["enum"]):
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']}")
    if isinstance(instance, str) and len(instance) < schema.get("minLength", 0):
        errors.append(f"{path}: shorter than {schema['minLength']} characters")
    if ("minimum" in schema and _TYPE_CHECKS["integer"](instance)
            and instance < schema["minimum"]):
        errors.append(f"{path}: {instance} is below the minimum {schema['minimum']}")
    if isinstance(instance, dict):
        additional = schema.get("additionalProperties", True)
        if additional not in (True, False):
            raise BenchmarkError(f"schema additionalProperties must be a bool at {path}")
        for key in schema.get("required", ()):
            if key not in instance:
                errors.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        for key, value in instance.items():
            if key in properties:
                errors.extend(schema_errors(value, properties[key], f"{path}.{key}"))
            elif additional is False:
                errors.append(f"{path}: unexpected property {key!r}")
    if isinstance(instance, list) and "items" in schema:
        for index, item in enumerate(instance):
            errors.extend(schema_errors(item, schema["items"], f"{path}[{index}]"))
    return errors


def case_errors(obj, schema: dict) -> list[str]:
    """Schema errors plus the cross-field rules JSON Schema cannot express."""
    errors = schema_errors(obj, schema)
    if errors:
        return errors
    lines = code_line_count(obj["code"])
    seen = set()
    for index, defect in enumerate(obj["defects"]):
        path = f"$.defects[{index}]"
        if defect["defect_id"] in seen:
            errors.append(f"{path}: duplicate defect_id {defect['defect_id']!r}")
        seen.add(defect["defect_id"])
        if defect["line_end"] < defect["line_start"]:
            errors.append(f"{path}: line_end is before line_start")
        if defect["line_end"] > lines:
            errors.append(f"{path}: line_end {defect['line_end']} is beyond the "
                          f"{lines}-line snippet")
    return errors


def case_from_dict(obj, schema: dict) -> BenchmarkCase:
    """Validate one decoded case and return it as an immutable object."""
    errors = case_errors(obj, schema)
    if errors:
        case_id = obj.get("case_id") if isinstance(obj, dict) else None
        raise SchemaError(f"case {case_id!r} is invalid: " + "; ".join(errors))
    return BenchmarkCase(
        case_id=obj["case_id"], split=obj["split"], language=obj["language"],
        code=obj["code"],
        context=CaseContext(**obj["context"]),
        defects=tuple(Defect(**d) for d in obj["defects"]),
    )


# ── Integrity ────────────────────────────────────────────────────────────────

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_constant(name: str):
    raise ValueError(f"non-standard JSON constant {name}")


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _strict_json(text: str):
    return json.loads(text, object_pairs_hook=_unique_keys,
                      parse_constant=_reject_constant)


def _root(root) -> Path:
    return Path(root) if root is not None else BENCHMARK_ROOT


def _read_inside(root: Path, rel_path: str) -> bytes:
    base = root.resolve()
    path = (base / rel_path).resolve()
    if not path.is_relative_to(base):
        raise IntegrityError(f"manifest path {rel_path!r} escapes the benchmark root")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise IntegrityError(f"cannot read {rel_path}: {exc}") from exc


def _read_manifest(root: Path, version: str,
                   expected_manifest_sha256: str | None) -> dict:
    pin = expected_manifest_sha256 or PINNED_MANIFEST_SHA256.get(version)
    if pin is None:
        raise IntegrityError(f"no pinned manifest hash for version {version!r}; "
                             "pass expected_manifest_sha256 explicitly")
    raw = _read_inside(root, f"{version}/{MANIFEST_FILE}")
    if _sha256(raw) != pin:
        raise IntegrityError(f"{version}/{MANIFEST_FILE} does not match its pinned "
                             "SHA-256; the benchmark manifest was modified")
    try:
        manifest = _strict_json(raw.decode("utf-8"))
        ok = (manifest["benchmark"] == BENCHMARK_NAME
              and manifest["version"] == version
              and manifest["hash_algorithm"] == "sha256"
              and isinstance(manifest["schema"]["sha256"], str)
              and set(manifest["splits"]) == set(SPLITS)
              and all(manifest["splits"][s]["sealed"] is (s in SEALED_SPLITS)
                      and isinstance(manifest["splits"][s]["cases"], int)
                      and isinstance(manifest["splits"][s]["sha256"], str)
                      for s in SPLITS))
    except (ValueError, KeyError, TypeError) as exc:
        raise IntegrityError(f"malformed manifest: {exc}") from exc
    if not ok:
        raise IntegrityError("manifest does not describe this benchmark version")
    return manifest


def _read_verified(root: Path, entry: dict, label: str) -> bytes:
    data = _read_inside(root, entry["path"])
    if _sha256(data) != entry["sha256"]:
        raise IntegrityError(f"{label} ({entry['path']}) does not match the manifest "
                             "SHA-256; benchmark data was modified")
    return data


def _load_schema(root: Path, manifest: dict) -> dict:
    raw = _read_verified(root, manifest["schema"], "schema")
    try:
        return _strict_json(raw.decode("utf-8"))
    except ValueError as exc:
        raise IntegrityError(f"schema is not valid JSON: {exc}") from exc


def _parse_jsonl(data: bytes, split: str, schema: dict,
                 source: str) -> tuple[BenchmarkCase, ...]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SchemaError(f"{source} is not UTF-8: {exc}") from exc
    if text and not text.endswith("\n"):
        raise SchemaError(f"{source} must end with a newline")
    cases, seen = [], set()
    for number, line in enumerate(text.split("\n")[:-1], start=1):
        if not line.strip():
            raise SchemaError(f"{source}:{number}: blank line")
        try:
            obj = _strict_json(line)
        except ValueError as exc:
            raise SchemaError(f"{source}:{number}: invalid JSON: {exc}") from exc
        case = case_from_dict(obj, schema)
        if case.split != split:
            raise SchemaError(f"{source}:{number}: case {case.case_id!r} is labelled "
                              f"{case.split!r} but stored in the {split!r} split")
        if case.case_id in seen:
            raise SchemaError(f"{source}:{number}: duplicate case_id {case.case_id!r}")
        seen.add(case.case_id)
        cases.append(case)
    return tuple(cases)


# ── Optimisation phase guard ─────────────────────────────────────────────────

_phase_lock = threading.Lock()
_phase_depth = 0
_sealed_attempts = 0
_hook_installed = False


def _is_sealed_path(path: str) -> bool:
    parts = Path(os.path.realpath(path)).parts
    if BENCHMARK_NAME not in parts:
        return False
    return "sealed" in parts[parts.index(BENCHMARK_NAME) + 1:]


def _audit_hook(event: str, args: tuple) -> None:
    global _sealed_attempts
    if event != "open" or not _phase_depth or not args:
        return
    target = args[0]
    if not isinstance(target, (str, bytes, os.PathLike)):
        return   # file descriptors and other targets
    if _is_sealed_path(os.fsdecode(target)):
        with _phase_lock:
            _sealed_attempts += 1
        raise SealedSplitError("sealed benchmark data opened during an optimisation "
                               f"phase: {os.fsdecode(target)}")


def _install_hook() -> None:
    global _hook_installed
    with _phase_lock:
        if not _hook_installed:
            sys.addaudithook(_audit_hook)   # audit hooks cannot be removed
            _hook_installed = True


@contextmanager
def optimisation_phase() -> Iterator[None]:
    """Mark the process as optimising or selecting (re-entrant, thread-safe).

    Inside it the sealed split cannot be loaded or opened in this process.
    """
    global _phase_depth
    _install_hook()
    with _phase_lock:
        _phase_depth += 1
    try:
        yield
    finally:
        with _phase_lock:
            _phase_depth -= 1


def in_optimisation_phase() -> bool:
    return _phase_depth > 0


def sealed_access_attempts() -> int:
    """How many opens of sealed files the guard has blocked in this process."""
    return _sealed_attempts


# ── Public API ───────────────────────────────────────────────────────────────

def _check_request(splits: tuple[str, ...], allow_sealed_test) -> None:
    for split in splits:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        if split in SEALED_SPLITS and allow_sealed_test is not True:
            raise SealedSplitError(
                f"the {split!r} split is sealed for final evaluation; pass "
                "allow_sealed_test=True only outside optimisation and selection")
        if split in SEALED_SPLITS and in_optimisation_phase():
            raise SealedSplitError(
                f"the {split!r} split cannot be loaded during an optimisation phase")


def load_split(split: str, *, version: str = DEFAULT_VERSION, root=None,
               allow_sealed_test: bool = False,
               expected_manifest_sha256: str | None = None,
               ) -> tuple[BenchmarkCase, ...]:
    """Load and verify one split. The sealed split needs allow_sealed_test=True."""
    _check_request((split,), allow_sealed_test)
    base = _root(root)
    manifest = _read_manifest(base, version, expected_manifest_sha256)
    schema = _load_schema(base, manifest)
    entry = manifest["splits"][split]
    data = _read_verified(base, entry, f"{split} split")
    if split in SEALED_SPLITS:
        logger.warning("Unsealing %s %s %r split; results must not feed back into "
                       "optimisation or selection", BENCHMARK_NAME, version, split)
    cases = _parse_jsonl(data, split, schema, entry["path"])
    if len(cases) != entry["cases"]:
        raise IntegrityError(f"{split} split has {len(cases)} cases; manifest "
                             f"records {entry['cases']}")
    return cases


def load_benchmark(splits: Iterable[str] = OPEN_SPLITS, *,
                   version: str = DEFAULT_VERSION, root=None,
                   allow_sealed_test: bool = False,
                   expected_manifest_sha256: str | None = None,
                   ) -> dict[str, tuple[BenchmarkCase, ...]]:
    """Load several splits (open splits by default) and check they are disjoint."""
    splits = tuple(splits)
    if len(set(splits)) != len(splits):
        raise ValueError("splits must not repeat")
    _check_request(splits, allow_sealed_test)   # before any file is read
    loaded = {split: load_split(split, version=version, root=root,
                                allow_sealed_test=allow_sealed_test,
                                expected_manifest_sha256=expected_manifest_sha256)
              for split in splits}
    owners: dict[tuple[str, str], str] = {}
    for split, cases in loaded.items():
        for case in cases:
            keys = [("case_id", case.case_id), ("code", case.code)]
            keys += [("defect_id", d.defect_id) for d in case.defects]
            for key in keys:
                if key in owners:
                    raise IntegrityError(f"{key[0]} shared by the {owners[key]!r} and "
                                         f"{split!r} splits (case {case.case_id!r})")
                owners[key] = split
    return loaded


def verify_integrity(*, version: str = DEFAULT_VERSION, root=None,
                     expected_manifest_sha256: str | None = None) -> dict:
    """Check the pinned manifest and the hash of the schema and every split file.

    The sealed split is hashed but never parsed; counts come from the manifest.
    """
    base = _root(root)
    manifest = _read_manifest(base, version, expected_manifest_sha256)
    _read_verified(base, manifest["schema"], "schema")
    for split in SPLITS:
        _read_verified(base, manifest["splits"][split], f"{split} split")
    return {"benchmark": BENCHMARK_NAME, "version": version,
            "splits": {s: {"cases": manifest["splits"][s]["cases"],
                           "sealed": s in SEALED_SPLITS} for s in SPLITS}}


def compute_manifest(root=None, version: str = DEFAULT_VERSION) -> dict:
    """Build the manifest for the files currently on disk (maintainer tooling).

    Pure computation: it prints nothing and writes nothing.
    """
    base = _root(root)
    schema_bytes = _read_inside(base, SCHEMA_FILE)
    splits = {}
    for split in SPLITS:
        rel = f"{version}/{SPLIT_FILES[split]}"
        data = _read_inside(base, rel)
        lines = data.decode("utf-8").split("\n")
        splits[split] = {"path": rel, "sha256": _sha256(data),
                         "cases": sum(1 for line in lines if line.strip()),
                         "sealed": split in SEALED_SPLITS}
    return {"benchmark": BENCHMARK_NAME, "version": version,
            "hash_algorithm": "sha256",
            "schema": {"path": SCHEMA_FILE, "sha256": _sha256(schema_bytes)},
            "splits": splits}


def render_manifest(manifest: dict) -> str:
    """Canonical on-disk text of a manifest (its SHA-256 is what gets pinned)."""
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evolution.adaptive.benchmark",
        description="Read-only checks for the Python review benchmark.")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="check every file hash and validate the "
                                           "open splits (the sealed split is only hashed)")
    manifest = sub.add_parser("manifest", help="print a freshly computed manifest and "
                                               "its pin to stdout; writes nothing")
    for command in (verify, manifest):
        command.add_argument("--version", default=DEFAULT_VERSION)
        command.add_argument("--root", type=Path, default=BENCHMARK_ROOT)
    args = parser.parse_args(argv)

    if args.command == "manifest":
        text = render_manifest(compute_manifest(args.root, args.version))
        sys.stdout.write(text)
        print(f"# pin: {_sha256(text.encode('utf-8'))}", file=sys.stderr)
        return 0

    try:
        summary = verify_integrity(version=args.version, root=args.root)
        opened = load_benchmark(OPEN_SPLITS, version=args.version, root=args.root)
    except BenchmarkError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    for split, info in summary["splits"].items():
        if split in opened:
            cases = opened[split]
            clean = sum(c.is_clean for c in cases)
            defects = sum(len(c.defects) for c in cases)
            print(f"{split}: {len(cases)} cases ({clean} clean), {defects} defects, "
                  "hash OK, schema OK")
        else:
            print(f"{split}: {info['cases']} cases, hash OK, sealed (not parsed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
