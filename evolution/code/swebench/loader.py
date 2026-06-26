"""Load SWE-bench Lite and reduce to single-file repair organisms. The raw HF row
is retained on each SWEInstance because make_test_spec must receive it verbatim
(reconstructing a partial dict risks KeyError on required fields). patch_loc /
patch_hunks feed the honesty report's difficulty profile."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

_PLUS_FILE = re.compile(r"^\+\+\+ b/(.+?)\s*$", re.M)
_HUNK = re.compile(r"^@@ ", re.M)


def _is_test_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return (path.startswith("tests/") or "/tests/" in path or name == "conftest.py"
            or (name.startswith("test_") and name.endswith(".py")) or name.endswith("_test.py"))


def files_in_patch(patch: str) -> list[str]:
    return [p for p in _PLUS_FILE.findall(patch) if p != "/dev/null"]


def is_single_file(instance: dict) -> bool:
    return len([p for p in files_in_patch(instance["patch"]) if not _is_test_file(p)]) == 1


def patch_loc(patch: str) -> int:
    """Added+removed source lines in the diff (excludes +++/--- headers)."""
    n = 0
    for ln in patch.splitlines():
        if ln.startswith(("+++", "---")):
            continue
        if ln.startswith(("+", "-")):
            n += 1
    return n


def patch_hunks(patch: str) -> int:
    return len(_HUNK.findall(patch))


def _as_list(v) -> tuple[str, ...]:
    return tuple(json.loads(v)) if isinstance(v, str) else tuple(v)


@dataclass(frozen=True)
class SWEInstance:
    instance_id: str
    repo: str
    base_commit: str
    version: str
    gold_patch: str
    test_patch: str
    gold_file: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    raw: dict  # the verbatim HF row, for make_test_spec
    problem_statement: str = ""


def to_instance(row: dict) -> SWEInstance:
    nontest = [p for p in files_in_patch(row["patch"]) if not _is_test_file(p)]
    if len(nontest) != 1:
        raise ValueError(f"{row['instance_id']} is not single-file: {nontest}")
    return SWEInstance(
        instance_id=row["instance_id"], repo=row["repo"], base_commit=row["base_commit"],
        version=str(row["version"]), gold_patch=row["patch"], test_patch=row["test_patch"],
        gold_file=nontest[0], fail_to_pass=_as_list(row["FAIL_TO_PASS"]),
        pass_to_pass=_as_list(row["PASS_TO_PASS"]), raw=row,
        problem_statement=row.get("problem_statement", ""))


def load_single_file_lite(*, split: str = "test", limit: int | None = None) -> list[SWEInstance]:
    """All single-file Lite instances (optionally capped). Requires the [swebench] extra."""
    from datasets import load_dataset  # noqa: PLC0415
    ds = load_dataset("SWE-bench/SWE-bench_Lite", split=split)
    out: list[SWEInstance] = []
    for row in ds:
        if is_single_file(row):
            out.append(to_instance(dict(row)))
            if limit is not None and len(out) >= limit:
                break
    return out
