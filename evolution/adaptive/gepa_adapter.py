"""Register GEPA skill candidates in the candidate registry as ``proposed``.

``python -m evolution.skills.evolve_skill`` writes, per run,
``output/<skill>/<timestamp>/`` containing ``evolved_skill.md``,
``baseline_skill.md`` and ``metrics.json``. When the evolved skill fails
its constraints it writes ``output/<skill>/evolved_FAILED.md`` instead (no
baseline or metrics next to it); pass the baseline bytes explicitly then.

This adapter reads those files (read-only, no symlinks, size-capped),
registers the baseline if it is not known yet, and proposes the evolved
skill with the baseline as parent. It never admits, rejects or deploys a
candidate, and does not import the GEPA pipeline (no dspy, no LLM calls).

GEPA's own scores are copied into ``provenance`` under ``gepa_reported``
and labelled ``unverified``: they come from GEPA's LLM-judged evaluation,
not from the independent verifier, and must not decide admission.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from .registry import (
    ADMITTED, GEPA, Candidate, CandidateRegistry, RegistryError, candidate_id_for,
    content_sha256,
)

EVOLVED, FAILED, BASELINE_FILE, METRICS = (
    "evolved_skill.md", "evolved_FAILED.md", "baseline_skill.md", "metrics.json")
MAX_SKILL_BYTES = 1_000_000
ADAPTER_VERSION = "gepa-adapter/1"
_METRIC_KEYS = ("baseline_score", "evolved_score", "improvement", "iterations",
                "optimizer_model", "eval_model", "train_examples", "val_examples",
                "holdout_examples", "constraints_passed", "timestamp")
_NAME_LINE = re.compile(rb"^name:\s*['\"]?([^'\"\r\n]+?)['\"]?\s*$", re.MULTILINE)


class GepaOutputError(RegistryError, ValueError):
    """The GEPA output directory is missing files or malformed."""


@dataclass(frozen=True)
class GepaOutput:
    skill_name: str
    evolved: bytes
    baseline: bytes
    constraints_passed: bool | None   # None when GEPA did not report it
    metrics: dict
    source_dir: str


def _read(path: Path) -> bytes:
    if path.is_symlink():
        raise GepaOutputError(f"{path}: symlinks are not accepted")
    if not path.is_file():
        raise GepaOutputError(f"{path}: missing")
    if path.stat().st_size > MAX_SKILL_BYTES:
        raise GepaOutputError(f"{path}: larger than {MAX_SKILL_BYTES} bytes")
    with open(path, "rb") as fh:
        data = fh.read(MAX_SKILL_BYTES + 1)
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GepaOutputError(f"{path}: not UTF-8") from exc
    return data


def _frontmatter_name(content: bytes) -> str | None:
    if not content.startswith(b"---"):
        return None
    end = content.find(b"\n---", 3)
    match = _NAME_LINE.search(content[: end if end != -1 else len(content)])
    return match.group(1).decode("utf-8").strip() if match else None


def _clean_metrics(raw) -> dict:
    if not isinstance(raw, dict):
        raise GepaOutputError("metrics.json must hold a JSON object")
    out = {}
    for key in _METRIC_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, float) and not math.isfinite(value):
            raise GepaOutputError(f"metrics.json {key} is not finite")
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise GepaOutputError(f"metrics.json {key} must be a scalar")
        out[key] = value
    return out


def read_gepa_output(output_dir: str | Path, *, skill_name: str | None = None,
                     baseline: bytes | None = None) -> GepaOutput:
    """Read one GEPA run directory without modifying it."""
    root = Path(output_dir)
    if not root.is_dir() or root.is_symlink():
        raise GepaOutputError(f"{root}: not a directory")
    metrics: dict = {}
    if (root / METRICS).exists():
        try:
            metrics_raw = json.loads(_read(root / METRICS),
                                     parse_constant=lambda c: float("nan"))
        except json.JSONDecodeError as exc:
            raise GepaOutputError(f"{root / METRICS}: invalid JSON: {exc}") from exc
        metrics = _clean_metrics(metrics_raw)
        reported_name = metrics_raw.get("skill_name")
    else:
        reported_name = None

    has_evolved, has_failed = (root / EVOLVED).exists(), (root / FAILED).exists()
    if has_evolved == has_failed:
        raise GepaOutputError(f"{root}: expected exactly one of {EVOLVED} or {FAILED}")
    evolved = _read(root / (EVOLVED if has_evolved else FAILED))
    if baseline is None:
        baseline = _read(root / BASELINE_FILE)
    elif not isinstance(baseline, bytes):
        raise TypeError("baseline must be bytes")

    constraints = metrics.get("constraints_passed")
    if has_failed:
        constraints = False
    elif not isinstance(constraints, bool):
        constraints = None

    names = {n for n in (skill_name, reported_name, _frontmatter_name(baseline)) if n}
    if len(names) != 1:
        raise GepaOutputError(f"{root}: skill name unknown or inconsistent: {sorted(names)}")
    if evolved == baseline:
        raise GepaOutputError(f"{root}: evolved skill is identical to the baseline")
    return GepaOutput(names.pop(), evolved, baseline, constraints, metrics, str(root))


def register_gepa_output(registry: CandidateRegistry, output_dir: str | Path, *,
                         skill_name: str | None = None,
                         baseline: bytes | None = None) -> Candidate:
    """Register a GEPA run's evolved skill as a ``proposed`` candidate.

    The baseline is registered (as an admitted baseline) only if this exact
    content is not in the registry yet. There is no way to request another
    status: admission is a separate, evidence-backed step (gate 6).
    """
    out = read_gepa_output(output_dir, skill_name=skill_name, baseline=baseline)
    parent_id = candidate_id_for(out.skill_name, content_sha256(out.baseline))
    try:
        parent = registry.get(parent_id)   # e.g. GEPA evolved an admitted candidate
    except KeyError:
        parent = registry.register_baseline(
            out.skill_name, out.baseline,
            provenance={"origin": "gepa baseline_skill.md", "source_dir": out.source_dir,
                        "adapter": ADAPTER_VERSION})
    if parent.status != ADMITTED:
        raise RegistryError(f"GEPA baseline {parent.candidate_id} is {parent.status}; "
                            "candidates must descend from an admitted skill")
    return registry.propose(
        out.skill_name, out.evolved, parent_id=parent.candidate_id, source=GEPA,
        provenance={"adapter": ADAPTER_VERSION, "source_dir": out.source_dir,
                    "constraints_passed": out.constraints_passed,
                    "gepa_reported": out.metrics,
                    "gepa_reported_status": "unverified: GEPA LLM-judged scores, "
                                            "not the independent verifier",
                    "data_label": "synthetic"})
