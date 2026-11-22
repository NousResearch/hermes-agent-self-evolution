"""Secondary LENIENT scoring and paired bootstrap for the one-time final evaluation.

STRICT scoring is the gate 3 verifier (``verifier.score_case``), unchanged:
a finding must have the label's exact (normalised) category. It is the
PRIMARY metric of the final evaluation.

LENIENT scoring (SECONDARY) is identical except for the category rule: a
finding is also category-compatible with a defect if its category is one of
the alternate valid categories that the independent label audit recorded for
that defect. The alternates are a scoring-side table,
``category_alternates_v1.json``, pinned by SHA-256 (``ALTERNATES_SHA256``).
The benchmark data, its labels and its manifest are not changed, and the
verifier module is not modified (its fingerprint is unchanged).

Line tolerance, overlap ratio, one-to-one maximum matching and the counting
conventions are the verifier's own (``verifier._overlap_ratio`` and
``verifier._prf``). With an empty table, :func:`score_case` equals
``verifier.score_case`` exactly (tested on every open case).

:func:`paired_bootstrap` gives the candidate-minus-baseline difference with a
percentile bootstrap that resamples the same case indices for both skills,
using the verifier's resampling scheme (cases sorted by ``case_id``,
``random.Random(seed)``, ``int(random() * n)``, interpolated quantiles).

Standard library only.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
from types import MappingProxyType
from typing import Mapping, Sequence

from . import verifier

ALTERNATES_SCHEMA = "kafo-category-alternates/1"
ALTERNATES_PATH = Path(__file__).resolve().with_name("category_alternates_v1.json")
ALTERNATES_SHA256 = "042fbfaca8018664de4b3f4b1a14a0831098d1cb3deb865aee2510bed7b06dbd"


class AlternatesError(ValueError):
    """The alternates table is missing, modified or malformed."""


def load_alternates(path: str | Path = ALTERNATES_PATH,
                    expected_sha256: str = ALTERNATES_SHA256
                    ) -> Mapping[str, frozenset[str]]:
    """defect_id -> alternate categories, after checking the pinned SHA-256."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise AlternatesError(f"{Path(path).name} does not match its pinned SHA-256")
    data = json.loads(raw.decode("utf-8"))
    if data.get("schema") != ALTERNATES_SCHEMA or not isinstance(data.get("alternates"), dict):
        raise AlternatesError("not a kafo-category-alternates/1 table")
    table = {}
    for defect_id, cats in data["alternates"].items():
        if not isinstance(defect_id, str) or not isinstance(cats, list) or not cats \
                or not all(isinstance(c, str) and c.strip() for c in cats):
            raise AlternatesError(f"bad alternates entry for {defect_id!r}")
        table[defect_id] = frozenset(cats)
    return MappingProxyType(table)


def alternates_provenance(path: str | Path = ALTERNATES_PATH) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"table": Path(path).name, "sha256": ALTERNATES_SHA256,
            "schema": data["schema"], "benchmark": data["benchmark"],
            "source": data["source"], "n_defects": len(data["alternates"])}


def score_case(case: verifier.BenchmarkCase, findings: Sequence,
               alternates: Mapping[str, frozenset[str]],
               config: verifier.MatchConfig = verifier.MatchConfig()) -> verifier.CaseResult:
    """``verifier.score_case`` with each defect also accepting its alternates."""
    if not isinstance(case, verifier.BenchmarkCase):
        raise TypeError("case must be a verifier.BenchmarkCase")
    if isinstance(findings, (str, bytes, Mapping)) or not isinstance(findings, Sequence):
        raise TypeError("findings must be a list of finding objects")
    parsed = [verifier.parse_finding(f) for f in findings]
    defects = case.defects
    accepted = [frozenset({config.canonical(d.category)}
                          | {config.canonical(a) for a in alternates.get(d.defect_id, ())})
                for d in defects]

    adjacency: list[list[int]] = []
    for f in parsed:
        options = []
        if f is not None:
            cat = config.canonical(f.category)
            for j, d in enumerate(defects):
                if cat not in accepted[j]:
                    continue
                ratio = verifier._overlap_ratio(f, d, config.line_tolerance)
                if ratio > 0 and ratio >= config.min_overlap_ratio:
                    options.append((-ratio, d.line_start, d.defect_id, j))
        adjacency.append([j for *_, j in sorted(options)])

    owner: dict[int, int] = {}

    def augment(i: int, visited: set[int]) -> bool:
        for j in adjacency[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in owner or augment(owner[j], visited):
                owner[j] = i
                return True
        return False

    for i in range(len(parsed)):
        if adjacency[i]:
            augment(i, set())

    matched = {i: j for j, i in owner.items()}
    matches = tuple(sorted((i, defects[j].defect_id) for i, j in matched.items()))
    fp_indices = tuple(i for i in range(len(parsed)) if i not in matched)
    missed = tuple(d.defect_id for j, d in enumerate(defects) if j not in owner)
    tp, fp, fn = len(matches), len(fp_indices), len(missed)
    precision, recall, f1 = verifier._prf(tp, fp, fn)
    return verifier.CaseResult(
        case_id=case.case_id, split=case.split,
        true_positives=tp, false_positives=fp, false_negatives=fn,
        precision=precision, recall=recall, f1=f1,
        matches=matches, missed_defects=missed, false_positive_findings=fp_indices,
        duplicate_findings=sum(1 for i in fp_indices if adjacency[i]),
        invalid_findings=sum(1 for f in parsed if f is None))


def _stats(results: Sequence[verifier.CaseResult]) -> dict[str, float]:
    tp = sum(r.true_positives for r in results)
    fp = sum(r.false_positives for r in results)
    fn = sum(r.false_negatives for r in results)
    p, r, f = verifier._prf(tp, fp, fn)
    return {"micro_f1": f, "micro_precision": p, "micro_recall": r,
            "macro_f1": math.fsum(x.f1 for x in results) / len(results)}


def _quantile(values: Sequence[float], q: float) -> float:
    pos = q * (len(values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def paired_bootstrap(baseline: Sequence[verifier.CaseResult],
                     candidate: Sequence[verifier.CaseResult], *, seed: int,
                     n_bootstrap: int, confidence: float = 0.95) -> dict:
    """Candidate minus baseline, with a paired percentile bootstrap over cases."""
    a = sorted(baseline, key=lambda r: r.case_id)
    b = sorted(candidate, key=lambda r: r.case_id)
    if not a or [r.case_id for r in a] != [r.case_id for r in b]:
        raise ValueError("paired results need the same nonempty set of case_ids")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive int")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    rng, n = random.Random(seed), len(a)
    samples: dict[str, list[float]] = {k: [] for k in _stats(a)}
    for _ in range(n_bootstrap):
        idx = [int(rng.random() * n) for _ in range(n)]
        sa, sb = _stats([a[i] for i in idx]), _stats([b[i] for i in idx])
        for k in samples:
            samples[k].append(sb[k] - sa[k])
    full_a, full_b = _stats(a), _stats(b)
    alpha = 1.0 - confidence
    out = {}
    for k, values in samples.items():
        values.sort()
        out[k] = {"point": full_b[k] - full_a[k], "low": _quantile(values, alpha / 2),
                  "high": _quantile(values, 1 - alpha / 2)}
    return out
