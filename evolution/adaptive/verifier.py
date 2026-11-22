"""Independent, deterministic verifier for Python code-review benchmark cases.

Scores a model's review findings against ground-truth defects and returns
precision, recall and F1 per case, plus micro and macro aggregates with
seeded bootstrap confidence intervals. No LLM judge is used.

Independence
------------
This module imports only the Python standard library. It must never import
or call executor, policy or reward code (``evolution.adaptive.core``,
``evolution.adaptive.cli``) and it treats model output purely as data.
Run it in a separate process from the agent under test, from a checkout the
agent cannot write to::

    python -m evolution.adaptive.verifier --cases cases.jsonl \\
        --reviews reviews.jsonl --seed 0

Every report carries ``verifier_sha256``, the SHA-256 of this file, so any
score can be audited against the reviewed version of the scoring rules.

Benchmark case schema (one JSON object per JSONL line)
------------------------------------------------------
``{case_id, split, language, code, context: {difficulty, domain},
defects: [{defect_id, category, line_start, line_end, severity,
description}]}``. Line numbers are 1-based and inclusive, and must fall
inside ``code``; lines are counted on ``\n`` exactly as in
``evolution.adaptive.benchmark.code_line_count``. Cases are trusted
evaluation data and are validated strictly: a malformed case raises
``ValueError``.

Sealed split
------------
``load_cases`` (and so the CLI) refuses any case whose split is in
``SEALED_SPLITS``: sealed test cases are never read from a file by the
verifier. Final evaluation on the sealed split goes through
``evolution.adaptive.evaluation.evaluate_sealed_test``, which loads it with
the integrity-checked benchmark loader and passes the cases in memory.

Review schema
-------------
A review is a list of findings ``{line_start, line_end, category,
description?}``. ``line_end`` defaults to ``line_start``. Reviews are
untrusted model output: a malformed finding does not raise, it is counted
as an invalid finding and scored as a false positive. The CLI reads
reviews as JSONL ``{case_id, findings: [...]}``.

Matching rule (``MATCH_RULE``)
------------------------------
1. Categories are normalised: trimmed, lower-cased, spaces and hyphens
   become underscores, then optional ``category_aliases`` are applied to
   both sides. Categories must be equal after normalisation.
2. The defect's line range is widened by ``line_tolerance`` lines (default
   1) on each side. At least one line of the finding must fall inside it,
   and at least ``min_overlap_ratio`` (default 0.5) of the finding's own
   lines must. A finding that spans the whole file therefore cannot match
   a small defect.
3. A finding that satisfies 1 and 2 is *compatible* with that defect.
   Findings are assigned to defects one-to-one using a maximum-cardinality
   bipartite matching (augmenting paths). Findings are processed in input
   order; each tries its compatible defects in order of highest overlap
   ratio, then earliest ``line_start``, then ``defect_id``. The number of
   true positives is the maximum matching size, so it does not depend on
   the order of findings.
4. The finding ``description`` is recorded but never affects matching.

Counting
--------
TP = matched pairs. FP = findings not matched, including duplicates (a
compatible finding whose defects were all taken by other findings) and
invalid findings. FN = defects not matched.

Zero-denominator convention: precision is 1.0 when there are no findings
and no missed defects, else 0.0; recall is 1.0 when there are no defects
and no false positives, else 0.0. F1 = 2TP / (2TP + FP + FN), and 1.0 when
all three counts are zero. A clean case with a silent reviewer therefore
scores 1.0 on all three; silence on a defective case scores 0.0.

Reward quality
--------------
``CaseResult.quality`` (the per-case F1) is the value in [0, 1] that
``evolution.adaptive.core.Reward.quality`` expects. The caller builds the
``Reward``; this module does not import it.

Aggregates and confidence intervals
-----------------------------------
Micro scores apply the formulas above to TP/FP/FN summed over cases.
Macro scores are the unweighted mean of the per-case values. Confidence
intervals use a percentile bootstrap over cases: cases are sorted by
``case_id``, resampled with replacement ``n_bootstrap`` times using
``random.Random(seed)``, and the interval bounds are linearly interpolated
quantiles. Same inputs and seed give identical output.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Iterable, Mapping, Sequence

VERIFIER_VERSION = "kafo-verifier/1"
MATCH_RULE = "category-equal+line-overlap/1"
DEFAULT_SEED = 0
DEFAULT_BOOTSTRAP = 1000
DEFAULT_CONFIDENCE = 0.95
# Must equal evolution.adaptive.benchmark.SEALED_SPLITS (checked by tests).
SEALED_SPLITS = ("test",)


# ── Data types ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Defect:
    defect_id: str
    category: str
    line_start: int
    line_end: int
    severity: str
    description: str


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    split: str
    language: str
    code: str
    difficulty: str
    domain: str
    defects: tuple[Defect, ...]


@dataclass(frozen=True)
class Finding:
    line_start: int
    line_end: int
    category: str
    description: str | None = None


@dataclass(frozen=True)
class MatchConfig:
    """Fixed matching parameters. Recorded in every report."""

    line_tolerance: int = 1
    min_overlap_ratio: float = 0.5
    category_aliases: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not _is_int(self.line_tolerance) or self.line_tolerance < 0:
            raise ValueError("line_tolerance must be a nonnegative int")
        if not _is_real(self.min_overlap_ratio) \
                or not 0.0 <= self.min_overlap_ratio <= 1.0:
            raise ValueError("min_overlap_ratio must be in [0, 1]")
        if not isinstance(self.category_aliases, tuple):
            raise ValueError("category_aliases must be a tuple of pairs")
        for pair in self.category_aliases:
            if (not isinstance(pair, tuple) or len(pair) != 2
                    or not all(_is_nonempty_str(p) for p in pair)):
                raise ValueError("category_aliases must be (alias, canonical) "
                                 "pairs of nonempty strings")

    def canonical(self, category: str) -> str:
        norm = _normalise(category)
        for alias, target in self.category_aliases:
            if _normalise(alias) == norm:
                return _normalise(target)
        return norm

    def to_dict(self) -> dict:
        return {"line_tolerance": self.line_tolerance,
                "min_overlap_ratio": self.min_overlap_ratio,
                "category_aliases": [list(p) for p in self.category_aliases]}


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    split: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    matches: tuple[tuple[int, str], ...]  # (finding index, defect_id)
    missed_defects: tuple[str, ...]
    false_positive_findings: tuple[int, ...]
    duplicate_findings: int
    invalid_findings: int

    @property
    def quality(self) -> float:
        """Verified quality in [0, 1] for ``Reward.quality`` (per-case F1)."""
        return self.f1

    def to_dict(self) -> dict:
        return {"case_id": self.case_id, "split": self.split,
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
                "precision": self.precision, "recall": self.recall,
                "f1": self.f1, "quality": self.quality,
                "matches": [list(m) for m in self.matches],
                "missed_defects": list(self.missed_defects),
                "false_positive_findings": list(self.false_positive_findings),
                "duplicate_findings": self.duplicate_findings,
                "invalid_findings": self.invalid_findings}


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def to_dict(self) -> dict:
        return {"point": self.point, "low": self.low, "high": self.high}


@dataclass(frozen=True)
class Scores:
    precision: Interval
    recall: Interval
    f1: Interval

    def to_dict(self) -> dict:
        return {"precision": self.precision.to_dict(),
                "recall": self.recall.to_dict(), "f1": self.f1.to_dict()}


@dataclass(frozen=True)
class EvaluationReport:
    cases: tuple[CaseResult, ...]
    micro: Scores
    macro: Scores
    seed: int
    n_bootstrap: int
    confidence: float
    config: MatchConfig
    missing_reviews: tuple[str, ...] = ()
    verifier_version: str = VERIFIER_VERSION
    match_rule: str = MATCH_RULE
    verifier_sha256: str = field(default_factory=lambda: verifier_fingerprint())

    def to_dict(self) -> dict:
        return {"verifier_version": self.verifier_version,
                "match_rule": self.match_rule,
                "verifier_sha256": self.verifier_sha256,
                "config": self.config.to_dict(),
                "seed": self.seed, "n_bootstrap": self.n_bootstrap,
                "confidence": self.confidence, "n_cases": len(self.cases),
                "missing_reviews": list(self.missing_reviews),
                "micro": self.micro.to_dict(), "macro": self.macro.to_dict(),
                "cases": [c.to_dict() for c in self.cases]}


# ── Validation helpers ─────────────────────────────────────────────────────

def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _is_nonempty_str(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _line_count(code: str) -> int:
    """Lines in code, counted on '\\n' (a final newline adds none)."""
    if not code:
        return 0
    return code.count("\n") + (0 if code.endswith("\n") else 1)


def _normalise(category: str) -> str:
    return "_".join(category.strip().lower().replace("-", " ").split())


def _require(cond: bool, where: str, message: str) -> None:
    if not cond:
        raise ValueError(f"{where}: {message}")


# ── Parsing ────────────────────────────────────────────────────────────────

def parse_case(raw: Mapping) -> BenchmarkCase:
    """Validate one benchmark case strictly; raise ValueError if malformed."""
    _require(isinstance(raw, Mapping), "case", "must be a JSON object")
    case_id = raw.get("case_id")
    _require(_is_nonempty_str(case_id), "case", "case_id must be a nonempty string")
    where = f"case {case_id!r}"
    for key in ("split", "language"):
        _require(_is_nonempty_str(raw.get(key)), where, f"{key} must be a nonempty string")
    code = raw.get("code")
    _require(isinstance(code, str), where, "code must be a string")
    context = raw.get("context")
    _require(isinstance(context, Mapping), where, "context must be an object")
    for key in ("difficulty", "domain"):
        _require(_is_nonempty_str(context.get(key)), where,
                 f"context.{key} must be a nonempty string")
    raw_defects = raw.get("defects")
    _require(isinstance(raw_defects, list), where, "defects must be a list")
    n_lines = _line_count(code)
    defects, seen = [], set()
    for i, d in enumerate(raw_defects):
        dwhere = f"{where} defect[{i}]"
        _require(isinstance(d, Mapping), dwhere, "must be an object")
        for key in ("defect_id", "category", "severity"):
            _require(_is_nonempty_str(d.get(key)), dwhere, f"{key} must be a nonempty string")
        _require(isinstance(d.get("description"), str), dwhere, "description must be a string")
        start, end = d.get("line_start"), d.get("line_end")
        _require(_is_int(start) and _is_int(end), dwhere, "line_start/line_end must be ints")
        _require(1 <= start <= end <= n_lines, dwhere,
                 f"need 1 <= line_start <= line_end <= {n_lines} (lines in code)")
        _require(d["defect_id"] not in seen, dwhere, "duplicate defect_id")
        seen.add(d["defect_id"])
        defects.append(Defect(d["defect_id"], d["category"], start, end,
                              d["severity"], d["description"]))
    return BenchmarkCase(case_id, raw["split"], raw["language"], code,
                         context["difficulty"], context["domain"], tuple(defects))


def parse_finding(raw) -> Finding | None:
    """Return a Finding, or None if the model emitted a malformed one."""
    if not isinstance(raw, Mapping):
        return None
    start = raw.get("line_start")
    end = raw.get("line_end", start)
    category = raw.get("category")
    description = raw.get("description")
    if not (_is_int(start) and _is_int(end) and 1 <= start <= end):
        return None
    if not _is_nonempty_str(category):
        return None
    if description is not None and not isinstance(description, str):
        return None
    return Finding(start, end, category, description)


def load_cases(path: str | Path) -> list[BenchmarkCase]:
    """Read open-split benchmark cases from JSONL (read-only).

    case_ids must be unique; sealed-split cases raise PermissionError.
    """
    cases = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                case = parse_case(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            if case.split in SEALED_SPLITS:
                raise PermissionError(
                    f"{path}:{lineno}: case {case.case_id!r} is in the sealed "
                    f"{case.split!r} split; sealed cases are never read from files "
                    "(use evolution.adaptive.evaluation.evaluate_sealed_test)")
            cases.append(case)
    _check_unique([c.case_id for c in cases], "case_id")
    return cases


def load_reviews(path: str | Path) -> dict[str, list]:
    """Read reviews from JSONL ``{case_id, findings}`` lines (read-only)."""
    reviews: dict[str, list] = {}
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            if not isinstance(obj, dict) or not _is_nonempty_str(obj.get("case_id")):
                raise ValueError(f"{path}:{lineno}: review needs a case_id")
            if not isinstance(obj.get("findings"), list):
                raise ValueError(f"{path}:{lineno}: findings must be a list")
            if obj["case_id"] in reviews:
                raise ValueError(f"{path}:{lineno}: duplicate review for "
                                 f"{obj['case_id']!r}")
            reviews[obj["case_id"]] = obj["findings"]
    return reviews


def _check_unique(ids: Sequence[str], name: str) -> None:
    seen = set()
    for i in ids:
        if i in seen:
            raise ValueError(f"duplicate {name}: {i!r}")
        seen.add(i)


# ── Matching and per-case scoring ──────────────────────────────────────────

def _overlap_ratio(finding: Finding, defect: Defect, tolerance: int) -> float:
    lo = max(finding.line_start, defect.line_start - tolerance)
    hi = min(finding.line_end, defect.line_end + tolerance)
    overlap = hi - lo + 1
    if overlap <= 0:
        return 0.0
    return overlap / (finding.line_end - finding.line_start + 1)


def _ratio(num: int, den: int, empty_ok: bool) -> float:
    if den == 0:
        return 1.0 if empty_ok else 0.0
    return num / den


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = _ratio(tp, tp + fp, empty_ok=fn == 0)
    recall = _ratio(tp, tp + fn, empty_ok=fp == 0)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn, empty_ok=True)
    return precision, recall, f1


def score_case(case: BenchmarkCase, findings: Sequence,
               config: MatchConfig = MatchConfig()) -> CaseResult:
    """Match raw model findings to the case's defects and score them."""
    if not isinstance(case, BenchmarkCase):
        raise TypeError("case must be a BenchmarkCase (use parse_case)")
    if isinstance(findings, (str, bytes, Mapping)) or not isinstance(findings, Sequence):
        raise TypeError("findings must be a list of finding objects")
    parsed = [parse_finding(f) for f in findings]
    defects = case.defects
    defect_cats = [config.canonical(d.category) for d in defects]

    # Compatible defects per finding, in preference order.
    adjacency: list[list[int]] = []
    for f in parsed:
        options = []
        if f is not None:
            cat = config.canonical(f.category)
            for j, d in enumerate(defects):
                if defect_cats[j] != cat:
                    continue
                ratio = _overlap_ratio(f, d, config.line_tolerance)
                if ratio > 0 and ratio >= config.min_overlap_ratio:
                    options.append((-ratio, d.line_start, d.defect_id, j))
        adjacency.append([j for *_, j in sorted(options)])

    # Maximum-cardinality bipartite matching (Kuhn's augmenting paths).
    owner: dict[int, int] = {}  # defect index -> finding index

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

    matched_findings = {i: j for j, i in owner.items()}
    matches = tuple(sorted((i, defects[j].defect_id)
                           for i, j in matched_findings.items()))
    fp_indices = tuple(i for i in range(len(parsed)) if i not in matched_findings)
    missed = tuple(d.defect_id for j, d in enumerate(defects) if j not in owner)
    tp, fp, fn = len(matches), len(fp_indices), len(missed)
    precision, recall, f1 = _prf(tp, fp, fn)
    return CaseResult(
        case_id=case.case_id, split=case.split,
        true_positives=tp, false_positives=fp, false_negatives=fn,
        precision=precision, recall=recall, f1=f1,
        matches=matches, missed_defects=missed,
        false_positive_findings=fp_indices,
        duplicate_findings=sum(1 for i in fp_indices if adjacency[i]),
        invalid_findings=sum(1 for f in parsed if f is None))


def quality(case: BenchmarkCase, findings: Sequence,
            config: MatchConfig = MatchConfig()) -> float:
    """Verified quality in [0, 1] for one case (its F1)."""
    return score_case(case, findings, config).quality


# ── Aggregation and bootstrap ──────────────────────────────────────────────

def _micro(results: Sequence[CaseResult]) -> tuple[float, float, float]:
    return _prf(sum(r.true_positives for r in results),
                sum(r.false_positives for r in results),
                sum(r.false_negatives for r in results))


def _macro(results: Sequence[CaseResult]) -> tuple[float, float, float]:
    n = len(results)
    return (math.fsum(r.precision for r in results) / n,
            math.fsum(r.recall for r in results) / n,
            math.fsum(r.f1 for r in results) / n)


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def aggregate(results: Iterable[CaseResult], *, seed: int = DEFAULT_SEED,
              n_bootstrap: int = DEFAULT_BOOTSTRAP,
              confidence: float = DEFAULT_CONFIDENCE) -> tuple[Scores, Scores]:
    """Micro and macro scores with percentile bootstrap intervals over cases."""
    results = sorted(results, key=lambda r: r.case_id)
    if not results:
        raise ValueError("at least one case result is required")
    _check_unique([r.case_id for r in results], "case_id")
    if not _is_int(seed):
        raise ValueError("seed must be an int")
    if not _is_int(n_bootstrap) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive int")
    if not _is_real(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    rng = random.Random(seed)
    n = len(results)
    samples: list[list[float]] = [[] for _ in range(6)]
    for _ in range(n_bootstrap):
        # int(random() * n) is stable across Python versions for a given seed.
        draw = [results[int(rng.random() * n)] for _ in range(n)]
        for k, value in enumerate(_micro(draw) + _macro(draw)):
            samples[k].append(value)
    alpha = 1.0 - confidence
    points = _micro(results) + _macro(results)
    intervals = []
    for point, values in zip(points, samples):
        values.sort()
        intervals.append(Interval(point, _quantile(values, alpha / 2),
                                  _quantile(values, 1 - alpha / 2)))
    return Scores(*intervals[:3]), Scores(*intervals[3:])


def evaluate(cases: Iterable[BenchmarkCase], reviews: Mapping[str, Sequence], *,
             config: MatchConfig = MatchConfig(), seed: int = DEFAULT_SEED,
             n_bootstrap: int = DEFAULT_BOOTSTRAP,
             confidence: float = DEFAULT_CONFIDENCE) -> EvaluationReport:
    """Score every case. A case without a review is scored as an empty review
    and listed in ``missing_reviews``; a review for an unknown case raises."""
    cases = sorted(cases, key=lambda c: c.case_id)
    _check_unique([c.case_id for c in cases], "case_id")
    known = {c.case_id for c in cases}
    unknown = sorted(set(reviews) - known)
    if unknown:
        raise ValueError(f"reviews for unknown case_ids: {unknown}")
    results = tuple(score_case(c, reviews.get(c.case_id, []), config) for c in cases)
    micro, macro = aggregate(results, seed=seed, n_bootstrap=n_bootstrap,
                             confidence=confidence)
    return EvaluationReport(
        cases=results, micro=micro, macro=macro, seed=seed,
        n_bootstrap=n_bootstrap, confidence=confidence, config=config,
        missing_reviews=tuple(c.case_id for c in cases if c.case_id not in reviews))


def verifier_fingerprint() -> str:
    """SHA-256 of this module's source, for auditing which rules scored a run."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# ── CLI ────────────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score review findings against benchmark defects (offline, read-only).")
    parser.add_argument("--cases", required=True, help="benchmark cases JSONL")
    parser.add_argument("--reviews", required=True, help="reviews JSONL {case_id, findings}")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--split", action="append",
                        help="only score cases from this split (repeatable)")
    args = parser.parse_args(argv)
    cases = load_cases(args.cases)
    reviews = load_reviews(args.reviews)
    unknown = sorted(set(reviews) - {c.case_id for c in cases})
    if unknown:
        parser.error(f"reviews for unknown case_ids: {unknown}")
    if args.split:
        cases = [c for c in cases if c.split in set(args.split)]
        ids = {c.case_id for c in cases}
        reviews = {k: v for k, v in reviews.items() if k in ids}
    report = evaluate(cases, reviews, seed=args.seed, n_bootstrap=args.bootstrap,
                      confidence=args.confidence)
    json.dump(report.to_dict(), sys.stdout, indent=2, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
