"""Score code reviews on the gate 2 benchmark with the gate 3 verifier.

Integration glue only: it changes neither the benchmark data nor the scoring
rules, and it never writes to disk.

Data access rule (KAFO-EVO-001, "Preserve train / validation / sealed test
separation"):

* Benchmark cases reach the adaptive loop, the verifier and the policy
  comparison harness only through ``evolution.adaptive.benchmark``
  (``load_benchmark`` / ``load_split``), which checks the pinned SHA-256
  manifest. No other module builds paths into ``datasets/python_review``.
* Optimisation and selection use the open splits only (:func:`open_cases`,
  :func:`evaluate_open`).
* :func:`sealed_test_cases` is the single place that unseals the test
  split; :func:`evaluate_sealed_test` scores through it. Both need
  ``final_evaluation=True`` and refuse inside
  ``benchmark.optimisation_phase()``.

The verifier keeps its own case type and imports only the standard library,
so cases cross over as plain dicts (``BenchmarkCase.to_dict``) and are
re-validated by ``verifier.parse_case``.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from . import benchmark, verifier


def to_verifier_case(case: benchmark.BenchmarkCase) -> verifier.BenchmarkCase:
    """Convert a loaded benchmark case to the verifier's (re-validated) type."""
    if not isinstance(case, benchmark.BenchmarkCase):
        raise TypeError("expected a case returned by evolution.adaptive.benchmark")
    return verifier.parse_case(case.to_dict())


def ground_truth_findings(case) -> list[dict]:
    """A review that reports exactly the labelled defects (reference ceiling)."""
    return [{"line_start": d.line_start, "line_end": d.line_end,
             "category": d.category, "description": d.description}
            for d in case.defects]


def open_cases(splits: Sequence[str] = benchmark.OPEN_SPLITS, *, version: str = benchmark.DEFAULT_VERSION,
               ) -> dict[str, tuple[verifier.BenchmarkCase, ...]]:
    """Open-split cases in verifier form. Sealed splits are refused."""
    loaded = benchmark.load_benchmark(tuple(splits), version=version)
    return {split: tuple(to_verifier_case(c) for c in cases)
            for split, cases in loaded.items()}


def _evaluate(cases: Sequence[verifier.BenchmarkCase], reviews: Mapping[str, Sequence],
              **options) -> verifier.EvaluationReport:
    ids = {c.case_id for c in cases}
    return verifier.evaluate(cases, {k: v for k, v in reviews.items() if k in ids},
                             **options)


def evaluate_open(reviews: Mapping[str, Sequence], *,
                  splits: Sequence[str] = benchmark.OPEN_SPLITS,
                  version: str = benchmark.DEFAULT_VERSION,
                  **options) -> dict[str, verifier.EvaluationReport]:
    """Verifier report per open split. ``reviews`` maps case_id -> findings.

    A review for a case outside the requested splits (for example a sealed
    test case) raises ``ValueError``. Cases without a review are scored as
    empty reviews and listed in ``missing_reviews``. ``options`` are passed to
    ``verifier.evaluate`` (config, seed, n_bootstrap, confidence).
    """
    by_split = open_cases(splits, version=version)
    known = {c.case_id for cases in by_split.values() for c in cases}
    unknown = sorted(set(reviews) - known)
    if unknown:
        raise ValueError(f"reviews for case_ids outside {tuple(splits)}: {unknown}")
    return {split: _evaluate(cases, reviews, **options)
            for split, cases in by_split.items()}


def review_input(case) -> Mapping:
    """What a reviewer may see of a case: no defects, labels or split."""
    return MappingProxyType({
        "case_id": case.case_id, "language": case.language, "code": case.code,
        "context": MappingProxyType({"difficulty": case.difficulty, "domain": case.domain})})


def sealed_test_cases(*, final_evaluation: bool = False,
                      version: str = benchmark.DEFAULT_VERSION
                      ) -> tuple[verifier.BenchmarkCase, ...]:
    """The sealed test cases, labels included, sorted by ``case_id``.

    The single place that unseals the test split. Same rules as
    :func:`evaluate_sealed_test` (which uses it): ``final_evaluation=True``,
    refused inside an optimisation phase. Only the recorded, run-once steps
    in ``evolution.adaptive.final_evaluation`` may call it (guard scan in
    ``tests/test_benchmark_integration.py``). Reviewers get
    :func:`review_input` of these cases, never the cases themselves.
    """
    if final_evaluation is not True:
        raise benchmark.SealedSplitError(
            "sealed_test_cases needs final_evaluation=True; use open_cases "
            "during optimisation and selection")
    cases = tuple(to_verifier_case(c) for c in benchmark.load_split(
        "test", version=version, allow_sealed_test=True))
    return tuple(sorted(cases, key=lambda c: c.case_id))


def evaluate_sealed_test(reviews: Mapping[str, Sequence] | None = None, *,
                         reviewer: Callable[[Mapping], Sequence] | None = None,
                         final_evaluation: bool = False,
                         version: str = benchmark.DEFAULT_VERSION,
                         **options) -> verifier.EvaluationReport:
    """Final, one-off evaluation on the sealed test split.

    Requires ``final_evaluation=True`` and refuses inside an optimisation
    phase. Its result must not feed back into optimisation or selection.
    Pass exactly one of ``reviews`` (case_id -> findings) or ``reviewer``, a
    callable that receives :func:`review_input` for each test case (code and
    context only, never the labelled defects) and returns its findings. Use
    ``evolution.adaptive.final_evaluation.run_final_evaluation`` for the
    recorded, run-once final evaluation.
    """
    if final_evaluation is not True:
        raise benchmark.SealedSplitError(
            "evaluate_sealed_test needs final_evaluation=True; use evaluate_open "
            "during optimisation and selection")
    if (reviews is None) == (reviewer is None):
        raise ValueError("pass exactly one of reviews or reviewer")
    if reviewer is not None and not callable(reviewer):
        raise TypeError("reviewer must be callable")
    cases = sealed_test_cases(final_evaluation=True, version=version)
    if reviewer is not None:
        reviews = {c.case_id: reviewer(review_input(c)) for c in cases}
    unknown = sorted(set(reviews) - {c.case_id for c in cases})
    if unknown:
        raise ValueError(f"reviews for case_ids outside the test split: {unknown}")
    return _evaluate(cases, reviews, **options)
