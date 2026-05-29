"""Run the Phase 3 TBLite dry-run fixture benchmark adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from evolution.benchmarks.contract import run_fixture_benchmark

PASS_CONDITION = "no_regression_against_baseline"


def run_tblite_benchmark(
    *,
    baseline_prompt: str | Path,
    candidate_prompt: str | Path,
    fixtures_jsonl: str | Path,
    output_json: str | Path,
    dry_run: bool,
) -> dict[str, object]:
    """Run the read-only TBLite fixture adapter and write a JSON report."""

    return run_fixture_benchmark(
        benchmark="TBLite",
        pass_condition=PASS_CONDITION,
        baseline_prompt=baseline_prompt,
        candidate_prompt=candidate_prompt,
        fixtures_jsonl=fixtures_jsonl,
        output_json=output_json,
        dry_run=dry_run,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 3 TBLite dry-run fixture benchmark adapter")
    parser.add_argument("--baseline-prompt", required=True, type=Path)
    parser.add_argument("--candidate-prompt", required=True, type=Path)
    parser.add_argument("--fixtures-jsonl", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Required safety flag; real benchmark mode is not implemented")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("Phase 3 TBLite adapter currently requires --dry-run")
    report = run_tblite_benchmark(
        baseline_prompt=args.baseline_prompt,
        candidate_prompt=args.candidate_prompt,
        fixtures_jsonl=args.fixtures_jsonl,
        output_json=args.output_json,
        dry_run=args.dry_run,
    )
    if report["passed"]:
        print(f"TBLite dry-run fixture benchmark passed: {args.output_json}")
        return 0
    print(f"TBLite dry-run fixture benchmark failed: {report['failed_checks']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
