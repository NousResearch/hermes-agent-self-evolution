"""Run the Phase 3 YC-Bench dry-run fixture benchmark adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from evolution.benchmarks.contract import run_fixture_benchmark

PASS_CONDITION = "coherence_score_holds_or_improves"
SUPPORTED_PRESETS = ("fast_test",)


def run_yc_bench(
    *,
    baseline_prompt: str | Path,
    candidate_prompt: str | Path,
    fixtures_jsonl: str | Path,
    output_json: str | Path,
    dry_run: bool,
    preset: str = "fast_test",
) -> dict[str, object]:
    """Run the read-only YC-Bench fixture adapter and write a JSON report."""

    if preset not in SUPPORTED_PRESETS:
        raise ValueError(f"Unsupported YC-Bench preset: {preset}")
    return run_fixture_benchmark(
        benchmark="YC-Bench",
        pass_condition=PASS_CONDITION,
        baseline_prompt=baseline_prompt,
        candidate_prompt=candidate_prompt,
        fixtures_jsonl=fixtures_jsonl,
        output_json=output_json,
        dry_run=dry_run,
        preset=preset,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 3 YC-Bench dry-run fixture benchmark adapter")
    parser.add_argument("--baseline-prompt", required=True, type=Path)
    parser.add_argument("--candidate-prompt", required=True, type=Path)
    parser.add_argument("--fixtures-jsonl", required=True, type=Path)
    parser.add_argument("--preset", choices=SUPPORTED_PRESETS, default="fast_test")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Required safety flag; real benchmark mode is not implemented")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("Phase 3 YC-Bench adapter currently requires --dry-run")
    try:
        report = run_yc_bench(
            baseline_prompt=args.baseline_prompt,
            candidate_prompt=args.candidate_prompt,
            fixtures_jsonl=args.fixtures_jsonl,
            output_json=args.output_json,
            dry_run=args.dry_run,
            preset=args.preset,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if report["passed"]:
        print(f"YC-Bench dry-run fixture benchmark passed: {args.output_json}")
        return 0
    print(f"YC-Bench dry-run fixture benchmark failed: {report['failed_checks']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
