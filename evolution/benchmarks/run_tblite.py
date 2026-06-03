"""Run the Phase 3 TBLite benchmark adapter."""

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
    benchmark_root: str | Path | None = None,
    task_limit: int = 3,
) -> dict[str, object]:
    """Run the TBLite adapter and write a JSON report.

    Without ``--dry-run`` this records bounded evidence from a pinned local
    terminal-bench-lite task corpus. It does not mutate active Hermes prompt or
    runtime state.
    """

    evidence = None
    if not dry_run:
        if benchmark_root is None:
            raise ValueError("real TBLite mode requires --benchmark-root")
        evidence = collect_tblite_real_benchmark_evidence(benchmark_root, task_limit=task_limit)

    return run_fixture_benchmark(
        benchmark="TBLite",
        pass_condition=PASS_CONDITION,
        baseline_prompt=baseline_prompt,
        candidate_prompt=candidate_prompt,
        fixtures_jsonl=fixtures_jsonl,
        output_json=output_json,
        dry_run=dry_run,
        real_benchmark_evidence=evidence,
    )


def collect_tblite_real_benchmark_evidence(benchmark_root: str | Path, *, task_limit: int = 3) -> dict[str, object]:
    """Validate local terminal-bench-lite assets and return bounded evidence."""

    root = Path(benchmark_root)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"TBLite benchmark root must be an existing directory: {root}")
    if task_limit <= 0:
        raise ValueError("task-limit must be positive")

    task_files = sorted(root.glob("*/task.toml"))
    if not task_files:
        raise ValueError(f"TBLite benchmark root contains no task.toml files: {root}")

    sample_tasks: list[str] = []
    missing_required_files: list[str] = []
    for task_file in task_files[:task_limit]:
        task_dir = task_file.parent
        task_name = task_dir.name
        sample_tasks.append(task_name)
        for required in ("task.toml", "tests/test.sh", "solution/solve.sh"):
            required_path = task_dir / required
            if not required_path.exists():
                missing_required_files.append(f"{task_name}/{required}")

    if missing_required_files:
        raise ValueError(f"TBLite task sample missing required files: {missing_required_files}")

    return {
        "benchmark_root": str(root),
        "task_count": len(task_files),
        "validated_task_count": min(task_limit, len(task_files)),
        "sample_tasks": sample_tasks,
        "required_files_checked": ["task.toml", "tests/test.sh", "solution/solve.sh"],
        "execution_scope": "local_pinned_task_corpus_smoke",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 3 TBLite benchmark adapter")
    parser.add_argument("--baseline-prompt", required=True, type=Path)
    parser.add_argument("--candidate-prompt", required=True, type=Path)
    parser.add_argument("--fixtures-jsonl", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Run deterministic read-only fixture mode")
    parser.add_argument("--benchmark-root", type=Path, help="Pinned terminal-bench-lite corpus root for real smoke mode")
    parser.add_argument("--task-limit", type=int, default=3, help="Number of TBLite tasks to validate in real smoke mode")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_tblite_benchmark(
            baseline_prompt=args.baseline_prompt,
            candidate_prompt=args.candidate_prompt,
            fixtures_jsonl=args.fixtures_jsonl,
            output_json=args.output_json,
            dry_run=args.dry_run,
            benchmark_root=args.benchmark_root,
            task_limit=args.task_limit,
        )
    except ValueError as exc:
        parser.error(str(exc))
    mode_label = "dry-run fixture benchmark" if args.dry_run else "real benchmark smoke"
    if report["passed"]:
        print(f"TBLite {mode_label} passed: {args.output_json}")
        return 0
    print(f"TBLite {mode_label} failed: {report['failed_checks']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
