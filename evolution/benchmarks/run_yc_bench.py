"""Run the Phase 3 YC-Bench benchmark adapter."""

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
    benchmark_root: str | Path | None = None,
) -> dict[str, object]:
    """Run the YC-Bench adapter and write a JSON report."""

    if preset not in SUPPORTED_PRESETS:
        raise ValueError(f"Unsupported YC-Bench preset: {preset}")
    evidence = None
    if not dry_run:
        if benchmark_root is None:
            raise ValueError("real YC-Bench mode requires --benchmark-root")
        evidence = collect_yc_bench_real_benchmark_evidence(benchmark_root, preset=preset)
    return run_fixture_benchmark(
        benchmark="YC-Bench",
        pass_condition=PASS_CONDITION,
        baseline_prompt=baseline_prompt,
        candidate_prompt=candidate_prompt,
        fixtures_jsonl=fixtures_jsonl,
        output_json=output_json,
        dry_run=dry_run,
        preset=preset,
        real_benchmark_evidence=evidence,
    )


def collect_yc_bench_real_benchmark_evidence(benchmark_root: str | Path, *, preset: str) -> dict[str, object]:
    """Validate local YC-Bench package assets and return bounded evidence."""

    root = Path(benchmark_root)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"YC-Bench benchmark root must be an existing directory: {root}")

    pyproject = root / "pyproject.toml"
    package_root = root / "src" / "yc_bench"
    cli_entry = package_root / "__main__.py"
    preset_dir = package_root / "config" / "presets"
    missing = [
        str(path.relative_to(root))
        for path in (pyproject, package_root, preset_dir)
        if not path.exists()
    ]
    if missing:
        raise ValueError(f"YC-Bench package layout missing required paths: {missing}")

    available_presets = sorted(path.stem for path in preset_dir.glob("*.toml"))
    if not available_presets:
        raise ValueError(f"YC-Bench preset directory contains no .toml files: {preset_dir}")
    preset_source = preset if preset in available_presets else "default"
    if preset_source not in available_presets:
        raise ValueError(f"YC-Bench preset {preset!r} unavailable and no default preset exists")

    return {
        "benchmark_root": str(root),
        "package_layout_valid": True,
        "requested_preset": preset,
        "preset_source": preset_source,
        "available_presets": available_presets,
        "pyproject": str(pyproject),
        "cli_entry_present": cli_entry.exists(),
        "execution_scope": "local_pinned_package_smoke",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 3 YC-Bench benchmark adapter")
    parser.add_argument("--baseline-prompt", required=True, type=Path)
    parser.add_argument("--candidate-prompt", required=True, type=Path)
    parser.add_argument("--fixtures-jsonl", required=True, type=Path)
    parser.add_argument("--preset", choices=SUPPORTED_PRESETS, default="fast_test")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Run deterministic read-only fixture mode")
    parser.add_argument("--benchmark-root", type=Path, help="Pinned YC-Bench package root for real smoke mode")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_yc_bench(
            baseline_prompt=args.baseline_prompt,
            candidate_prompt=args.candidate_prompt,
            fixtures_jsonl=args.fixtures_jsonl,
            output_json=args.output_json,
            dry_run=args.dry_run,
            preset=args.preset,
            benchmark_root=args.benchmark_root,
        )
    except ValueError as exc:
        parser.error(str(exc))
    mode_label = "dry-run fixture benchmark" if args.dry_run else "real benchmark smoke"
    if report["passed"]:
        print(f"YC-Bench {mode_label} passed: {args.output_json}")
        return 0
    print(f"YC-Bench {mode_label} failed: {report['failed_checks']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
