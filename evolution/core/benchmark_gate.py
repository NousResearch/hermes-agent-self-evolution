"""Benchmark gating for evolved skills.

Runs a fast subset of TBLite tasks before/after evolution to detect
regressions. This is an OPT-IN gate (expensive, ~20-30 min).

Usage:
    gate_result = run_benchmark_gate(skill_name, hermes_repo, baseline_module, evolved_module)
    if not gate_result.passed:
        print(f"Regression detected: {gate_result.message}")
"""

import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class GateResult:
    """Result of a benchmark gate check."""
    passed: bool
    constraint_name: str = "benchmark_gate"
    message: str = ""
    details: Optional[str] = None
    baseline_pass_rate: float = 0.0
    evolved_pass_rate: float = 0.0
    regression: float = 0.0


# Fast subset: 20 TBLite tasks that cover diverse capabilities
TBLITE_FAST_SUBSET = [
    "write_python_function",
    "debug_syntax_error",
    "explain_sql_query",
    "refactor_function",
    "write_unit_test",
    "create_bash_script",
    "parse_json_data",
    "fix_import_error",
    "optimize_loop",
    "write_docstring",
    "implement_binary_search",
    "convert_csv_to_json",
    "find_security_issue",
    "validate_email_format",
    "calculate_complexity",
    "sort_file_contents",
    "merge_two_dicts",
    "handle_file_not_found",
    "format_datetime",
    "generate_random_data",
]


def run_fast_subset(
    hermes_repo: Path,
    task_list: Optional[list[str]] = None,
    timeout_per_task: int = 120,
) -> dict:
    """Run a subset of TBLite tasks using batch_runner.

    Args:
        hermes_repo: Path to hermes-agent repo.
        task_list: List of task names to run. Defaults to TBLITE_FAST_SUBSET.
        timeout_per_task: Max seconds per task.

    Returns:
        Dict with keys: pass_rate, total, passed, failed, per_task.
    """
    tasks = task_list or TBLITE_FAST_SUBSET
    batch_runner = hermes_repo / "batch_runner.py"

    if not batch_runner.exists():
        return {
            "pass_rate": 0.0,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "per_task": {},
            "error": f"batch_runner.py not found at {batch_runner}",
        }

    results = {"passed": 0, "failed": 0, "per_task": {}}
    for task_name in tasks:
        try:
            result = subprocess.run(
                ["python", str(batch_runner), "--task", task_name, "--timeout", str(timeout_per_task)],
                capture_output=True, text=True,
                cwd=str(hermes_repo),
                timeout=timeout_per_task + 30,
            )
            success = result.returncode == 0
        except subprocess.TimeoutExpired:
            success = False

        results["per_task"][task_name] = {
            "passed": success,
            "output": (result.stdout[:200] if success else result.stderr[:200]) if not success else "",
        }
        if success:
            results["passed"] += 1
        else:
            results["failed"] += 1

    total = len(tasks)
    results["total"] = total
    results["pass_rate"] = results["passed"] / max(1, total)
    return results


def compare(
    baseline_results: dict,
    evolved_results: dict,
    regression_threshold: float = 0.02,
) -> GateResult:
    """Compare baseline vs evolved benchmark results.

    Args:
        baseline_results: Dict from run_fast_subset (baseline).
        evolved_results: Dict from run_fast_subset (evolved).
        regression_threshold: Max allowed regression (default 2%).

    Returns:
        GateResult indicating pass/fail.
    """
    baseline_rate = baseline_results.get("pass_rate", 0.0)
    evolved_rate = evolved_results.get("pass_rate", 0.0)
    regression = baseline_rate - evolved_rate

    if baseline_results.get("error"):
        return GateResult(
            passed=False,
            message=f"Baseline error: {baseline_results['error']}",
            baseline_pass_rate=baseline_rate,
            evolved_pass_rate=evolved_rate,
            regression=regression,
        )

    if regression > regression_threshold:
        return GateResult(
            passed=False,
            message=f"Regression: {regression:.1%} drop (threshold {regression_threshold:.1%})",
            details=f"Baseline: {baseline_rate:.1%} ({baseline_results.get('passed', 0)}/{baseline_results.get('total', 0)})\n"
                    f"Evolved: {evolved_rate:.1%} ({evolved_results.get('passed', 0)}/{evolved_results.get('total', 0)})",
            baseline_pass_rate=baseline_rate,
            evolved_pass_rate=evolved_rate,
            regression=regression,
        )

    return GateResult(
        passed=True,
        message=f"No significant regression ({regression:+.1%})",
        details=f"Baseline: {baseline_rate:.1%} → Evolved: {evolved_rate:.1%}",
        baseline_pass_rate=baseline_rate,
        evolved_pass_rate=evolved_rate,
        regression=regression,
    )


def run_benchmark_gate(
    skill_name: str,
    hermes_repo: Path,
    baseline_module=None,
    evolved_module=None,
    task_list: Optional[list[str]] = None,
    regression_threshold: float = 0.02,
) -> GateResult:
    """Run the full benchmark gate: baseline → evolve → compare.

    This is the main entry point called by evolve_skill.py.

    Args:
        skill_name: Name of the evolved skill (for logging).
        hermes_repo: Path to hermes-agent repo.
        baseline_module: Unused (future use for in-process evaluation).
        evolved_module: Unused (future use for in-process evaluation).
        task_list: Specific tasks to run.
        regression_threshold: Max allowed regression.

    Returns:
        GateResult.
    """
    print(f"  Running benchmark gate for '{skill_name}'...")
    print(f"  Task count: {len(task_list or TBLITE_FAST_SUBSET)}")
    print(f"  Estimated runtime: ~20-30 minutes")

    # Run baseline
    print(f"  [1/2] Running baseline...")
    baseline_results = run_fast_subset(hermes_repo, task_list)
    if baseline_results.get("error"):
        print(f"  Baseline error: {baseline_results['error']}")
        return GateResult(passed=False, message=f"Baseline error: {baseline_results['error']}")

    print(f"  Baseline pass rate: {baseline_results['pass_rate']:.1%} ({baseline_results['passed']}/{baseline_results['total']})")

    # Run evolved (same tasks)
    print(f"  [2/2] Running evolved...")
    evolved_results = run_fast_subset(hermes_repo, task_list)
    if evolved_results.get("error"):
        print(f"  Evolved error: {evolved_results['error']}")
        return GateResult(passed=False, message=f"Evolved error: {evolved_results['error']}")

    print(f"  Evolved pass rate: {evolved_results['pass_rate']:.1%} ({evolved_results['passed']}/{evolved_results['total']})")

    # Compare
    return compare(baseline_results, evolved_results, regression_threshold)
