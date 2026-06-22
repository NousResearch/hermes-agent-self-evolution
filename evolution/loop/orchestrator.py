"""Cross-phase orchestration — run multiple evolution phases in order."""

import time
from dataclasses import dataclass, field


@dataclass
class PhaseResult:
    name: str
    status: str
    improvement: float = 0.0
    reverted: list = field(default_factory=list)
    elapsed: int = 0
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name, "status": self.status,
            "improvement": self.improvement, "reverted": self.reverted,
            "elapsed": self.elapsed, "message": self.message,
        }


@dataclass
class OrchestrationResult:
    phases: dict
    status: str = "success"
    total_elapsed: int = 0

    def summary_metrics(self) -> dict:
        total_imp = sum(
            p.get("improvement", 0.0) for p in self.phases.values()
            if p.get("status") == "success"
        )
        return {
            "total_improvement": total_imp,
            "phases_run": sum(1 for p in self.phases.values() if p.get("status") in ("success", "failed")),
            "phases_skipped": sum(1 for p in self.phases.values() if p.get("status") == "skipped"),
            "phases_failed": sum(1 for p in self.phases.values() if p.get("status") == "failed"),
        }


def run_phases(
    phase_funcs: dict,
    phases: list,
    dry_run: bool = False,
) -> OrchestrationResult:
    """Run evolution phases in order, aggregating results."""
    results = {}
    overall_status = "success"
    start_total = time.time()

    for name in phases:
        if name not in phase_funcs:
            results[name] = PhaseResult(name=name, status="skipped", message="No function registered").to_dict()
            continue
        if dry_run:
            results[name] = PhaseResult(name=name, status="skipped", message="dry-run").to_dict()
            continue
        try:
            t0 = time.time()
            output = phase_funcs[name]()
            elapsed = int(time.time() - t0)
            improvement = 0.0
            reverted = []
            if output:
                improvement = output.get("evolved_score", 0) - output.get("baseline_score", 0)
                reverted = output.get("reverted_tools", output.get("reverted_blocks", output.get("reverted", [])))
            results[name] = PhaseResult(
                name=name, status="success",
                improvement=improvement, reverted=reverted,
                elapsed=elapsed,
            ).to_dict()
        except Exception as e:
            results[name] = PhaseResult(name=name, status="failed", message=str(e)).to_dict()
            overall_status = "partial"

    total_elapsed = int(time.time() - start_total)
    active = [r for r in results.values() if r.get("status") != "skipped"]
    if active and all(r.get("status") == "failed" for r in active):
        overall_status = "failed"

    return OrchestrationResult(phases=results, status=overall_status, total_elapsed=total_elapsed)
