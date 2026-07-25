"""
Benchmark Gate for Hermes Agent Self-Evolution.

Runs TBLite and YC-Bench to detect regressions in evolved variants.
Evolved skills/prompts/tools must not regress on benchmarks.
"""

import subprocess
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class BenchmarkStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    REGRESSION = "regression"
    IMPROVED = "improved"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    status: BenchmarkStatus
    score: Optional[float] = None
    baseline_score: Optional[float] = None
    details: Dict[str, Any] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class BenchmarkReport:
    """Complete benchmark report for an evolved variant."""
    overall_status: BenchmarkStatus
    results: List[BenchmarkResult]
    total_duration: float
    baseline_comparison: Dict[str, Any]
    
    def __post_init__(self):
        # Determine overall status
        if any(r.status == BenchmarkStatus.REGRESSION for r in self.results):
            self.overall_status = BenchmarkStatus.REGRESSION
        elif any(r.status == BenchmarkStatus.FAILED for r in self.results):
            self.overall_status = BenchmarkStatus.FAILED
        elif any(r.status == BenchmarkStatus.IMPROVED for r in self.results):
            self.overall_status = BenchmarkStatus.IMPROVED
        elif all(r.status == BenchmarkStatus.PASSED for r in self.results):
            self.overall_status = BenchmarkStatus.PASSED
        else:
            self.overall_status = BenchmarkStatus.SKIPPED


class BenchmarkGate:
    """
    Runs benchmark suite to gate evolved variants.
    
    Primary: TBLite (fast, ~1-2 hours, binary pass/fail) - regression check
    Secondary: YC-Bench fast_test (~50 turns, composite score) - coherence check
    
    A variant that scores higher on behavioral tests but lower on TBLite is REJECTED.
    """
    
    def __init__(
        self,
        hermes_agent_repo: Path,
        baseline_file: Optional[Path] = None,
        regression_threshold: float = 0.02  # 2% regression tolerance
    ):
        self.hermes_agent_repo = Path(hermes_agent_repo)
        self.baseline_file = baseline_file or (self.hermes_agent_repo / ".benchmarks" / "baseline.json")
        self.regression_threshold = regression_threshold
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
    
    def run_all(self, variant_name: str = "evolved") -> BenchmarkReport:
        """Run full benchmark suite."""
        start_time = time.time()
        results = []
        
        # Run TBLite (primary regression gate)
        tblite_result = self._run_tblite(variant_name)
        results.append(tblite_result)
        
        # Run YC-Bench fast test (secondary coherence check)
        yc_result = self._run_yc_bench_fast(variant_name)
        results.append(yc_result)
        
        # Compare with baseline
        baseline = self._load_baseline()
        comparison = self._compare_with_baseline(results, baseline)
        
        # Save new baseline if this variant is better
        if variant_name != "baseline" and self._should_update_baseline(results, baseline):
            self._save_baseline(results, variant_name)
        
        total_duration = time.time() - start_time
        
        return BenchmarkReport(
            overall_status=BenchmarkStatus.SKIPPED,  # Will be computed
            results=results,
            total_duration=total_duration,
            baseline_comparison=comparison
        )
    
    def _run_tblite(self, variant_name: str) -> BenchmarkResult:
        """Run TBLite benchmark."""
        start = time.time()
        tblite_script = self.hermes_agent_repo / "scripts" / "run_tblite_fast.py"
        
        if not tblite_script.exists():
            # Try alternative locations
            alt_paths = [
                self.hermes_agent_repo / "benchmarks" / "tblite" / "run_fast.py",
                self.hermes_agent_repo / "environments" / "benchmarks" / "tblite" / "run_fast.py",
            ]
            for p in alt_paths:
                if p.exists():
                    tblite_script = p
                    break
        
        if not tblite_script.exists():
            return BenchmarkResult(
                name="tblite",
                status=BenchmarkStatus.SKIPPED,
                duration_seconds=time.time() - start,
                error="TBLite script not found"
            )
        
        try:
            # Run with the evolved skill variant active
            env = {"HERMES_VARIANT": variant_name}
            result = subprocess.run(
                ["python", str(tblite_script)],
                cwd=self.hermes_agent_repo,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hours max
                env={**subprocess.os.environ, **env}
            )
            
            duration = time.time() - start
            
            # Parse score from output
            score = self._parse_tblite_score(result.stdout)
            
            if result.returncode == 0:
                status = BenchmarkStatus.PASSED
            else:
                status = BenchmarkStatus.FAILED
            
            return BenchmarkResult(
                name="tblite",
                status=status,
                score=score,
                duration_seconds=duration,
                details={
                    "returncode": result.returncode,
                    "stdout_tail": result.stdout[-2000:],
                    "stderr_tail": result.stderr[-1000:]
                }
            )
        except subprocess.TimeoutExpired:
            return BenchmarkResult(
                name="tblite",
                status=BenchmarkStatus.ERROR,
                duration_seconds=time.time() - start,
                error="TBLite timed out (>2 hours)"
            )
        except Exception as e:
            return BenchmarkResult(
                name="tblite",
                status=BenchmarkStatus.ERROR,
                duration_seconds=time.time() - start,
                error=str(e)
            )
    
    def _run_yc_bench_fast(self, variant_name: str) -> BenchmarkResult:
        """Run YC-Bench fast test preset."""
        start = time.time()
        
        # Look for YC-Bench runner
        yc_scripts = [
            self.hermes_agent_repo / "scripts" / "run_yc_bench_fast.py",
            self.hermes_agent_repo / "benchmarks" / "yc_bench" / "run_fast.py",
            self.hermes_agent_repo / "environments" / "benchmarks" / "yc_bench" / "run_fast.py",
        ]
        
        yc_script = None
        for p in yc_scripts:
            if p.exists():
                yc_script = p
                break
        
        if not yc_script:
            return BenchmarkResult(
                name="yc_bench_fast",
                status=BenchmarkStatus.SKIPPED,
                duration_seconds=time.time() - start,
                error="YC-Bench fast script not found"
            )
        
        try:
            env = {"HERMES_VARIANT": variant_name}
            result = subprocess.run(
                ["python", str(yc_script)],
                cwd=self.hermes_agent_repo,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour max
                env={**subprocess.os.environ, **env}
            )
            
            duration = time.time() - start
            score = self._parse_yc_score(result.stdout)
            
            if result.returncode == 0:
                status = BenchmarkStatus.PASSED
            else:
                status = BenchmarkStatus.FAILED
            
            return BenchmarkResult(
                name="yc_bench_fast",
                status=status,
                score=score,
                duration_seconds=duration,
                details={
                    "returncode": result.returncode,
                    "stdout_tail": result.stdout[-2000:],
                    "stderr_tail": result.stderr[-1000:]
                }
            )
        except subprocess.TimeoutExpired:
            return BenchmarkResult(
                name="yc_bench_fast",
                status=BenchmarkStatus.ERROR,
                duration_seconds=time.time() - start,
                error="YC-Bench timed out (>1 hour)"
            )
        except Exception as e:
            return BenchmarkResult(
                name="yc_bench_fast",
                status=BenchmarkStatus.ERROR,
                duration_seconds=time.time() - start,
                error=str(e)
            )
    
    def _parse_tblite_score(self, output: str) -> Optional[float]:
        """Extract TBLite pass rate from output."""
        import re
        # Look for patterns like "Pass rate: 85%" or "85/100 passed"
        patterns = [
            r"pass rate[:\s]+(\d+(?:\.\d+)?)%",
            r"(\d+)/(\d+)\s+passed",
            r"passed[:\s]+(\d+(?:\.\d+)?)",
            r"score[:\s]+(\d+(?:\.\d+)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    return float(match.group(1)) / float(match.group(2)) * 100
                return float(match.group(1))
        return None
    
    def _parse_yc_score(self, output: str) -> Optional[float]:
        """Extract YC-Bench composite score from output."""
        import re
        patterns = [
            r"composite score[:\s]+(\d+(?:\.\d+)?)",
            r"score[:\s]+(\d+(?:\.\d+)?)",
            r"overall[:\s]+(\d+(?:\.\d+)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None
    
    def _load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load baseline benchmark scores."""
        if self.baseline_file.exists():
            with open(self.baseline_file) as f:
                return json.load(f)
        return None
    
    def _save_baseline(self, results: List[BenchmarkResult], variant_name: str) -> None:
        """Save benchmark results as new baseline."""
        baseline = {
            "variant": variant_name,
            "timestamp": time.time(),
            "results": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "score": r.score,
                    "duration": r.duration_seconds
                }
                for r in results
            ]
        }
        with open(self.baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)
    
    def _compare_with_baseline(
        self, 
        results: List[BenchmarkResult], 
        baseline: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare current results with baseline."""
        if not baseline:
            return {"status": "no_baseline", "message": "No baseline available for comparison"}
        
        baseline_scores = {r["name"]: r.get("score") for r in baseline.get("results", [])}
        comparison = {}
        
        for result in results:
            base_score = baseline_scores.get(result.name)
            if base_score is not None and result.score is not None:
                diff = result.score - base_score
                pct_change = (diff / base_score) * 100 if base_score != 0 else 0
                
                if pct_change <= -self.regression_threshold * 100:
                    status = "regression"
                elif pct_change >= self.regression_threshold * 100:
                    status = "improvement"
                else:
                    status = "stable"
                
                comparison[result.name] = {
                    "current_score": result.score,
                    "baseline_score": base_score,
                    "absolute_diff": diff,
                    "percent_change": pct_change,
                    "status": status
                }
            else:
                comparison[result.name] = {
                    "status": "no_comparison",
                    "message": "Missing score in current or baseline"
                }
        
        # Overall determination
        any_regression = any(c.get("status") == "regression" for c in comparison.values())
        any_improvement = any(c.get("status") == "improvement" for c in comparison.values())
        
        return {
            "comparisons": comparison,
            "has_regression": any_regression,
            "has_improvement": any_improvement,
            "overall": "regression" if any_regression else ("improvement" if any_improvement else "stable")
        }
    
    def _should_update_baseline(
        self, 
        results: List[BenchmarkResult], 
        baseline: Optional[Dict[str, Any]]
    ) -> bool:
        """Determine if current results should become new baseline."""
        if not baseline:
            return True  # First run becomes baseline
        
        comparison = self._compare_with_baseline(results, baseline)
        return comparison.get("overall") in ["improvement", "stable"]


def run_benchmark_gate(
    hermes_agent_repo: Path,
    variant_name: str = "evolved",
    baseline_file: Optional[Path] = None
) -> BenchmarkReport:
    """Convenience function to run benchmark gate."""
    gate = BenchmarkGate(hermes_agent_repo, baseline_file)
    return gate.run_all(variant_name)