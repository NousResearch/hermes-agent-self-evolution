"""Gate that decides whether an evolved variant may auto-merge."""
from dataclasses import dataclass

@dataclass
class GateDecision:
    auto_merge: bool
    reason: str
    improvement: float
    regression: bool = False

class AutoMergeGate:
    def __init__(self, min_improvement: float = 0.02, regression_tolerance: float = 0.01):
        self.min_improvement = min_improvement
        self.regression_tolerance = regression_tolerance

    def evaluate(self, baseline: float, evolved: float, constraints_passed: bool) -> GateDecision:
        delta = evolved - baseline
        if not constraints_passed:
            return GateDecision(False, "Constraint check failed", delta)
        if delta < -self.regression_tolerance:
            return GateDecision(False, f"Regression detected: {delta:+.3f}", delta, regression=True)
        if delta < self.min_improvement:
            return GateDecision(False, f"Improvement {delta:+.3f} below threshold +{self.min_improvement:.3f}", delta)
        return GateDecision(True, f"Improvement {delta:+.3f} meets threshold", delta)
