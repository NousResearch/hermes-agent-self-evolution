from evolution.core.regression_guard import AutoMergeGate

def test_gate_rejects_below_threshold():
    gate = AutoMergeGate(min_improvement=0.02, regression_tolerance=0.01)
    d = gate.evaluate(baseline=0.70, evolved=0.71, constraints_passed=True)
    assert d.auto_merge is False and "below threshold" in d.reason.lower()

def test_gate_accepts_meeting_threshold():
    gate = AutoMergeGate(min_improvement=0.02)
    d = gate.evaluate(baseline=0.70, evolved=0.73, constraints_passed=True)
    assert d.auto_merge is True

def test_gate_rejects_constraint_failure():
    gate = AutoMergeGate(min_improvement=0.02)
    d = gate.evaluate(baseline=0.70, evolved=0.80, constraints_passed=False)
    assert d.auto_merge is False and "constraint" in d.reason.lower()

def test_gate_detects_regression():
    gate = AutoMergeGate(min_improvement=0.02, regression_tolerance=0.01)
    d = gate.evaluate(baseline=0.70, evolved=0.65, constraints_passed=True)
    assert d.auto_merge is False and d.regression is True
