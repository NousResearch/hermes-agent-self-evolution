"""Fitness for tool-description evolution: selection accuracy + cross-tool penalty."""


def _norm(tool: str) -> str:
    return (tool or "").strip().lower()


def selection_score(example, prediction, trace=None, pred_name=None, pred_trace=None) -> float:
    """Per-example score: 1.0 exact, 0.5 acceptable alt, 0.0 miss.

    DSPy-compatible signature (extra args ignored) so it can serve as a metric.
    """
    chosen = _norm(getattr(prediction, "chosen_tool", ""))
    correct = _norm(getattr(example, "correct_tool", ""))
    alts = {_norm(a) for a in getattr(example, "alt_tools", []) or []}
    if chosen == correct:
        return 1.0
    if chosen in alts:
        return 0.5
    return 0.0


def per_tool_accuracy(pairs: list) -> dict:
    """{correct_tool: mean selection_score} grouped by each example's correct_tool."""
    sums, counts = {}, {}
    for example, prediction in pairs:
        tool = getattr(example, "correct_tool", "")
        sums[tool] = sums.get(tool, 0.0) + selection_score(example, prediction)
        counts[tool] = counts.get(tool, 0) + 1
    return {tool: sums[tool] / counts[tool] for tool in sums}


def cross_tool_penalty(baseline_acc: dict, candidate_acc: dict, threshold: float = 0.05) -> float:
    """Sum of regressions beyond `threshold` for tools present in the baseline."""
    penalty = 0.0
    for tool, base in baseline_acc.items():
        cand = candidate_acc.get(tool, 0.0)
        drop = base - cand
        if drop > threshold:
            penalty += drop - threshold
    return penalty


def overall_accuracy(pairs: list) -> float:
    if not pairs:
        return 0.0
    return sum(selection_score(e, p) for e, p in pairs) / len(pairs)


def fitness_with_penalty(pairs: list, baseline_acc: dict, threshold: float = 0.05) -> float:
    """Overall accuracy minus the cross-tool regression penalty."""
    cand_acc = per_tool_accuracy(pairs)
    return overall_accuracy(pairs) - cross_tool_penalty(baseline_acc, cand_acc, threshold)
