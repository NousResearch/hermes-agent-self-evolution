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


# ── Phase 4: Parameter-level accuracy ───────────────────────────────────────────

def _norm_params(params: dict) -> dict:
    """Normalize param dict keys and values for comparison."""
    return {k.strip().lower(): str(v).strip().lower() for k, v in (params or {}).items()}


def param_selection_score(example, prediction, trace=None, pred_name=None, pred_trace=None) -> float:
    """Score for parameter-level accuracy. DSPy-compatible signature.

    Weighted: 0.6 for exact param match, 0.4 for tool match.
    """
    tool_score = selection_score(example, prediction)

    correct_params = _norm_params(getattr(example, "correct_params", {}))
    pred_params = _norm_params(getattr(prediction, "chosen_params", {}))

    if not correct_params:
        return tool_score  # no params to evaluate

    # Per-param similarity: exact value match = 1.0, key exists = 0.5, missing = 0.0
    param_scores = []
    for key, expected_val in correct_params.items():
        if key in pred_params:
            if pred_params[key] == expected_val:
                param_scores.append(1.0)
            else:
                param_scores.append(0.5)
        else:
            param_scores.append(0.0)

    param_acc = sum(param_scores) / len(param_scores)
    return 0.6 * tool_score + 0.4 * param_acc


def per_param_accuracy(pairs: list) -> dict:
    """{tool.param: mean_selection_score} grouped by each example's correct_params."""
    sums, counts = {}, {}
    for example, prediction in pairs:
        correct_params = getattr(example, "correct_params", {}) or {}
        pred_params = getattr(prediction, "chosen_params", {}) or {}
        tool = getattr(example, "correct_tool", "")
        for key, expected_val in correct_params.items():
            label = f"{tool}.{key}"
            val = str(pred_params.get(key, "")).strip().lower()
            expected = str(expected_val).strip().lower()
            score = 1.0 if val == expected else 0.0
            sums[label] = sums.get(label, 0.0) + score
            counts[label] = counts.get(label, 0) + 1
    return {label: sums[label] / counts[label] for label in sums}


def cross_param_penalty(baseline_acc: dict, candidate_acc: dict, threshold: float = 0.05) -> float:
    """Sum of regressions beyond threshold for parameters present in the baseline."""
    penalty = 0.0
    for param, base in baseline_acc.items():
        cand = candidate_acc.get(param, 0.0)
        drop = base - cand
        if drop > threshold:
            penalty += drop - threshold
    return penalty
