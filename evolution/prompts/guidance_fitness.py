"""Fitness for system-prompt evolution: behavioral score x token efficiency - contradiction penalty."""

import re


def behavioral_score(probe_scores: list) -> float:
    """Mean of per-probe behavioral compliance scores."""
    if not probe_scores:
        return 0.0
    return sum(probe_scores) / len(probe_scores)


def token_efficiency_multiplier(evolved_text: str, baseline_text: str) -> float:
    """Return multiplier >= 1.0 for shorter text, <= 1.0 for longer.

    Raises ValueError if evolved text exceeds 120% of baseline length.
    """
    ratio = len(evolved_text) / max(1, len(baseline_text))
    if ratio <= 0.7:
        return 1.15
    if ratio <= 0.85:
        return 1.07
    if ratio <= 1.0:
        return 1.0
    if ratio <= 1.2:
        return 0.95
    raise ValueError(
        f"Evolved text ({len(evolved_text)} chars) exceeds 120% of baseline "
        f"({len(baseline_text)} chars): ratio={ratio:.2f}"
    )


def _word_overlap(text_a: str, text_b: str) -> float:
    """Jaccard word overlap between two texts."""
    words_a = set(re.findall(r'\w+', text_a.lower()))
    words_b = set(re.findall(r'\w+', text_b.lower()))
    if not words_a and not words_b:
        return 1.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / max(1, len(union))


def contradiction_penalty(evolved_texts: dict, threshold: float = 0.15) -> float:
    """Measure semantic overlap between distinct evolved blocks as a contradiction proxy.

    If two distinct evolved blocks have very high word overlap (>0.85),
    they may contain contradictory instructions. Identical blocks do not
    incur a penalty (no contradiction when both say the same thing).
    Returns a penalty score (0.0 to 1.0).
    """
    names = list(evolved_texts.keys())
    max_overlap = 0.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if evolved_texts[names[i]] == evolved_texts[names[j]]:
                continue
            overlap = _word_overlap(evolved_texts[names[i]], evolved_texts[names[j]])
            if overlap > max_overlap:
                max_overlap = overlap
    if max_overlap > 0.85:
        return round((max_overlap - 0.85) / 0.15, 10)
    return 0.0


def composite_fitness(
    behavioral_scores: dict,
    evolved_texts: dict,
    baseline_texts: dict,
) -> float:
    """Composite fitness: mean behavioral x mean efficiency - contradiction penalty."""
    names = [n for n in behavioral_scores if n in evolved_texts and n in baseline_texts]
    if not names:
        return 0.0

    mean_behavioral = sum(behavioral_scores[n] for n in names) / len(names)
    effs = [token_efficiency_multiplier(evolved_texts[n], baseline_texts[n]) for n in names]
    mean_efficiency = sum(effs) / len(effs)

    changed = {n: evolved_texts[n] for n in names if evolved_texts[n] != baseline_texts.get(n, "")}
    c_penalty = contradiction_penalty(changed) if changed else 0.0

    return (mean_behavioral * mean_efficiency) - c_penalty
