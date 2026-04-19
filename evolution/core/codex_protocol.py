"""Prompt/protocol builders for codex-batched self-evolution.

Kept separate from the CLI entrypoint so prompt evolution stays isolated and
merge-friendly across upstream updates.
"""

from __future__ import annotations

import json


def build_mutation_prompt(skill_name: str, baseline_skill: str, iterations: int) -> str:
    return (
        f"Mutation phase for skill {skill_name}. Return only a JSON object with key "
        f"candidate_skill_markdown. Iterations={iterations}. "
        "Make only minimal, surgical edits. Do not rewrite the whole skill. "
        "Preserve structure, tone, frontmatter fields, and overall intent. "
        "Keep growth under 20% versus the baseline and prefer zero or near-zero size growth. "
        "If no clearly better compact variant exists, return a very small edit rather than a broad rewrite. "
        f"Baseline skill:\n{baseline_skill}"
    )


def build_evaluation_prompt(
    skill_name: str,
    baseline_skill: str,
    candidate_skill: str,
    holdout_examples: list[dict],
) -> str:
    return (
        f"Evaluation phase for skill {skill_name}. Return only a JSON object with keys "
        f"baseline_score, candidate_score, improvement, per_example, recommendation. "
        "The recommendation field must be an object with keys winner, reason, confidence. "
        f"Baseline:\n{baseline_skill}\nCandidate:\n{candidate_skill}\nHoldout: {json.dumps(holdout_examples)}"
    )
