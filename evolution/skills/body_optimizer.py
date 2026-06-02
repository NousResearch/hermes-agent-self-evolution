"""Direct optimization of SKILL.md body text as an artifact.

The original DSPy wrapper passes the skill body as a normal input field, which
means GEPA/MIPRO can improve the predictor instructions or demos without ever
rewriting the SKILL.md body itself. This module makes the artifact explicit:
given the current body and an eval brief, ask the optimizer model to return a
revised body candidate, then let the caller validate/evaluate that candidate.
"""

from __future__ import annotations

from typing import Any

import dspy

from evolution.core.dataset_builder import EvalDataset


class SkillBodyRewrite(dspy.Signature):
    """Rewrite a Hermes Agent skill body to better satisfy the eval rubrics.

    Preserve the skill's purpose, major modes, linked-reference guidance, safety
    boundaries, and markdown structure. Improve only the body text: procedures,
    triggers, quality gates, pitfalls, and verification steps. Do not include
    YAML frontmatter. Avoid aggressive summarization: the revised body should
    usually stay within 85-115% of the original length unless it removes clear
    duplication while preserving every operational mode.
    """

    current_body: str = dspy.InputField(desc="Current SKILL.md markdown body without YAML frontmatter")
    eval_brief: str = dspy.InputField(desc="Train/validation tasks and expected behavior rubrics")
    previous_feedback: str = dspy.InputField(desc="Feedback from previous rewrite attempts")
    evolved_body: str = dspy.OutputField(desc="Rewritten SKILL.md markdown body only, no YAML frontmatter")


def build_eval_brief(dataset: EvalDataset, max_examples: int = 12) -> str:
    """Return a compact rubric brief from train/validation examples."""

    lines: list[str] = []
    examples = dataset.train + dataset.val
    for idx, ex in enumerate(examples[:max_examples], start=1):
        category = f" [{ex.category}]" if ex.category else ""
        lines.append(f"{idx}.{category} Task: {ex.task_input}")
        lines.append(f"   Expected: {ex.expected_behavior}")
    return "\n".join(lines)


def optimize_skill_body(
    baseline_body: str,
    dataset: EvalDataset,
    optimizer_model: str,
    iterations: int = 1,
    generator: Any | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate an evolved SKILL.md body candidate.

    Args:
        baseline_body: Markdown body without YAML frontmatter.
        dataset: Evaluation dataset whose train/val splits define target behavior.
        optimizer_model: DSPy/LiteLLM model name for candidate generation.
        iterations: Number of rewrite passes. Each pass rewrites the latest body.
        generator: Optional test seam. Callable with current_body/eval_brief/
            previous_feedback returning an object with ``evolved_body``.

    Returns:
        (evolved_body, metadata). Empty candidates are rejected and leave the
        latest accepted body unchanged.
    """

    current = baseline_body
    eval_brief = build_eval_brief(dataset)
    previous_feedback = (
        "Improve the skill body so future Hermes runs satisfy these rubrics more reliably. "
        "Preserve the original purpose, Korean clinic safety posture, linked reference filenames, "
        "durable-output workflows, card-news mode, and all operational boundaries. Do not compress "
        "the skill into a generic summary; produce a complete replacement body."
    )
    rejected_empty = 0
    iterations_run = 0

    rewriter = generator or dspy.ChainOfThought(SkillBodyRewrite)

    for _ in range(max(1, iterations)):
        iterations_run += 1
        if generator is None:
            lm = dspy.LM(optimizer_model)
            with dspy.context(lm=lm):
                result = rewriter(
                    current_body=current,
                    eval_brief=eval_brief,
                    previous_feedback=previous_feedback,
                )
        else:
            result = rewriter(
                current_body=current,
                eval_brief=eval_brief,
                previous_feedback=previous_feedback,
            )

        candidate = str(getattr(result, "evolved_body", "")).strip()
        if not candidate:
            rejected_empty += 1
            previous_feedback = "Previous rewrite returned an empty body. Return a complete markdown skill body."
            continue

        current = candidate
        previous_feedback = "Use the latest accepted body as the baseline and make only high-value refinements."

    metadata = {
        "changed": current != baseline_body,
        "iterations_run": iterations_run,
        "rejected_empty": rejected_empty,
        "eval_brief_examples": min(len(dataset.train) + len(dataset.val), 12),
    }
    return current, metadata
