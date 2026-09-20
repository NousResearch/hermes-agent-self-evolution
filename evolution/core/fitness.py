"""Fitness functions for evaluating evolved artifacts.

Uses LLM-as-judge with rubrics to score agent outputs.
Supports length penalties and multi-dimensional scoring.
"""

import dspy
from dataclasses import dataclass
from typing import Optional

from evolution.core.config import EvolutionConfig


@dataclass
class FitnessScore:
    """Multi-dimensional fitness score."""
    correctness: float = 0.0  # Did the agent produce correct output? (0-1)
    procedure_following: float = 0.0  # Did it follow the skill's procedure? (0-1)
    conciseness: float = 0.0  # Was it appropriately concise? (0-1)
    length_penalty: float = 0.0  # Penalty for being too verbose (0-1, 0 = no penalty)
    feedback: str = ""  # Textual feedback for GEPA's reflective analysis

    @property
    def composite(self) -> float:
        """Weighted composite score."""
        raw = (
            0.5 * self.correctness
            + 0.3 * self.procedure_following
            + 0.2 * self.conciseness
        )
        return max(0.0, raw - self.length_penalty)


class LLMJudge:
    """LLM-as-judge scorer with rubric-based evaluation.

    Scores agent outputs on multiple dimensions and provides
    textual feedback that GEPA can use for reflective mutation.
    """

    class JudgeSignature(dspy.Signature):
        """Evaluate an agent's response against an expected behavior rubric.

        Score the response on three dimensions (0.0 to 1.0 each):
        1. correctness: Did the response correctly address the task?
        2. procedure_following: Did it follow the expected approach/procedure?
        3. conciseness: Was it appropriately concise without omitting important info?

        Also provide specific, actionable feedback on what could be improved.
        """
        task_input: str = dspy.InputField(desc="The task the agent was given")
        expected_behavior: str = dspy.InputField(desc="Rubric describing what a good response looks like")
        agent_output: str = dspy.InputField(desc="The agent's actual response")
        skill_text: str = dspy.InputField(desc="The skill/instructions the agent was following")
        correctness: float = dspy.OutputField(desc="Score 0.0-1.0: Did the response correctly address the task?")
        procedure_following: float = dspy.OutputField(desc="Score 0.0-1.0: Did it follow the expected procedure?")
        conciseness: float = dspy.OutputField(desc="Score 0.0-1.0: Appropriately concise?")
        feedback: str = dspy.OutputField(desc="Specific, actionable feedback on what could be improved")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.judge = dspy.ChainOfThought(self.JudgeSignature)

    def score(
        self,
        task_input: str,
        expected_behavior: str,
        agent_output: str,
        skill_text: str,
        artifact_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> FitnessScore:
        """Score an agent output using LLM-as-judge."""

        import os as _os_fit
        _proxy = _os_fit.getenv("HERMES_PROXY_URL")
        _pkey = _os_fit.getenv("HERMES_PROXY_KEY", "fake")
        if _proxy:
            lm = dspy.LM(self.config.eval_model, api_base=_proxy, api_key=_pkey, custom_llm_provider="openai")
        else:
            lm = dspy.LM(self.config.eval_model)

        with dspy.context(lm=lm):
            result = self.judge(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
                skill_text=skill_text,
            )

        # Parse scores (clamp to 0-1)
        correctness = _parse_score(result.correctness)
        procedure_following = _parse_score(result.procedure_following)
        conciseness = _parse_score(result.conciseness)

        # Length penalty
        length_penalty = 0.0
        if artifact_size is not None and max_size is not None:
            ratio = artifact_size / max_size
            if ratio > 0.9:
                # Penalty ramps from 0 at 90% to 0.3 at 100%+
                length_penalty = min(0.3, (ratio - 0.9) * 3.0)

        return FitnessScore(
            correctness=correctness,
            procedure_following=procedure_following,
            conciseness=conciseness,
            length_penalty=length_penalty,
            feedback=str(result.feedback),
        )


def skill_fitness_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> float:
    """DSPy-compatible metric function for skill optimization.

    This is what gets passed to dspy.GEPA(metric=...).
    Accepts 5 args per DSPy 3.3.1 GEPA API: (gold, pred, trace, pred_name, pred_trace).
    Returns a float 0-1 score.

    Uses a fast, signal-bearing heuristic for optimization. For accurate final
    evaluation, use LLMJudge directly (see evolve_skill.py holdout eval).
    """
    agent_output = getattr(pred, "output", "") or ""
    expected = getattr(gold, "expected_behavior", "") or ""
    task = getattr(gold, "task_input", "") or ""

    if not agent_output.strip():
        return 0.0

    output_lower = agent_output.lower()
    output_len = len(agent_output)

    # Penalize degenerate outputs (too short = garbage, too long = verbose)
    if output_len < 30:
        return 0.1
    if output_len > 3000:
        length_factor = 0.7
    elif output_len > 2000:
        length_factor = 0.85
    else:
        length_factor = 1.0

    # Extract meaningful expected-behavior phrases (not single words)
    expected_phrases = [p.strip() for p in expected.lower().split(",") if len(p.strip()) > 3]
    if len(expected_phrases) < 2:
        # Fallback: words longer than 4 chars (skip stopwords)
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should", "may", "might", "can", "shall",
                     "this", "that", "these", "those", "it", "its", "and", "or",
                     "but", "not", "no", "if", "then", "with", "for", "from",
                     "into", "about", "between", "through", "during", "before",
                     "after", "above", "below", "to", "of", "in", "on", "at",
                     "by", "than", "too", "very", "just", "because", "as", "until",
                     "while", "where", "when", "what", "which", "who", "how", "why"}
        expected_phrases = [w for w in expected.lower().split()
                           if len(w) > 4 and w not in stopwords]

    if not expected_phrases:
        return 0.2 * length_factor

    # Phrase coverage: what fraction of expected concepts appear in output
    phrase_hits = sum(1 for p in expected_phrases if p in output_lower)
    phrase_coverage = phrase_hits / len(expected_phrases)

    # Task relevance: does output actually address the specific task?
    task_content_words = set(w for w in task.lower().split()
                            if len(w) > 3 and w not in {
                                "the", "a", "an", "is", "are", "was", "were", "be",
                                "have", "has", "had", "do", "does", "did", "will",
                                "would", "could", "should", "may", "might", "can",
                                "this", "that", "these", "those", "it", "its", "and",
                                "or", "but", "not", "if", "then", "with", "for",
                                "from", "into", "about", "between", "how", "what",
                                "which", "who", "when", "where", "why", "your", "you",
                                "please", "using", "provide", "create", "make", "write",
                                "generate", "show", "give", "tell", "explain", "help"
                            })
    output_word_set = set(output_lower.split())
    task_overlap = len(task_content_words & output_word_set) / max(1, len(task_content_words))

    # Weighted combination, scaled by length factor
    score = (0.65 * phrase_coverage + 0.35 * task_overlap) * length_factor
    return min(1.0, max(0.0, score))


def _parse_score(value) -> float:
    """Parse a score value, handling various LLM output formats."""
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    try:
        return min(1.0, max(0.0, float(str(value).strip())))
    except (ValueError, TypeError):
        return 0.5  # Default to neutral on parse failure
