"""Fitness functions for evaluating evolved artifacts.

Uses LLM-as-judge with rubrics to score agent outputs.
Supports length penalties and multi-dimensional scoring.
"""

import re
from dataclasses import dataclass
from typing import Optional

import dspy

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


def skill_fitness_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> dspy.Prediction:
    """DSPy-compatible metric function for skill optimization.

    This is what gets passed to dspy.GEPA(metric=...).
    Returns a dspy.Prediction with score (float 0-1) and feedback (str).

    Scoring components (weighted):
    - keyword_overlap: Traditional word-level overlap with stop-word filtering (25%)
    - char_ngram_similarity: Character 3-gram Jaccard similarity (25%)
    - structural_match: Structural pattern alignment (20%)
    - length_quality: Output length reasonableness (15%)
    - content_density: Ratio of meaningful content to total output (15%)
    """
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""
    task = getattr(example, "task_input", "") or ""

    if not agent_output.strip():
        return dspy.Prediction(score=0.0, feedback="Empty output — no response was generated.")

    # --- Component 1: Keyword overlap (with stop-word filtering) ---
    keyword_score = _keyword_overlap_score(expected, agent_output)

    # --- Component 2: Character n-gram similarity ---
    char_ngram_score = _char_ngram_similarity(expected, agent_output, n=3)

    # --- Component 3: Structural matching ---
    structural_score = _structural_match_score(expected, agent_output)

    # --- Component 4: Length quality ---
    length_score = _length_quality_score(expected, agent_output)

    # --- Component 5: Content density ---
    density_score = _content_density_score(agent_output)

    # Weighted composite
    composite = (
        0.25 * keyword_score
        + 0.25 * char_ngram_score
        + 0.20 * structural_score
        + 0.15 * length_score
        + 0.15 * density_score
    )

    score = min(1.0, max(0.0, composite))

    # Build feedback string for GEPA reflective mutation
    feedback_parts = []
    if score < 0.3:
        feedback_parts.append("Output has very low alignment with expected behavior.")
    if keyword_score < 0.2:
        feedback_parts.append(f"Low keyword overlap ({keyword_score:.2f}) — key terms from expected output are missing.")
    if char_ngram_score < 0.2:
        feedback_parts.append(f"Low character similarity ({char_ngram_score:.2f}) — wording differs significantly from expected.")
    if structural_score < 0.3:
        feedback_parts.append(f"Structural mismatch ({structural_score:.2f}) — output format differs from expected patterns.")
    if length_score < 0.5:
        feedback_parts.append(f"Length issue ({length_score:.2f}) — output is too short or too long relative to expectation.")
    if density_score < 0.3:
        feedback_parts.append(f"Low content density ({density_score:.2f}) — output contains excessive filler or repetition.")
    if not feedback_parts and score >= 0.7:
        feedback_parts.append(f"Good overall quality (score: {score:.2f}). All scoring components above threshold.")
    elif not feedback_parts:
        feedback_parts.append(f"Moderate quality (score: {score:.2f}). Room for improvement in alignment with expected output.")

    feedback = " ".join(feedback_parts)
    return dspy.Prediction(score=score, feedback=feedback)


# ── Stop words for keyword scoring ──────────────────────────────────────────

_STOP_WORDS = frozenset(
    "a an the and or but in on at to for of is it this that with from by as are was were be been "
    "has have had do does did will would shall should may might can could not no nor so if then than "
    "too very just about also more most other some such only own same both each few most much many "
    "into over after before between through during above below up down out off all any".split()
)


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, filtering punctuation."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1]


def _keyword_overlap_score(expected: str, output: str) -> float:
    """Keyword overlap with stop-word filtering and TF-weighted scoring."""
    expected_tokens = _tokenize(expected)
    output_tokens = _tokenize(output)

    if not expected_tokens:
        return 0.5  # Neutral when no expected keywords

    # Filter stop words
    expected_meaningful = [w for w in expected_tokens if w not in _STOP_WORDS]
    output_meaningful = [w for w in output_tokens if w not in _STOP_WORDS]

    if not expected_meaningful:
        # All expected words are stop words — fall back to all tokens
        expected_meaningful = expected_tokens
        output_meaningful = output_tokens

    expected_set = set(expected_meaningful)
    output_set = set(output_meaningful)

    if not expected_set:
        return 0.5

    # Jaccard-like: intersection over expected (recall-oriented)
    recall = len(expected_set & output_set) / len(expected_set)

    # Bonus for precision (how much of output is relevant)
    if output_set:
        precision = len(expected_set & output_set) / len(output_set)
    else:
        precision = 0.0

    # F1-like combination
    if recall + precision > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0.0

    # Blend recall-heavy (0.6 recall + 0.4 F1)
    return 0.6 * recall + 0.4 * f1


def _char_ngram_similarity(text_a: str, text_b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity — captures partial word/substring overlap."""
    def get_ngrams(text: str) -> set[str]:
        text = text.lower().strip()
        if len(text) < n:
            return {text} if text else set()
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    ngrams_a = get_ngrams(text_a)
    ngrams_b = get_ngrams(text_b)

    if not ngrams_a and not ngrams_b:
        return 0.5
    if not ngrams_a or not ngrams_b:
        return 0.0

    intersection = ngrams_a & ngrams_b
    union = ngrams_a | ngrams_b

    return len(intersection) / len(union) if union else 0.0


def _structural_match_score(expected: str, output: str) -> float:
    """Check structural pattern alignment between expected and output.

    Looks for: code blocks, bullet lists, numbered lists, headers,
    inline code, URLs, and paragraph structure.
    """
    patterns = {
        "code_block": r"```[\s\S]*?```",
        "inline_code": r"`[^`]+`",
        "bullet_list": r"(?:^|\n)\s*[-*•]\s+",
        "numbered_list": r"(?:^|\n)\s*\d+[.)]\s+",
        "header": r"(?:^|\n)#{1,6}\s+",
        "url": r"https?://\S+",
        "bold": r"\*\*[^*]+\*\*",
        "paragraph_break": r"\n{2,}",
    }

    expected_features = set()
    output_features = set()

    for name, pattern in patterns.items():
        if re.search(pattern, expected):
            expected_features.add(name)
        if re.search(pattern, output):
            output_features.add(name)

    if not expected_features:
        # No structural expectations — give moderate score
        return 0.6

    # How many expected features are present in output?
    matched = expected_features & output_features
    recall = len(matched) / len(expected_features)

    # Penalize unexpected structural features (noise)
    unexpected = output_features - expected_features
    noise_penalty = min(0.2, len(unexpected) * 0.05)

    return max(0.0, 0.3 + 0.7 * recall - noise_penalty)


def _length_quality_score(expected: str, output: str) -> float:
    """Score based on output length relative to expected length.

    Outputs that are proportionally similar in length to expected
    receive higher scores. Very short or very long outputs are penalized.
    """
    expected_len = len(expected.strip())
    output_len = len(output.strip())

    if expected_len == 0:
        # No expected length — score based on reasonable output length
        if output_len < 10:
            return 0.2  # Too short
        elif output_len > 5000:
            return 0.4  # Possibly too verbose
        else:
            return 0.8

    if output_len == 0:
        return 0.0

    ratio = output_len / expected_len

    # Ideal ratio is close to 1.0, with tolerance for 0.5x-2.0x
    if 0.7 <= ratio <= 1.5:
        return 1.0
    elif 0.4 <= ratio < 0.7:
        # Somewhat short
        return 0.6 + 0.4 * (ratio - 0.4) / 0.3
    elif 1.5 < ratio <= 3.0:
        # Somewhat long
        return 1.0 - 0.4 * (ratio - 1.5) / 1.5
    elif ratio < 0.4:
        # Very short
        return max(0.1, 0.6 * ratio / 0.4)
    else:
        # Very long (>3x)
        return max(0.1, 0.6 / (ratio / 3.0))


def _content_density_score(text: str) -> float:
    """Measure content density — ratio of meaningful tokens to total.

    High density = concise, information-rich output.
    Low density = excessive filler, repetition, or padding.
    """
    tokens = _tokenize(text)
    if not tokens:
        return 0.0

    # Unique token ratio (penalize repetition)
    unique_ratio = len(set(tokens)) / len(tokens)

    # Average token length (longer tokens tend to be more meaningful)
    avg_token_len = sum(len(t) for t in tokens) / len(tokens)
    length_score = min(1.0, avg_token_len / 6.0)  # Normalize: 6+ char avg = 1.0

    # Sentence-level: check for varied sentence structure
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if len(sentences) > 1:
        # Penalize if all sentences start the same way
        first_words = [s.split()[0].lower() for s in sentences if s.split()]
        variety = len(set(first_words)) / len(first_words) if first_words else 0.5
    else:
        variety = 0.5  # Single sentence — neutral

    return 0.4 * unique_ratio + 0.3 * length_score + 0.3 * variety


def _parse_score(value) -> float:
    """Parse a score value, handling various LLM output formats."""
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    try:
        return min(1.0, max(0.0, float(str(value).strip())))
    except (ValueError, TypeError):
        return 0.5  # Default to neutral on parse failure
