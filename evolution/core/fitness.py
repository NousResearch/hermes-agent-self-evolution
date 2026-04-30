"""Fitness functions for evaluating evolved artifacts.

Two scoring modes:

1. **LLM-as-judge** (preferred for serious runs). Wires `LLMJudge` through
   `init_fitness_metric(config, skill_text)`. Scores correctness,
   procedure-following, and completeness on a 0-1 scale and returns textual
   feedback that GEPA's reflective optimizer can read.

2. **Deterministic multi-signal fallback** (default, zero LLM cost). Combines
   keyword overlap, character n-gram similarity, structural pattern alignment,
   length quality, and content density. Useful for cheap iteration loops and
   for runs where the user does not want eval-set content sent to a third
   party judge model.

Note on `expected_behavior`: the importers and dataset builder document this
field as a *rubric* (what a good response looks like), not as a literal
ground-truth string. The deterministic metric therefore approximates rubric
match rather than ground-truth match — a real LLM judge is recommended when
absolute scores matter.
"""

import re
from dataclasses import dataclass
from typing import Optional

import dspy

from evolution.core.config import EvolutionConfig


@dataclass
class FitnessScore:
    """Multi-dimensional fitness score."""
    correctness: float = 0.0          # Did the agent produce correct output? (0-1)
    procedure_following: float = 0.0  # Did it follow the skill's procedure? (0-1)
    completeness: float = 0.0         # Did it include all necessary detail? (0-1)
    length_penalty: float = 0.0       # Penalty for over-verbosity (0-1, 0 = no penalty)
    feedback: str = ""                # Textual feedback for GEPA's reflective analysis

    @property
    def composite(self) -> float:
        """Weighted composite score."""
        raw = (
            0.4 * self.correctness
            + 0.3 * self.procedure_following
            + 0.3 * self.completeness
        )
        return max(0.0, raw - self.length_penalty)


class LLMJudge:
    """LLM-as-judge scorer with rubric-based evaluation.

    Scores agent outputs on multiple dimensions and provides textual feedback
    that GEPA/MIPROv2 can use for reflective mutation.
    """

    class JudgeSignature(dspy.Signature):
        """Evaluate an agent's response against an expected-behaviour rubric.

        Score the response on three dimensions (0.0 to 1.0 each):
        1. correctness: Did the response correctly address the task?
        2. procedure_following: Did it follow the expected approach/procedure?
        3. completeness: Did it include all necessary details, references,
           examples, and edge cases?

        Completeness is critical — a response that is correct but omits
        important API references, code examples, or error-handling
        instructions is INCOMPLETE. Do NOT reward brevity over thoroughness.
        A longer, detailed response covering all necessary information is
        better than a terse one that skips important details.

        Also provide specific, actionable feedback on what could be improved.
        """
        task_input: str = dspy.InputField(desc="The task the agent was given")
        expected_behavior: str = dspy.InputField(desc="Rubric describing what a good response looks like")
        agent_output: str = dspy.InputField(desc="The agent's actual response")
        skill_text: str = dspy.InputField(desc="The skill/instructions the agent was following")
        correctness: float = dspy.OutputField(desc="Score 0.0-1.0: Did the response correctly address the task?")
        procedure_following: float = dspy.OutputField(desc="Score 0.0-1.0: Did it follow the expected procedure?")
        completeness: float = dspy.OutputField(desc="Score 0.0-1.0: Did it include all necessary details, references, examples, and edge cases? Penalise omissions.")
        feedback: str = dspy.OutputField(desc="Specific, actionable feedback on what could be improved or was missing")

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

        lm = self.config.make_lm(self.config.eval_model)

        with dspy.context(lm=lm):
            result = self.judge(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
                skill_text=skill_text,
            )

        correctness = _parse_score(result.correctness)
        procedure_following = _parse_score(result.procedure_following)
        completeness = _parse_score(result.completeness)

        length_penalty = 0.0
        if artifact_size is not None and max_size is not None and max_size > 0:
            ratio = artifact_size / max_size
            if ratio > 0.9:
                length_penalty = min(0.3, (ratio - 0.9) * 3.0)

        return FitnessScore(
            correctness=correctness,
            procedure_following=procedure_following,
            completeness=completeness,
            length_penalty=length_penalty,
            feedback=str(result.feedback),
        )


# ── Global judge instance (initialized lazily by the runner) ──────────────

_judge: Optional[LLMJudge] = None
_judge_skill_text: str = ""
_judge_max_size: Optional[int] = None


def init_fitness_metric(
    config: EvolutionConfig,
    skill_text: str = "",
    *,
    use_llm_judge: bool = False,
    max_skill_size: Optional[int] = None,
) -> None:
    """Configure the global state read by `skill_fitness_metric`.

    When `use_llm_judge=True`, an LLMJudge is constructed and used as the
    primary scorer; the deterministic multi-signal scorer becomes the fallback
    for transient judge failures. When `use_llm_judge=False` (default) the
    metric is purely deterministic.
    """
    global _judge, _judge_skill_text, _judge_max_size
    _judge = LLMJudge(config) if use_llm_judge else None
    _judge_skill_text = skill_text
    _judge_max_size = max_skill_size


def reset_fitness_metric() -> None:
    """Clear global judge state. Tests use this to avoid cross-test pollution."""
    global _judge, _judge_skill_text, _judge_max_size
    _judge = None
    _judge_skill_text = ""
    _judge_max_size = None


def skill_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> dspy.Prediction:
    """DSPy-compatible metric function for skill optimization.

    Accepts the 5-arg GEPA signature `(gold, pred, trace, pred_name, pred_trace)`
    so it works with both GEPA and the older 3-arg metric API used by MIPROv2.

    Returns a `dspy.Prediction(score: float, feedback: str)`.
    """
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""
    task = getattr(example, "task_input", "") or ""

    if not agent_output.strip():
        return dspy.Prediction(
            score=0.0,
            feedback="Empty output — no response was generated.",
        )

    # ── LLM-as-judge (preferred when initialized) ──────────────────────────
    if _judge is not None:
        try:
            score_obj = _judge.score(
                task_input=task,
                expected_behavior=expected,
                agent_output=agent_output,
                skill_text=_judge_skill_text,
                artifact_size=len(_judge_skill_text) if _judge_skill_text else None,
                max_size=_judge_max_size,
            )
            return dspy.Prediction(
                score=score_obj.composite,
                feedback=score_obj.feedback,
            )
        except Exception as exc:
            # Fall through to deterministic. The feedback string flags the
            # judge failure so the user can see why scores look heuristic.
            judge_failure = f"[judge unavailable: {exc.__class__.__name__}] "
    else:
        judge_failure = ""

    # ── Deterministic multi-signal fallback ────────────────────────────────
    keyword_score = _keyword_overlap_score(expected, agent_output)
    char_ngram_score = _char_ngram_similarity(expected, agent_output, n=3)
    structural_score = _structural_match_score(expected, agent_output)
    length_score = _length_quality_score(expected, agent_output)
    density_score = _content_density_score(agent_output)

    composite = (
        0.25 * keyword_score
        + 0.25 * char_ngram_score
        + 0.20 * structural_score
        + 0.15 * length_score
        + 0.15 * density_score
    )
    score = min(1.0, max(0.0, composite))

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

    feedback = judge_failure + " ".join(feedback_parts)
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
        return 0.5

    expected_meaningful = [w for w in expected_tokens if w not in _STOP_WORDS]
    output_meaningful = [w for w in output_tokens if w not in _STOP_WORDS]

    if not expected_meaningful:
        expected_meaningful = expected_tokens
        output_meaningful = output_tokens

    expected_set = set(expected_meaningful)
    output_set = set(output_meaningful)

    if not expected_set:
        return 0.5

    recall = len(expected_set & output_set) / len(expected_set)
    precision = len(expected_set & output_set) / len(output_set) if output_set else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

    return 0.6 * recall + 0.4 * f1


def _char_ngram_similarity(text_a: str, text_b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity — captures partial word overlap."""
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
    """Check structural pattern alignment between expected and output."""
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
        return 0.6

    matched = expected_features & output_features
    recall = len(matched) / len(expected_features)
    unexpected = output_features - expected_features
    noise_penalty = min(0.2, len(unexpected) * 0.05)

    return max(0.0, 0.3 + 0.7 * recall - noise_penalty)


def _length_quality_score(expected: str, output: str) -> float:
    """Score based on output length relative to expected length."""
    expected_len = len(expected.strip())
    output_len = len(output.strip())

    if expected_len == 0:
        if output_len < 10:
            return 0.2
        elif output_len > 5000:
            return 0.4
        else:
            return 0.8

    if output_len == 0:
        return 0.0

    ratio = output_len / expected_len

    if 0.7 <= ratio <= 1.5:
        return 1.0
    elif 0.4 <= ratio < 0.7:
        return 0.6 + 0.4 * (ratio - 0.4) / 0.3
    elif 1.5 < ratio <= 3.0:
        return 1.0 - 0.4 * (ratio - 1.5) / 1.5
    elif ratio < 0.4:
        return max(0.1, 0.6 * ratio / 0.4)
    else:
        return max(0.1, 0.6 / (ratio / 3.0))


def _content_density_score(text: str) -> float:
    """Measure content density — meaningful tokens / total."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0

    unique_ratio = len(set(tokens)) / len(tokens)
    avg_token_len = sum(len(t) for t in tokens) / len(tokens)
    length_score = min(1.0, avg_token_len / 6.0)

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if len(sentences) > 1:
        first_words = [s.split()[0].lower() for s in sentences if s.split()]
        variety = len(set(first_words)) / len(first_words) if first_words else 0.5
    else:
        variety = 0.5

    return 0.4 * unique_ratio + 0.3 * length_score + 0.3 * variety


def _parse_score(value) -> float:
    """Parse an LLM-emitted score value, clamping to [0, 1]."""
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    try:
        return min(1.0, max(0.0, float(str(value).strip())))
    except (ValueError, TypeError):
        return 0.5
