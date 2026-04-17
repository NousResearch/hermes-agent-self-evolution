"""Cascade evaluator — 3-stage screening for efficient fitness evaluation.

Stage 1: Constraint validation (free, 0 API calls)
Stage 2: Keyword overlap fast screen (free)
Stage 3: LLM-as-Judge full evaluation (costs API calls)

This cascade eliminates most candidates early, saving expensive LLM calls
for only the most promising individuals.
"""

from __future__ import annotations

import logging
from typing import Optional

from evolution.ea.individual import Individual
from evolution.ea.llm_client import LLMClient
from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator
from evolution.core.dataset_builder import EvalDataset, EvalExample
from evolution.core.fitness import FitnessScore, _parse_score

logger = logging.getLogger(__name__)

# LLM judge prompt — replaces the DSPy ChainOfThought JudgeSignature
JUDGE_PROMPT = """Evaluate an agent's response against an expected behavior rubric.

TASK: {task_input}
EXPECTED BEHAVIOR: {expected_behavior}
AGENT OUTPUT: {agent_output}
SKILL INSTRUCTIONS: {skill_text}

Score on three dimensions (0.0 to 1.0 each):
1. correctness: Did the response correctly address the task?
2. procedure_following: Did it follow the expected procedure from the skill?
3. conciseness: Was it appropriately concise without omitting important info?

Also provide specific, actionable feedback on what could be improved.

Respond in this exact JSON format:
{{"correctness": 0.0, "procedure_following": 0.0, "conciseness": 0.0, "feedback": "..."}}"""

# Simple task execution prompt — replaces SkillModule forward pass
TASK_PROMPT = """Follow these skill instructions to complete the task.

SKILL INSTRUCTIONS:
{skill_text}

TASK:
{task_input}

Complete the task following the skill instructions above."""


class CascadeEvaluator:
    """Three-stage cascade evaluator for evolutionary fitness.

    Designed to minimize expensive LLM calls by filtering out
    clearly bad candidates at earlier, cheaper stages.
    """

    def __init__(
        self,
        config: EvolutionConfig,
        llm: LLMClient,
        baseline_text: str,
        fast_screen_threshold: float = 0.3,
    ):
        self.config = config
        self.llm = llm
        self.baseline_text = baseline_text
        self.fast_screen_threshold = fast_screen_threshold
        self.constraint_validator = ConstraintValidator(config)

    def evaluate(
        self,
        individual: Individual,
        dataset: EvalDataset,
        split: str = "val",
    ) -> FitnessScore:
        """Evaluate an individual through the cascade.

        Returns a FitnessScore. Stages are tried in order; early stages
        can reject a candidate without proceeding to expensive stages.
        """
        # Stage 1: Constraint validation (free)
        results = self.constraint_validator.validate_all(
            individual.genome, "skill", baseline_text=self.baseline_text,
        )
        if any(not r.passed for r in results):
            failures = [r.message for r in results if not r.passed]
            return FitnessScore(
                feedback=f"Constraint violations: {'; '.join(failures)}",
            )

        examples = getattr(dataset, split, dataset.val)
        if not examples:
            return FitnessScore(feedback="No evaluation examples available")

        # Stage 2: Keyword overlap fast screen (free, on first 3 examples)
        fast_examples = examples[:3]
        fast_score = self._keyword_screen(individual.genome, fast_examples)
        if fast_score < self.fast_screen_threshold:
            return FitnessScore(
                correctness=fast_score,
                feedback=f"Failed fast screen (score={fast_score:.2f} < {self.fast_screen_threshold})",
            )

        # Stage 3: Full LLM-as-Judge evaluation
        return self._llm_judge(individual.genome, examples)

    def _keyword_screen(self, skill_text: str, examples: list[EvalExample]) -> float:
        """Fast keyword-overlap screening without LLM calls."""
        if not examples:
            return 0.0

        scores = []
        for ex in examples:
            # Simulate what the skill would produce by checking keyword overlap
            expected_words = set(ex.expected_behavior.lower().split())
            skill_words = set(skill_text.lower().split())
            if expected_words:
                overlap = len(expected_words & skill_words) / len(expected_words)
                scores.append(0.3 + 0.7 * overlap)
            else:
                scores.append(0.5)
        return sum(scores) / len(scores)

    def _llm_judge(self, skill_text: str, examples: list[EvalExample]) -> FitnessScore:
        """Full LLM-as-Judge evaluation on all examples."""
        all_correctness = []
        all_procedure = []
        all_conciseness = []
        all_feedback = []

        for ex in examples:
            # Step 1: Execute task with skill
            task_prompt = TASK_PROMPT.format(
                skill_text=skill_text,
                task_input=ex.task_input,
            )
            agent_output = self.llm.complete(task_prompt)
            if not agent_output.strip():
                all_correctness.append(0.0)
                all_procedure.append(0.0)
                all_conciseness.append(0.0)
                all_feedback.append("Empty output")
                continue

            # Step 2: Judge the output
            judge_prompt = JUDGE_PROMPT.format(
                task_input=ex.task_input,
                expected_behavior=ex.expected_behavior,
                agent_output=agent_output,
                skill_text=skill_text,
            )
            result = self.llm.complete_json(judge_prompt, system="You are a fair evaluator.")
            if result:
                all_correctness.append(_parse_score(result.get("correctness", 0.5)))
                all_procedure.append(_parse_score(result.get("procedure_following", 0.5)))
                all_conciseness.append(_parse_score(result.get("conciseness", 0.5)))
                all_feedback.append(result.get("feedback", ""))
            else:
                all_correctness.append(0.5)
                all_procedure.append(0.5)
                all_conciseness.append(0.5)
                all_feedback.append("Judge parse failure")

        n = max(1, len(all_correctness))
        return FitnessScore(
            correctness=sum(all_correctness) / n,
            procedure_following=sum(all_procedure) / n,
            conciseness=sum(all_conciseness) / n,
            feedback="; ".join(f for f in all_feedback if f),
        )
