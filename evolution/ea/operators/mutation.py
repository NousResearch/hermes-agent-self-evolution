"""LLM-powered mutation operator.

The LLM is a TOOL within the operator, not the controller.
We construct specific prompts for different mutation strategies,
call the LLM, parse and validate the output.
"""

from __future__ import annotations

import random
import logging
from typing import Optional

from evolution.ea.individual import Individual
from evolution.ea.operators.base import GeneticOperator
from evolution.ea.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Mutation strategy prompt templates
MUTATION_PROMPTS = {
    "targeted_fix": """You are improving an agent skill document based on specific feedback.

CURRENT SKILL TEXT:
{genome}

EVALUATION FEEDBACK:
{feedback}

Rewrite the skill text to address the feedback. Keep the overall structure and purpose,
but fix the specific issues mentioned. Output ONLY the improved skill text, nothing else.""",

    "rephrase": """You are rephrasing an agent skill document to be clearer and more precise.

CURRENT SKILL TEXT:
{genome}

Rephrase the instructions to be clearer, more actionable, and less ambiguous.
Do not change the procedure or add new steps — only improve how existing instructions
are communicated. Output ONLY the rephrased skill text, nothing else.""",

    "simplify": """You are simplifying an agent skill document.

CURRENT SKILL TEXT:
{genome}

Simplify the skill by:
- Removing redundant instructions
- Merging overlapping steps
- Making the language more concise
Keep all essential procedures intact. Output ONLY the simplified skill text, nothing else.""",

    "elaborate": """You are enhancing an agent skill document with more detail.

CURRENT SKILL TEXT:
{genome}

Add more specific guidance for edge cases and common failure modes.
Include concrete examples where helpful. Do not change the core procedure.
Output ONLY the enhanced skill text, nothing else.""",

    "restructure": """You are restructuring an agent skill document for better flow.

CURRENT SKILL TEXT:
{genome}

Reorganize the skill for better logical flow:
- Group related instructions together
- Improve the ordering of steps
- Add clearer section headers if needed
Keep the same content and procedures. Output ONLY the restructured skill text, nothing else.""",
}

STRATEGIES = list(MUTATION_PROMPTS.keys())


class LLMMutator(GeneticOperator):
    """Mutates an Individual's genome using LLM-powered rewriting.

    Supports multiple mutation strategies that produce different
    kinds of variations. Strategy selection can be automatic
    (based on fitness feedback) or manual.
    """

    def __init__(self, llm: LLMClient, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries

    def apply(
        self,
        *parents: Individual,
        strategy: str = "auto",
        feedback: str = "",
        generation: int = 0,
        **kwargs,
    ) -> Optional[Individual]:
        """Mutate a single parent Individual.

        Args:
            parents: Exactly one parent Individual.
            strategy: Mutation strategy name, or "auto" to pick based on context.
            feedback: Evaluation feedback to guide targeted mutations.
            generation: Current generation number for the child.
        """
        if not parents:
            return None
        parent = parents[0]

        if strategy == "auto":
            strategy = self._pick_strategy(feedback)

        prompt_template = MUTATION_PROMPTS.get(strategy, MUTATION_PROMPTS["rephrase"])
        prompt = prompt_template.format(genome=parent.genome, feedback=feedback or "No specific feedback.")

        for attempt in range(self.max_retries):
            result = self.llm.complete(prompt, system="You are a skill optimization expert.")
            if result and result.strip() and len(result.strip()) > 50:
                return Individual(
                    genome=result.strip(),
                    generation=generation,
                    parent_ids=[parent.id],
                    mutation_type=strategy,
                )
            logger.warning(f"Mutation attempt {attempt + 1} produced empty/short output")

        return None

    def _pick_strategy(self, feedback: str) -> str:
        """Auto-select mutation strategy based on feedback content."""
        if not feedback:
            return random.choice(STRATEGIES)

        feedback_lower = feedback.lower()
        if any(w in feedback_lower for w in ["incorrect", "wrong", "error", "fail", "miss"]):
            return "targeted_fix"
        if any(w in feedback_lower for w in ["verbose", "long", "redundant", "wordy"]):
            return "simplify"
        if any(w in feedback_lower for w in ["unclear", "ambiguous", "confus"]):
            return "rephrase"
        if any(w in feedback_lower for w in ["incomplete", "missing", "lack", "edge case"]):
            return "elaborate"

        return random.choice(STRATEGIES)
