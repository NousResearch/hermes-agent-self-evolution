"""LLM-powered crossover operator.

Merges the best aspects of two parent skill texts into a new child.
This is a capability that DSPy GEPA/MIPROv2 does not support —
it's a uniquely evolutionary approach to prompt optimization.
"""

from __future__ import annotations

import logging
from typing import Optional

from evolution.ea.individual import Individual
from evolution.ea.operators.base import GeneticOperator
from evolution.ea.llm_client import LLMClient

logger = logging.getLogger(__name__)

CROSSOVER_PROMPT = """You are merging two versions of an agent skill document into a better version.

PARENT A (score: {score_a:.2f}):
{genome_a}

PARENT B (score: {score_b:.2f}):
{genome_b}

Create a new version that combines the strengths of both parents:
- Take the better-structured procedures from whichever parent is clearer
- Keep the more precise instructions from whichever parent is more specific
- Merge complementary content (e.g., if A has edge cases B lacks, include them)
- Resolve any contradictions by favoring the higher-scoring parent

Output ONLY the merged skill text, nothing else."""


class LLMCrossover(GeneticOperator):
    """Crossover operator that merges two parent genomes via LLM.

    The LLM acts as an intelligent recombination function,
    understanding the semantic content of both parents and
    producing a child that inherits the best aspects of each.
    """

    def __init__(self, llm: LLMClient, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries

    def apply(
        self,
        *parents: Individual,
        generation: int = 0,
        **kwargs,
    ) -> Optional[Individual]:
        """Crossover two parent Individuals.

        Args:
            parents: Exactly two parent Individuals.
            generation: Current generation number for the child.
        """
        if len(parents) < 2:
            return None
        parent_a, parent_b = parents[0], parents[1]

        prompt = CROSSOVER_PROMPT.format(
            genome_a=parent_a.genome,
            genome_b=parent_b.genome,
            score_a=parent_a.score,
            score_b=parent_b.score,
        )

        for attempt in range(self.max_retries):
            result = self.llm.complete(prompt, system="You are a skill optimization expert.")
            if result and result.strip() and len(result.strip()) > 50:
                return Individual(
                    genome=result.strip(),
                    generation=generation,
                    parent_ids=[parent_a.id, parent_b.id],
                    mutation_type="crossover",
                )
            logger.warning(f"Crossover attempt {attempt + 1} produced empty/short output")

        return None
