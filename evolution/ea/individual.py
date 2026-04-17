"""Individual — the genome representation for evolutionary optimization.

Each Individual holds a skill text (the genome) and its fitness score.
Supports lineage tracking for understanding evolutionary history.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from evolution.core.fitness import FitnessScore


@dataclass
class Individual:
    """A single candidate in the evolutionary population.

    The genome is the skill text (markdown). Fitness is assigned
    after evaluation by the CascadeEvaluator.
    """
    genome: str
    fitness: Optional[FitnessScore] = None
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    mutation_type: str = ""  # e.g. "targeted_fix", "crossover", "seed"
    id: str = field(default_factory=lambda: uuid4().hex[:12])

    @property
    def score(self) -> float:
        """Composite fitness score, or 0.0 if not yet evaluated."""
        if self.fitness is None:
            return 0.0
        return self.fitness.composite

    @property
    def is_evaluated(self) -> bool:
        return self.fitness is not None

    def clone(self, new_generation: Optional[int] = None) -> Individual:
        """Deep copy with new ID, preserving lineage."""
        return Individual(
            genome=self.genome,
            fitness=copy.deepcopy(self.fitness),
            generation=new_generation if new_generation is not None else self.generation,
            parent_ids=[self.id],
            mutation_type="clone",
        )

    def __repr__(self) -> str:
        score_str = f"{self.score:.3f}" if self.is_evaluated else "?"
        return f"Individual({self.id}, gen={self.generation}, score={score_str}, type={self.mutation_type})"
