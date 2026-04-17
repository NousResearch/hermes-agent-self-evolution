"""Base class for genetic operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from evolution.ea.individual import Individual


class GeneticOperator(ABC):
    """Abstract base class for all genetic operators.

    Subclasses implement selection, mutation, crossover, etc.
    Each operator takes one or more parent Individuals and produces
    a new Individual (or None on failure).
    """

    @abstractmethod
    def apply(self, *parents: Individual, **kwargs) -> Optional[Individual]:
        """Apply this operator to produce a new Individual.

        Returns None if the operation fails (e.g., LLM produces invalid output).
        """
        ...
