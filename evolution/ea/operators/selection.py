"""Selection operators — tournament and elitist selection."""

from __future__ import annotations

import random
from typing import Optional

from evolution.ea.individual import Individual
from evolution.ea.population import Population
from evolution.ea.operators.base import GeneticOperator


class TournamentSelection(GeneticOperator):
    """Tournament selection: pick k random individuals, return the fittest.

    This is a standard EA selection pressure mechanism — higher k = more
    exploitation (stronger prefer best), lower k = more exploration.
    """

    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    def apply(self, *parents: Individual, population: Optional[Population] = None, **kwargs) -> Optional[Individual]:
        """Select one individual from the population via tournament."""
        if population is None or population.size == 0:
            return None
        return population.tournament_select(k=self.tournament_size)


class ElitistSelection:
    """Elitist selection: preserve the top-k individuals unchanged.

    These elites are carried directly into the next generation,
    ensuring the best solutions are never lost.
    """

    def __init__(self, elite_count: int = 2):
        self.elite_count = elite_count

    def apply(self, population: Population) -> list[Individual]:
        """Return clones of the top-k individuals."""
        return [ind.clone() for ind in population.best(self.elite_count)]
