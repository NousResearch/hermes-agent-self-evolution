"""Population — collection of Individuals with selection and diversity tracking."""

from __future__ import annotations

import random
from difflib import SequenceMatcher

from evolution.ea.individual import Individual


class Population:
    """A managed collection of Individuals.

    Supports tournament selection, elitism, diversity measurement,
    and automatic size enforcement.
    """

    def __init__(self, max_size: int = 8):
        self.max_size = max_size
        self._individuals: list[Individual] = []

    @property
    def individuals(self) -> list[Individual]:
        return self._individuals

    @individuals.setter
    def individuals(self, value: list[Individual]) -> None:
        self._individuals = value

    @property
    def size(self) -> int:
        return len(self._individuals)

    def seed(self, genomes: list[str], generation: int = 0) -> None:
        """Initialize population from raw text genomes."""
        self._individuals = [
            Individual(genome=g, generation=generation, mutation_type="seed")
            for g in genomes
        ]

    def add(self, ind: Individual) -> None:
        """Add an individual, evicting the worst if at capacity."""
        self._individuals.append(ind)
        if self.size > self.max_size:
            self._individuals.sort(key=lambda x: x.score, reverse=True)
            self._individuals = self._individuals[:self.max_size]

    def best(self, k: int = 1) -> list[Individual]:
        """Return the top-k individuals by fitness score."""
        sorted_pop = sorted(self._individuals, key=lambda x: x.score, reverse=True)
        return sorted_pop[:k]

    def worst(self, k: int = 1) -> list[Individual]:
        """Return the bottom-k individuals by fitness score."""
        sorted_pop = sorted(self._individuals, key=lambda x: x.score)
        return sorted_pop[:k]

    def tournament_select(self, k: int = 3) -> Individual:
        """Tournament selection: pick k random individuals, return the fittest."""
        if self.size == 0:
            raise ValueError("Cannot select from empty population")
        k = min(k, self.size)
        contestants = random.sample(self._individuals, k)
        return max(contestants, key=lambda x: x.score)

    def diversity_score(self) -> float:
        """Compute average pairwise dissimilarity of genomes (0=identical, 1=completely different)."""
        if self.size < 2:
            return 0.0
        pairs = 0
        total_dissim = 0.0
        for i in range(self.size):
            for j in range(i + 1, self.size):
                sim = SequenceMatcher(
                    None,
                    self._individuals[i].genome,
                    self._individuals[j].genome,
                ).ratio()
                total_dissim += 1.0 - sim
                pairs += 1
        return total_dissim / pairs

    def avg_score(self) -> float:
        """Average fitness score across the population."""
        if self.size == 0:
            return 0.0
        return sum(ind.score for ind in self._individuals) / self.size

    def replace_generation(self, elites: list[Individual], offspring: list[Individual]) -> None:
        """Replace current population with elites + best offspring, respecting max_size."""
        combined = elites + offspring
        combined.sort(key=lambda x: x.score, reverse=True)
        self._individuals = combined[:self.max_size]

    def __repr__(self) -> str:
        return f"Population(size={self.size}/{self.max_size}, avg={self.avg_score():.3f})"
