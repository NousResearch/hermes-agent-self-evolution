"""Island model — multiple populations with ring-topology migration.

Each island has its own population and operator configuration,
enabling diverse search strategies. Migration spreads good solutions
across islands while maintaining genetic isolation.
"""

from __future__ import annotations

import random
import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from evolution.ea.individual import Individual
from evolution.ea.population import Population
from evolution.ea.operators.selection import TournamentSelection, ElitistSelection
from evolution.ea.operators.mutation import LLMMutator, STRATEGIES
from evolution.ea.operators.crossover import LLMCrossover
from evolution.ea.cascade_evaluator import CascadeEvaluator
from evolution.ea.llm_client import LLMClient
from evolution.core.dataset_builder import EvalDataset

logger = logging.getLogger(__name__)


@dataclass
class IslandConfig:
    """Configuration for a single island's evolutionary strategy."""
    island_id: int
    population_size: int = 8
    elite_count: int = 2
    tournament_size: int = 3
    mutation_rate: float = 0.7
    crossover_rate: float = 0.3
    mutation_temperature: float = 1.0
    preferred_strategies: list[str] = field(default_factory=lambda: STRATEGIES.copy())


class Island:
    """A single evolutionary island with its own population and operators."""

    def __init__(
        self,
        config: IslandConfig,
        mutator: LLMMutator,
        crossover: LLMCrossover,
        evaluator: CascadeEvaluator,
        dataset: EvalDataset,
    ):
        self.config = config
        self.population = Population(max_size=config.population_size)
        self.selector = TournamentSelection(config.tournament_size)
        self.elitism = ElitistSelection(config.elite_count)
        self.mutator = mutator
        self.crossover = crossover
        self.evaluator = evaluator
        self.dataset = dataset

    def evolve_one_generation(self, generation: int) -> None:
        """Run one generation of evolution on this island."""
        if self.population.size == 0:
            return

        # Preserve elites
        elites = self.elitism.apply(self.population)
        offspring: list[Individual] = []
        target = self.config.population_size - len(elites)
        max_attempts = target * 3  # Avoid infinite loop on repeated failures
        attempts = 0

        while len(offspring) < target and attempts < max_attempts:
            attempts += 1

            if random.random() < self.config.crossover_rate and self.population.size >= 2:
                # Crossover
                parent_a = self.population.tournament_select(self.config.tournament_size)
                parent_b = self.population.tournament_select(self.config.tournament_size)
                child = self.crossover.apply(parent_a, parent_b, generation=generation)
            else:
                # Mutation
                parent = self.population.tournament_select(self.config.tournament_size)
                feedback = parent.fitness.feedback if parent.fitness else ""
                strategy = random.choice(self.config.preferred_strategies)
                child = self.mutator.apply(
                    parent,
                    strategy=strategy,
                    feedback=feedback,
                    generation=generation,
                )

            if child is not None:
                child.fitness = self.evaluator.evaluate(child, self.dataset)
                offspring.append(child)

        self.population.replace_generation(elites, offspring)

    def export_migrants(self, k: int = 1) -> list[Individual]:
        """Export top-k individuals for migration to another island."""
        return [ind.clone() for ind in self.population.best(k)]

    def receive_migrants(self, migrants: list[Individual]) -> None:
        """Receive migrants from another island, replacing worst individuals."""
        for migrant in migrants:
            migrant.mutation_type = "migrant"
            self.population.add(migrant)


class IslandTopology:
    """Manages multiple islands with ring-topology migration."""

    def __init__(self, islands: list[Island]):
        self.islands = islands

    @classmethod
    def create_diverse(
        cls,
        num_islands: int,
        llm: LLMClient,
        evaluator: CascadeEvaluator,
        dataset: EvalDataset,
        population_size: int = 8,
        elite_count: int = 2,
    ) -> IslandTopology:
        """Create islands with diverse configurations to avoid convergence."""
        temps = [1.0, 1.3, 0.7, 1.5, 0.9]
        mutation_rates = [0.7, 0.6, 0.8, 0.5, 0.9]
        crossover_rates = [0.3, 0.4, 0.2, 0.5, 0.1]
        strategy_sets = [
            STRATEGIES.copy(),
            ["targeted_fix", "rephrase"],
            ["simplify", "restructure"],
            ["elaborate", "targeted_fix"],
            STRATEGIES.copy(),
        ]

        islands = []
        for i in range(num_islands):
            config = IslandConfig(
                island_id=i,
                population_size=population_size,
                elite_count=elite_count,
                mutation_rate=mutation_rates[i % len(mutation_rates)],
                crossover_rate=crossover_rates[i % len(crossover_rates)],
                mutation_temperature=temps[i % len(temps)],
                preferred_strategies=strategy_sets[i % len(strategy_sets)],
            )
            # Each island gets its own mutator with different temperature
            island_llm = LLMClient(
                model=llm.model,
                api_base=llm.api_base,
                api_key=llm.api_key,
                temperature=config.mutation_temperature,
            )
            islands.append(Island(
                config=config,
                mutator=LLMMutator(island_llm),
                crossover=LLMCrossover(island_llm),
                evaluator=evaluator,
                dataset=dataset,
            ))
        return cls(islands)

    def seed_all(self, genome: str) -> None:
        """Seed all islands with the baseline genome."""
        for island in self.islands:
            island.population.seed(
                [genome] * island.config.population_size,
            )

    def migrate_ring(self, k: int = 1) -> None:
        """Ring-topology migration: island[i] receives from island[i-1]."""
        if len(self.islands) < 2:
            return
        migrants = [island.export_migrants(k) for island in self.islands]
        for i in range(len(self.islands)):
            donor = (i - 1) % len(self.islands)
            self.islands[i].receive_migrants(migrants[donor])

    def best_overall(self) -> Individual:
        """Return the best individual across all islands."""
        all_best = []
        for island in self.islands:
            best = island.population.best(1)
            if best:
                all_best.extend(best)
        if not all_best:
            raise ValueError("No individuals in any island")
        return max(all_best, key=lambda x: x.score)

    def evolve_generation_parallel(self, generation: int) -> None:
        """Evolve all islands in parallel for one generation."""
        with ThreadPoolExecutor(max_workers=len(self.islands)) as executor:
            futures = {
                executor.submit(island.evolve_one_generation, generation): island
                for island in self.islands
            }
            for future in as_completed(futures):
                island = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Island {island.config.island_id} failed: {e}")
