"""Evolution engine — the main generational loop.

Orchestrates the full evolutionary process:
initialization → generation loop → migration → stagnation detection → result.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Callable

from evolution.ea.individual import Individual
from evolution.ea.population import Population
from evolution.ea.island import IslandTopology
from evolution.ea.cascade_evaluator import CascadeEvaluator
from evolution.ea.llm_client import LLMClient
from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import EvalDataset
from evolution.core.fitness import FitnessScore

logger = logging.getLogger(__name__)


class EvolutionEngine:
    """Main evolutionary optimization engine.

    Manages the full lifecycle: island initialization, generational loop
    with parallel island evolution, periodic migration, stagnation
    detection with diversity injection, and final holdout evaluation.
    """

    def __init__(
        self,
        config: EvolutionConfig,
        dataset: EvalDataset,
        baseline_text: str,
        num_islands: int = 3,
        num_generations: int = 10,
        population_size: int = 8,
        migration_interval: int = 2,
        stagnation_limit: int = 4,
        on_generation: Optional[Callable] = None,
    ):
        self.config = config
        self.dataset = dataset
        self.baseline_text = baseline_text
        self.num_generations = num_generations
        self.migration_interval = migration_interval
        self.stagnation_limit = stagnation_limit
        self.on_generation = on_generation

        # Create LLM clients
        self.llm = LLMClient(
            model=config.optimizer_model,
            api_base=config.api_base,
            api_key=config.api_key,
        )
        self.eval_llm = LLMClient(
            model=config.eval_model,
            api_base=config.api_base,
            api_key=config.api_key,
            temperature=0.0,  # Deterministic for evaluation
        )

        # Create evaluator
        self.evaluator = CascadeEvaluator(
            config=config,
            llm=self.eval_llm,
            baseline_text=baseline_text,
        )

        # Create island topology
        self.topology = IslandTopology.create_diverse(
            num_islands=num_islands,
            llm=self.llm,
            evaluator=self.evaluator,
            dataset=dataset,
            population_size=population_size,
        )

    def run(self) -> Individual:
        """Execute the full evolutionary loop. Returns the best Individual."""
        start_time = time.time()

        # 1. Seed all islands with baseline
        self.topology.seed_all(self.baseline_text)

        # Evaluate initial population
        for island in self.topology.islands:
            for ind in island.population.individuals:
                ind.fitness = self.evaluator.evaluate(ind, self.dataset)

        global_best = self.topology.best_overall()
        stagnation_counter = 0

        # 2. Generational loop
        for gen in range(1, self.num_generations + 1):
            gen_start = time.time()

            # Evolve all islands in parallel
            self.topology.evolve_generation_parallel(gen)

            # Migration
            if gen % self.migration_interval == 0:
                self.topology.migrate_ring(k=1)

            # Track best
            prev_best_score = global_best.score
            current_best = self.topology.best_overall()
            if current_best.score > global_best.score:
                global_best = current_best

            # Stagnation detection
            if current_best.score <= prev_best_score:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            if stagnation_counter >= self.stagnation_limit:
                self._inject_diversity(global_best, gen)
                stagnation_counter = 0

            gen_elapsed = time.time() - gen_start

            # Progress callback
            if self.on_generation:
                self.on_generation(
                    generation=gen,
                    best_score=global_best.score,
                    island_scores=[
                        island.population.best(1)[0].score
                        if island.population.size > 0 else 0.0
                        for island in self.topology.islands
                    ],
                    island_diversity=[
                        island.population.diversity_score()
                        for island in self.topology.islands
                    ],
                    stagnation=stagnation_counter,
                    elapsed=gen_elapsed,
                    migrated=(gen % self.migration_interval == 0),
                )

        # 3. Final holdout evaluation on the global best
        global_best.fitness = self.evaluator.evaluate(
            global_best, self.dataset, split="holdout",
        )

        total_elapsed = time.time() - start_time
        logger.info(f"Evolution completed in {total_elapsed:.1f}s, best score={global_best.score:.3f}")

        return global_best

    def _inject_diversity(self, global_best: Individual, generation: int) -> None:
        """Inject diversity when evolution stagnates.

        Reset the worst-performing island with aggressive mutations
        of the global best individual.
        """
        # Find worst island
        worst_island = min(
            self.topology.islands,
            key=lambda i: i.population.avg_score(),
        )
        logger.info(f"Stagnation detected — injecting diversity into Island {worst_island.config.island_id}")

        # Generate diverse mutations of the global best
        new_genomes = []
        strategies = ["restructure", "elaborate", "simplify", "rephrase", "targeted_fix"]
        for strategy in strategies[:worst_island.config.population_size]:
            child = worst_island.mutator.apply(
                global_best,
                strategy=strategy,
                feedback="Create a substantially different version.",
                generation=generation,
            )
            if child:
                child.fitness = self.evaluator.evaluate(child, self.dataset)
                new_genomes.append(child)

        if new_genomes:
            # Keep the elite, replace rest with diverse mutations
            elite = worst_island.population.best(1)
            worst_island.population.replace_generation(elite, new_genomes)
