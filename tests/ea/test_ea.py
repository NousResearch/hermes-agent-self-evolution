"""Tests for the native evolutionary algorithm engine."""

import pytest
from unittest.mock import MagicMock, patch

from evolution.ea.individual import Individual
from evolution.ea.population import Population
from evolution.ea.operators.selection import TournamentSelection, ElitistSelection
from evolution.ea.operators.mutation import LLMMutator, STRATEGIES
from evolution.ea.operators.crossover import LLMCrossover
from evolution.ea.island import IslandConfig, Island, IslandTopology
from evolution.ea.llm_client import LLMClient
from evolution.core.fitness import FitnessScore


# ── Individual ────────────────────────────────────────────────────────


class TestIndividual:
    def test_score_unevaluated(self):
        ind = Individual(genome="test")
        assert ind.score == 0.0
        assert not ind.is_evaluated

    def test_score_evaluated(self):
        ind = Individual(genome="test", fitness=FitnessScore(correctness=0.8, procedure_following=0.6, conciseness=1.0))
        assert ind.is_evaluated
        assert 0.0 < ind.score <= 1.0

    def test_clone_preserves_lineage(self):
        ind = Individual(genome="original", generation=1)
        clone = ind.clone(new_generation=2)
        assert clone.genome == "original"
        assert clone.generation == 2
        assert ind.id in clone.parent_ids
        assert clone.id != ind.id

    def test_unique_ids(self):
        a = Individual(genome="a")
        b = Individual(genome="b")
        assert a.id != b.id

    def test_repr(self):
        ind = Individual(genome="test", mutation_type="seed")
        assert "seed" in repr(ind)


# ── Population ────────────────────────────────────────────────────────


class TestPopulation:
    def test_seed(self):
        pop = Population(max_size=5)
        pop.seed(["a", "b", "c"])
        assert pop.size == 3

    def test_add_evicts_worst(self):
        pop = Population(max_size=3)
        pop.seed(["a", "b", "c"])
        for i, ind in enumerate(pop.individuals):
            ind.fitness = FitnessScore(correctness=i * 0.3)
        pop.add(Individual(genome="d", fitness=FitnessScore(correctness=1.0)))
        assert pop.size == 3
        assert pop.best(1)[0].genome == "d"

    def test_best_returns_sorted(self):
        pop = Population(max_size=5)
        pop.seed(["a", "b", "c"])
        pop.individuals[0].fitness = FitnessScore(correctness=0.3)
        pop.individuals[1].fitness = FitnessScore(correctness=0.9)
        pop.individuals[2].fitness = FitnessScore(correctness=0.1)
        best = pop.best(2)
        assert best[0].genome == "b"
        assert len(best) == 2

    def test_tournament_select(self):
        pop = Population(max_size=10)
        pop.seed(["a", "b", "c", "d", "e"])
        for i, ind in enumerate(pop.individuals):
            ind.fitness = FitnessScore(correctness=i * 0.2)
        selected = pop.tournament_select(k=3)
        assert selected is not None
        assert selected.is_evaluated

    def test_diversity_score_identical(self):
        pop = Population(max_size=5)
        pop.seed(["same text", "same text", "same text"])
        assert pop.diversity_score() == 0.0

    def test_diversity_score_different(self):
        pop = Population(max_size=5)
        pop.seed(["aaaa", "bbbb", "cccc"])
        assert pop.diversity_score() > 0.5

    def test_replace_generation(self):
        pop = Population(max_size=4)
        elites = [Individual(genome="elite", fitness=FitnessScore(correctness=1.0))]
        offspring = [
            Individual(genome=f"off{i}", fitness=FitnessScore(correctness=i * 0.2))
            for i in range(5)
        ]
        pop.replace_generation(elites, offspring)
        assert pop.size == 4
        assert pop.best(1)[0].genome == "elite"

    def test_empty_population_raises_on_select(self):
        pop = Population(max_size=5)
        with pytest.raises(ValueError):
            pop.tournament_select()


# ── Selection ─────────────────────────────────────────────────────────


class TestSelection:
    def test_tournament_selection(self):
        pop = Population(max_size=10)
        pop.seed([f"genome_{i}" for i in range(5)])
        for i, ind in enumerate(pop.individuals):
            ind.fitness = FitnessScore(correctness=i * 0.2)
        selector = TournamentSelection(tournament_size=3)
        selected = selector.apply(population=pop)
        assert selected is not None

    def test_elitist_selection(self):
        pop = Population(max_size=10)
        pop.seed(["a", "b", "c", "d"])
        for i, ind in enumerate(pop.individuals):
            ind.fitness = FitnessScore(correctness=i * 0.3)
        elitism = ElitistSelection(elite_count=2)
        elites = elitism.apply(pop)
        assert len(elites) == 2
        assert elites[0].genome == "d"  # Highest score


# ── Mutation ──────────────────────────────────────────────────────────


class TestMutation:
    def test_strategies_exist(self):
        assert len(STRATEGIES) >= 5
        assert "targeted_fix" in STRATEGIES
        assert "rephrase" in STRATEGIES

    def test_mutator_with_mock_llm(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.complete.return_value = "This is the mutated skill text with enough content to pass."
        mutator = LLMMutator(mock_llm)
        parent = Individual(genome="original skill text", generation=0)
        child = mutator.apply(parent, strategy="rephrase", generation=1)
        assert child is not None
        assert child.generation == 1
        assert child.mutation_type == "rephrase"
        assert parent.id in child.parent_ids

    def test_mutator_returns_none_on_empty(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.complete.return_value = ""
        mutator = LLMMutator(mock_llm, max_retries=1)
        parent = Individual(genome="test")
        child = mutator.apply(parent, strategy="rephrase")
        assert child is None

    def test_auto_strategy_picks_targeted_fix(self):
        mock_llm = MagicMock(spec=LLMClient)
        mutator = LLMMutator(mock_llm)
        strategy = mutator._pick_strategy("The output was incorrect and missed key steps")
        assert strategy == "targeted_fix"

    def test_auto_strategy_picks_simplify(self):
        mock_llm = MagicMock(spec=LLMClient)
        mutator = LLMMutator(mock_llm)
        strategy = mutator._pick_strategy("Response was too verbose and redundant")
        assert strategy == "simplify"


# ── Crossover ─────────────────────────────────────────────────────────


class TestCrossover:
    def test_crossover_with_mock_llm(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.complete.return_value = "This is the merged child skill text with enough content."
        crossover = LLMCrossover(mock_llm)
        parent_a = Individual(genome="skill version A", fitness=FitnessScore(correctness=0.7))
        parent_b = Individual(genome="skill version B", fitness=FitnessScore(correctness=0.5))
        child = crossover.apply(parent_a, parent_b, generation=2)
        assert child is not None
        assert child.mutation_type == "crossover"
        assert len(child.parent_ids) == 2

    def test_crossover_needs_two_parents(self):
        mock_llm = MagicMock(spec=LLMClient)
        crossover = LLMCrossover(mock_llm)
        parent = Individual(genome="only one")
        child = crossover.apply(parent, generation=1)
        assert child is None


# ── Island ────────────────────────────────────────────────────────────


class TestIsland:
    def test_island_config_defaults(self):
        cfg = IslandConfig(island_id=0)
        assert cfg.mutation_rate == 0.7
        assert cfg.crossover_rate == 0.3
        assert cfg.elite_count == 2

    def test_export_migrants(self):
        cfg = IslandConfig(island_id=0, population_size=5)
        mock_llm = MagicMock(spec=LLMClient)
        mock_evaluator = MagicMock()
        mock_dataset = MagicMock()
        island = Island(cfg, LLMMutator(mock_llm), LLMCrossover(mock_llm), mock_evaluator, mock_dataset)
        island.population.seed(["a", "b", "c"])
        for i, ind in enumerate(island.population.individuals):
            ind.fitness = FitnessScore(correctness=i * 0.3)
        migrants = island.export_migrants(k=1)
        assert len(migrants) == 1
        assert migrants[0].genome == "c"  # Highest scoring

    def test_receive_migrants(self):
        cfg = IslandConfig(island_id=0, population_size=3)
        mock_llm = MagicMock(spec=LLMClient)
        island = Island(cfg, LLMMutator(mock_llm), LLMCrossover(mock_llm), MagicMock(), MagicMock())
        island.population.seed(["a", "b"])
        migrant = Individual(genome="migrant", fitness=FitnessScore(correctness=1.0))
        island.receive_migrants([migrant])
        assert island.population.size == 3


# ── IslandTopology ────────────────────────────────────────────────────


class TestIslandTopology:
    def test_create_diverse(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.model = "test"
        mock_llm.api_base = None
        mock_llm.api_key = None
        mock_evaluator = MagicMock()
        mock_dataset = MagicMock()
        topo = IslandTopology.create_diverse(3, mock_llm, mock_evaluator, mock_dataset)
        assert len(topo.islands) == 3
        # Check diversity in temperatures
        temps = [i.config.mutation_temperature for i in topo.islands]
        assert len(set(temps)) >= 2

    def test_seed_all(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.model = "test"
        mock_llm.api_base = None
        mock_llm.api_key = None
        topo = IslandTopology.create_diverse(2, mock_llm, MagicMock(), MagicMock(), population_size=3)
        topo.seed_all("baseline genome")
        for island in topo.islands:
            assert island.population.size == 3

    def test_migrate_ring(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.model = "test"
        mock_llm.api_base = None
        mock_llm.api_key = None
        topo = IslandTopology.create_diverse(2, mock_llm, MagicMock(), MagicMock(), population_size=3)
        topo.seed_all("baseline")
        for island in topo.islands:
            for ind in island.population.individuals:
                ind.fitness = FitnessScore(correctness=0.5)
        topo.migrate_ring(k=1)
        # After migration, populations should still be valid
        for island in topo.islands:
            assert island.population.size <= island.config.population_size + 1  # +1 for migrant before eviction
