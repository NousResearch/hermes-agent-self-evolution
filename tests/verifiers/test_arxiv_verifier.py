"""Tests for the arxiv objective verifier.

Everything here runs offline against the embedded ground truth; the live
arXiv API is only touched by the module's --validate CLI flag.
"""

import dspy
import pytest

from evolution.core.verifier import verifier_metric
from evolution.verifiers.arxiv_verifier import (
    ARXIV_ID_RE,
    KIND_AUTHOR,
    KIND_ID,
    KIND_TITLE,
    KIND_YEAR,
    PAPER_BY_ID,
    SEED_PAPERS,
    ArxivVerifier,
    _conciseness,
    grade_arxiv_id,
    grade_title,
    grade_year,
    task_input_for,
)

ATTENTION = PAPER_BY_ID["1706.03762"]
GPT3 = PAPER_BY_ID["2005.14165"]


@pytest.fixture()
def verifier():
    return ArxivVerifier()


class TestSeedPapers:
    def test_ids_are_unique_and_well_formed(self):
        ids = [p.arxiv_id for p in SEED_PAPERS]
        assert len(ids) == len(set(ids))
        for arxiv_id in ids:
            assert ARXIV_ID_RE.fullmatch(arxiv_id)

    def test_years_derive_from_id_prefix(self):
        for paper in SEED_PAPERS:
            assert 2013 <= paper.year <= 2025


class TestDataset:
    def test_default_build_shape(self, verifier):
        dataset = verifier.build_dataset()
        assert len(dataset.all_examples) == 24
        assert len(dataset.train) == 12
        assert len(dataset.val) == 6
        assert len(dataset.holdout) == 6

    def test_papers_are_disjoint_across_splits(self, verifier):
        dataset = verifier.build_dataset()

        groups = {
            split: {
                verifier.ground_truth_for(example.task_input).paper_id
                for example in getattr(dataset, split)
            }
            for split in ("train", "val", "holdout")
        }

        assert groups["train"].isdisjoint(groups["val"])
        assert groups["train"].isdisjoint(groups["holdout"])
        assert groups["val"].isdisjoint(groups["holdout"])

    def test_too_few_cases_is_rejected(self, verifier):
        with pytest.raises(ValueError, match="at least 3"):
            verifier.build_dataset(num_cases=2)

    def test_every_example_is_gradable(self, verifier):
        dataset = verifier.build_dataset()
        for example in dataset.all_examples:
            assert verifier.ground_truth_for(example.task_input) is not None
            assert example.expected_behavior
            assert example.source == "verifier"
            assert example.category in {KIND_ID, KIND_TITLE, KIND_AUTHOR, KIND_YEAR}

    def test_build_is_deterministic(self, verifier):
        first = [ex.task_input for ex in verifier.build_dataset().all_examples]
        second = [ex.task_input for ex in ArxivVerifier().build_dataset().all_examples]
        assert first == second

    def test_oversized_request_returns_full_pool(self, verifier):
        dataset = verifier.build_dataset(num_cases=10_000)
        assert len(dataset.all_examples) == len(SEED_PAPERS) * 4


class TestIdGrading:
    def test_correct_id_scores_full(self, verifier):
        task = task_input_for(ATTENTION, KIND_ID)
        fitness = verifier.score(task, "The paper is arXiv 1706.03762.")
        assert fitness.correctness == pytest.approx(1.0)
        assert fitness.composite == pytest.approx(1.0)

    def test_versioned_id_still_matches(self, verifier):
        task = task_input_for(ATTENTION, KIND_ID)
        fitness = verifier.score(task, "See 1706.03762v7 on arXiv.")
        assert fitness.correctness == pytest.approx(1.0)

    def test_wrong_id_scores_zero_correctness(self, verifier):
        task = task_input_for(ATTENTION, KIND_ID)
        fitness = verifier.score(task, "That would be arXiv 1706.03799.")
        assert fitness.correctness == pytest.approx(0.0)
        assert "1706.03762" in fitness.feedback

    def test_wrong_but_well_formed_id_has_zero_admission_score(self, verifier):
        metric = verifier_metric(verifier)
        task = task_input_for(ATTENTION, KIND_ID)
        gold = dspy.Example(task_input=task).with_inputs("task_input")

        score = metric(gold, dspy.Prediction(output="That would be arXiv 1706.03799."))

        assert verifier.score(task, "That would be arXiv 1706.03799.").composite > 0
        assert score == pytest.approx(0.0)

    def test_hedging_across_ids_gets_half_credit(self, verifier):
        task = task_input_for(ATTENTION, KIND_ID)
        fitness = verifier.score(task, "It is either 1706.03762 or 1810.04805.")
        assert fitness.correctness == pytest.approx(0.5)

    def test_no_id_scores_lowest(self, verifier):
        task = task_input_for(ATTENTION, KIND_ID)
        fitness = verifier.score(task, "A very famous transformer paper.")
        assert fitness.correctness == pytest.approx(0.0)
        assert fitness.procedure_following == pytest.approx(0.0)

    def test_wrong_beats_nothing_only_slightly(self):
        wrong = grade_arxiv_id("1706.03762", "It is 9999.99999.")
        nothing = grade_arxiv_id("1706.03762", "No idea, sorry.")
        assert wrong[0] == nothing[0] == 0.0
        assert wrong[1] > nothing[1]


class TestTitleGrading:
    def test_exact_title_scores_full(self, verifier):
        task = task_input_for(ATTENTION, KIND_TITLE)
        fitness = verifier.score(task, 'The title is "Attention Is All You Need".')
        assert fitness.correctness == pytest.approx(1.0)

    def test_fused_spelling_variant_still_scores_full(self):
        correctness, _, _ = grade_title(
            "LoRA: Low-Rank Adaptation of Large Language Models",
            "The title is LoRA: LowRank Adaptation of Large Language Models.",
        )
        assert correctness == pytest.approx(1.0)

    def test_partial_title_gets_partial_credit(self):
        correctness, well_formed, _ = grade_title(
            "Attention Is All You Need", "It is about attention, and you need it."
        )
        assert correctness == pytest.approx(0.75)
        assert well_formed == pytest.approx(1.0)

    def test_unrelated_output_scores_zero(self):
        correctness, well_formed, feedback = grade_title(
            "Attention Is All You Need", "No idea."
        )
        assert correctness == 0.0
        assert well_formed == 0.0
        assert "Attention Is All You Need" in feedback


class TestAuthorGrading:
    def test_last_name_is_enough(self, verifier):
        task = task_input_for(ATTENTION, KIND_AUTHOR)
        fitness = verifier.score(task, "The first author is Ashish Vaswani.")
        assert fitness.correctness == pytest.approx(1.0)

    def test_missing_author_scores_zero(self, verifier):
        task = task_input_for(ATTENTION, KIND_AUTHOR)
        fitness = verifier.score(task, "Written by researchers at Google Brain.")
        assert fitness.correctness == pytest.approx(0.0)
        assert "Vaswani" in fitness.feedback


class TestYearGrading:
    def test_correct_year_scores_full(self, verifier):
        task = task_input_for(ATTENTION, KIND_YEAR)
        fitness = verifier.score(task, "It was first submitted in 2017.")
        assert fitness.correctness == pytest.approx(1.0)

    def test_wrong_year_scores_zero(self, verifier):
        task = task_input_for(ATTENTION, KIND_YEAR)
        fitness = verifier.score(task, "That was 2018.")
        assert fitness.correctness == pytest.approx(0.0)

    def test_id_prefix_is_not_misread_as_year(self):
        # 2005.14165 must not have its YYMM prefix counted as the year 2005.
        correctness, _, _ = grade_year("2020", "arXiv 2005.14165 came out in 2020.")
        assert correctness == pytest.approx(1.0)

    def test_hedging_across_years_gets_half_credit(self):
        correctness, _, _ = grade_year("2017", "Around 2016 or 2017.")
        assert correctness == pytest.approx(0.5)


class TestScoreEdgeCases:
    def test_unknown_task_scores_zero_with_explanation(self, verifier):
        fitness = verifier.score("What is the meaning of life?", "42")
        assert fitness.composite == pytest.approx(0.0)
        assert "ground truth" in fitness.feedback.lower()

    def test_empty_output_scores_zero(self, verifier):
        task = task_input_for(ATTENTION, KIND_ID)
        assert verifier.score(task, "   ").composite == pytest.approx(0.0)

    def test_verbosity_lowers_the_score(self, verifier):
        task = task_input_for(ATTENTION, KIND_ID)
        short = verifier.score(task, "arXiv 1706.03762.")
        long = verifier.score(task, "arXiv 1706.03762. " + "Additional context. " * 250)
        assert short.composite > long.composite
        assert long.correctness == pytest.approx(1.0)

    def test_brief_evasion_scores_zero(self, verifier):
        # Being short must never outscore actually answering.
        task = task_input_for(ATTENTION, KIND_ID)
        evasive = verifier.score(task, "Great question!")
        verbose_correct = verifier.score(task, "arXiv 1706.03762. " + "Context. " * 400)
        assert evasive.composite == pytest.approx(0.0)
        assert verbose_correct.composite > evasive.composite

    def test_conciseness_curve(self):
        assert _conciseness("x" * 800) == pytest.approx(1.0)
        assert _conciseness("x" * 2400) == pytest.approx(0.5)
        assert _conciseness("x" * 4000) == pytest.approx(0.0)


class TestMetricIntegration:
    def test_metric_scores_dataset_examples(self, verifier):
        metric = verifier_metric(verifier)
        example = verifier.build_dataset().to_dspy_examples("train")[0]
        score = metric(example, dspy.Prediction(output="some answer with 1706.03762"))
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_gepa_form_carries_verifier_feedback(self, verifier):
        metric = verifier_metric(verifier)
        gold = dspy.Example(
            task_input=task_input_for(ATTENTION, KIND_ID)
        ).with_inputs("task_input")
        result = metric(gold, dspy.Prediction(output="No clue."), None, "predictor", None)
        assert result.score == pytest.approx(0.0)
        assert "1706.03762" in result.feedback
