"""The three admission counterexamples from the review of #150, kept together.

Each test here is a reproduction, not a paraphrase: the paper, the wrong
answer, the dataset parameters and the win/loss pattern are the ones the audit
reported at commit 448488e. They are collected in one module because they
describe one failure mode with three faces — the pipeline calling something
admissible when the evidence does not support it — and because a reviewer
checking whether the report was addressed should not have to hunt through
three packages to find out.

Each one fails on the pre-fix code and passes after its fix:

  1. a confidently wrong answer scored 0.500 on a metric it should score 0 on,
     because the 0.3 procedure and 0.2 conciseness weights exactly cover a
     correctness of zero;
  2. papers appeared in more than one split at the default num_cases=24,
     seed=13, so a holdout answer could be recited from training;
  3. a mean that moved up on six examples counted as an improvement, when six
     examples cannot make anything short of a near-unanimous flip significant.
"""

import dspy
import pytest

from evolution.core.admission import evaluate_admission
from evolution.core.verifier import verifier_metric
from evolution.skills import evolve_skill
from evolution.verifiers.arxiv_verifier import (
    KIND_ID,
    PAPER_BY_ID,
    ArxivVerifier,
    task_input_for,
)

ATTENTION = PAPER_BY_ID["1706.03762"]

# The audit's own wrong answer: right shape, right prefix, wrong paper.
CONFIDENTLY_WRONG = "The paper is arXiv 1706.03799."


@pytest.fixture
def verifier():
    return ArxivVerifier()


# ── 1. Incorrect facts must not reach an admission-grade score ────────────


def test_confidently_wrong_answer_scores_zero_for_admission(verifier):
    """correctness == 0 must force the admission score to 0.

    Pre-fix this returned 0.500: procedure_following and conciseness were both
    1.0 for a well-formed wrong answer, and 0.3 + 0.2 is exactly the credit a
    zero correctness gives up. Half marks for stating the wrong arXiv ID
    fluently is precisely the signal an objective verifier exists to remove.
    """
    task = task_input_for(ATTENTION, KIND_ID)
    fitness = verifier.score(task, CONFIDENTLY_WRONG)

    assert fitness.correctness == pytest.approx(0.0)

    # The feedback dimensions survive for GEPA's reflection, and still sum to
    # the 0.500 that used to be handed to the optimizer as a score.
    assert fitness.composite == pytest.approx(0.5)

    metric = verifier_metric(verifier)
    gold = dspy.Example(task_input=task).with_inputs("task_input")
    score = metric(gold, dspy.Prediction(output=CONFIDENTLY_WRONG))

    assert score == pytest.approx(0.0)


def test_wrong_answer_scores_zero_in_gepa_feedback_form_too(verifier):
    """The floor must hold on the feedback-aware call GEPA actually makes.

    ``verifier_metric`` has two return paths. Flooring only the plain float
    would leave the optimizer reading a 0.500 for a wrong answer through the
    path it uses for reflection, which is the one that steers evolution.
    """
    task = task_input_for(ATTENTION, KIND_ID)
    metric = verifier_metric(verifier)
    gold = dspy.Example(task_input=task).with_inputs("task_input")

    prediction = metric(
        gold,
        dspy.Prediction(output=CONFIDENTLY_WRONG),
        None,
        "predictor",
        None,
    )

    assert prediction.score == pytest.approx(0.0)
    # Reflection still gets told what was wrong, which is the point of keeping
    # the dimensions rather than zeroing the FitnessScore itself.
    assert "1706.03762" in prediction.feedback


def test_partial_credit_is_not_floored(verifier):
    """The floor is on zero correctness, not on partial correctness.

    Hedging across two IDs earns 0.5 correctness by design. Flooring that too
    would collapse a graded signal into pass/fail and cost GEPA the gradient
    it needs, so the boundary is drawn at correctness == 0 exactly.
    """
    task = task_input_for(ATTENTION, KIND_ID)
    hedged = "It is either 1706.03762 or 1810.04805."

    assert verifier.score(task, hedged).correctness == pytest.approx(0.5)

    metric = verifier_metric(verifier)
    gold = dspy.Example(task_input=task).with_inputs("task_input")

    assert metric(gold, dspy.Prediction(output=hedged)) > 0


# ── 2. Papers must not leak across splits ─────────────────────────────────


def _paper_of(task_input):
    """Which seed paper a task is about, read from the task text itself.

    Deliberately not ``ground_truth_for(...).paper_id``: the grouping the fix
    introduced is the thing under test, so the test derives the answer
    independently rather than asking the fix to confirm its own bookkeeping.
    """
    for paper_id, paper in PAPER_BY_ID.items():
        if paper_id in task_input or paper.title in task_input:
            return paper_id
    raise AssertionError(f"no seed paper found in task: {task_input!r}")


def _papers_by_split(verifier, dataset):
    """Map each split to the set of papers its tasks came from."""
    return {
        name: {_paper_of(example.task_input) for example in getattr(dataset, name)}
        for name in ("train", "val", "holdout")
    }


def test_default_dataset_has_no_paper_leakage(verifier):
    """The audit's exact reproduction: num_cases=24, seed=13.

    Pre-fix this shuffled 68 flat tasks and cut by index, so the same paper
    supplied an ID lookup to train and a title lookup to holdout. The reported
    overlaps were train n val = {1810.04805, 2201.11903}, train n holdout =
    {1810.04805, 2006.11239, 2106.09685} and val n holdout = {1810.04805}.
    """
    dataset = verifier.build_dataset(num_cases=24, seed=13)
    groups = _papers_by_split(verifier, dataset)

    assert groups["train"] & groups["val"] == set()
    assert groups["train"] & groups["holdout"] == set()
    assert groups["val"] & groups["holdout"] == set()

    # The split shape the pipeline depends on is unchanged by the grouping.
    assert (len(dataset.train), len(dataset.val), len(dataset.holdout)) == (12, 6, 6)


@pytest.mark.parametrize("seed", [1, 13, 7, 99, 2026])
@pytest.mark.parametrize("num_cases", [8, 12, 24, 40, 68])
def test_no_paper_leakage_at_any_size_or_seed(verifier, seed, num_cases):
    """Disjointness is structural, so it must hold everywhere, not at seed 13.

    A fix that only cleared the reported parameters would leave the defect
    live for every other run.
    """
    dataset = verifier.build_dataset(num_cases=num_cases, seed=seed)
    groups = _papers_by_split(verifier, dataset)

    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["holdout"])
    assert groups["val"].isdisjoint(groups["holdout"])

    total = len(dataset.train) + len(dataset.val) + len(dataset.holdout)
    assert total == min(num_cases, len(verifier._tasks))


# ── 3. A small mean gain on six examples is not an improvement ────────────


def test_single_win_on_six_examples_is_not_admitted():
    """The audit's power argument, as a decision.

    Six holdout examples, the candidate wins one and ties five. The mean rises
    by 0.167 and the artifact changed, so the old rule called this a success
    and reported an improvement. McNemar's exact test on the single discordant
    pair gives p = 0.5: this is one coin flip landing heads.
    """
    baseline_correct = [True, True, True, False, False, False]
    candidate_correct = [True, True, True, False, False, True]
    baseline_scores = [float(x) for x in baseline_correct]
    candidate_scores = [float(x) for x in candidate_correct]

    improvement = (
        sum(candidate_scores) / len(candidate_scores)
        - sum(baseline_scores) / len(baseline_scores)
    )

    # The defect, stated exactly as the code stated it. `if improvement > 0`
    # was the whole admission rule, and it is satisfied here.
    assert improvement > 0

    verdict = evolve_skill._admission_verdict(
        baseline_scores,
        candidate_scores,
        baseline_correct=baseline_correct,
        candidate_correct=candidate_correct,
    )

    assert not verdict.admitted
    assert not verdict.significant_improvement
    assert verdict.p_improvement == pytest.approx(0.5)


def test_six_examples_are_declared_underpowered():
    """Five aligned losses are needed to reach 0.05; four give p = 0.0625.

    So the smallest shift six examples can resolve is 5/6, and a
    'no significant regression' result here means 'no regression above 83.3%'.
    The verdict has to say that rather than let it read as a clean bill.
    """
    verdict = evaluate_admission([1.0] * 6, [1.0] * 6, tolerance=0.05)

    assert verdict.n == 6
    assert verdict.min_detectable_shift == pytest.approx(5 / 6)
    assert verdict.underpowered
    assert "cannot resolve anything smaller" in verdict.power_note


def test_a_clean_sweep_on_six_examples_is_admitted():
    """The gate must still be able to say yes, or it is not a gate.

    Five discordant pairs all favouring the candidate give p = 0.03125, which
    clears alpha. This is the smallest result on six examples that can.
    """
    baseline_correct = [False, False, False, False, False, True]
    candidate_correct = [True, True, True, True, True, True]

    verdict = evaluate_admission(
        [float(x) for x in baseline_correct],
        [float(x) for x in candidate_correct],
        baseline_correct=baseline_correct,
        candidate_correct=candidate_correct,
    )

    assert verdict.p_improvement == pytest.approx(0.03125)
    assert verdict.significant_improvement
    assert verdict.admitted


def test_a_significant_regression_is_refused():
    """Non-regression is enforced, not merely reported."""
    baseline_correct = [True, True, True, True, True, False]
    candidate_correct = [False, False, False, False, False, False]

    verdict = evaluate_admission(
        [float(x) for x in baseline_correct],
        [float(x) for x in candidate_correct],
        baseline_correct=baseline_correct,
        candidate_correct=candidate_correct,
    )

    assert verdict.significant_regression
    assert not verdict.admitted
    assert "regressed" in verdict.reason


def test_an_unchanged_artifact_is_refused_when_the_caller_checks():
    """An unchanged artifact is refused however good the numbers look."""
    baseline_correct = [False] * 5 + [True]
    candidate_correct = [True] * 6

    verdict = evaluate_admission(
        [float(x) for x in baseline_correct],
        [float(x) for x in candidate_correct],
        material_diff=False,
        baseline_correct=baseline_correct,
        candidate_correct=candidate_correct,
    )

    assert not verdict.admitted
    assert "did not change" in verdict.reason


def test_an_unchecked_artifact_is_recorded_as_unknown_not_as_passing():
    """Not asking the question must not be recorded as having passed it.

    This command does not compare the deployable artifacts, so the verdict
    carries None. A default of True would have written an unverified claim
    into metrics.json, which is the kind of quiet overstatement this whole
    change exists to remove.
    """
    baseline_correct = [False] * 5 + [True]
    candidate_correct = [True] * 6

    verdict = evaluate_admission(
        [float(x) for x in baseline_correct],
        [float(x) for x in candidate_correct],
        baseline_correct=baseline_correct,
        candidate_correct=candidate_correct,
    )

    assert verdict.material_diff is None
    assert verdict.to_dict()["material_diff"] is None
    # Unknown does not block a result that carries evidence.
    assert verdict.admitted
