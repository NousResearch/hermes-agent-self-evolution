"""Tests for direct skill-body optimization."""

from evolution.core.dataset_builder import EvalDataset, EvalExample
from evolution.skills.body_optimizer import build_eval_brief, optimize_skill_body


def _dataset():
    return EvalDataset(
        train=[
            EvalExample(
                task_input="Write a cautious clinic blog draft",
                expected_behavior="Use Korean, avoid guaranteed outcomes, include safety check",
                category="blog",
            )
        ],
        val=[
            EvalExample(
                task_input="Review exaggerated medical wording",
                expected_behavior="Identify overclaims and suggest softer alternatives",
                category="safety",
            )
        ],
    )


def test_build_eval_brief_includes_task_and_rubric():
    brief = build_eval_brief(_dataset())

    assert "Write a cautious clinic blog draft" in brief
    assert "avoid guaranteed outcomes" in brief
    assert "Review exaggerated medical wording" in brief
    assert "softer alternatives" in brief


def test_optimize_skill_body_returns_generator_candidate():
    baseline = "# Blog Writer\n\n## Procedure\nDo the old thing."

    class FakeResult:
        evolved_body = "# Blog Writer\n\n## Procedure\nDo the improved cautious thing.\n\n## Safety\nAvoid guarantees."

    class FakeGenerator:
        calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return FakeResult()

    fake = FakeGenerator()
    evolved, metadata = optimize_skill_body(
        baseline_body=baseline,
        dataset=_dataset(),
        optimizer_model="fake/model",
        iterations=1,
        generator=fake,
    )

    assert evolved != baseline
    assert "improved cautious" in evolved
    assert fake.calls[0]["current_body"] == baseline
    assert "avoid guaranteed outcomes" in fake.calls[0]["eval_brief"]
    assert metadata["changed"] is True
    assert metadata["iterations_run"] == 1


def test_optimize_skill_body_keeps_baseline_when_candidate_empty():
    baseline = "# Blog Writer\n\n## Procedure\nDo the old thing."

    class FakeResult:
        evolved_body = "   "

    class FakeGenerator:
        def __call__(self, **kwargs):
            return FakeResult()

    evolved, metadata = optimize_skill_body(
        baseline_body=baseline,
        dataset=_dataset(),
        optimizer_model="fake/model",
        iterations=1,
        generator=FakeGenerator(),
    )

    assert evolved == baseline
    assert metadata["changed"] is False
    assert metadata["rejected_empty"] == 1
