from types import SimpleNamespace

from evolution.skills import evolve_skill
from evolution.skills.evolve_skill import score_holdout_examples


class FakeModule:
    def __init__(self, output: str):
        self.output = output
        self.calls = 0

    def __call__(self, *, task_input: str):
        self.calls += 1
        return SimpleNamespace(output=f"{self.output}:{task_input}")


class ExplodingModule:
    calls = 0

    def __call__(self, *, task_input: str):
        self.calls += 1
        raise AssertionError("unchanged evolved skill should not be re-evaluated")


def test_score_holdout_reuses_baseline_scores_when_skill_text_is_unchanged(monkeypatch):
    def fake_metric(example, prediction, trace=None):
        return 0.7 if prediction.output.startswith("baseline") else 0.1

    monkeypatch.setattr(evolve_skill, "skill_fitness_metric", fake_metric)
    baseline = FakeModule("baseline")
    evolved = ExplodingModule()
    examples = [SimpleNamespace(task_input="review this", expected_behavior="flag risk")]

    baseline_scores, evolved_scores = score_holdout_examples(
        examples,
        baseline,
        evolved,
        lm=None,
        evolved_text_changed=False,
    )

    assert baseline_scores == [0.7]
    assert evolved_scores == [0.7]
    assert baseline.calls == 1
    assert evolved.calls == 0


def test_score_holdout_evaluates_evolved_module_when_skill_text_changed(monkeypatch):
    def fake_metric(example, prediction, trace=None):
        return 0.8 if prediction.output.startswith("evolved") else 0.4

    monkeypatch.setattr(evolve_skill, "skill_fitness_metric", fake_metric)
    baseline = FakeModule("baseline")
    evolved = FakeModule("evolved")
    examples = [SimpleNamespace(task_input="review this", expected_behavior="flag risk")]

    baseline_scores, evolved_scores = score_holdout_examples(
        examples,
        baseline,
        evolved,
        lm=None,
        evolved_text_changed=True,
    )

    assert baseline_scores == [0.4]
    assert evolved_scores == [0.8]
    assert baseline.calls == 1
    assert evolved.calls == 1
