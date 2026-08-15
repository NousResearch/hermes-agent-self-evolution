"""Tests for the GEPA-compatible fitness metric."""

import dspy

from evolution.core.fitness import skill_fitness_metric


def _example_and_pred():
    example = dspy.Example(
        task_input="task",
        expected_behavior="verify evidence before concluding done",
    )
    prediction = dspy.Prediction(output="I will verify the evidence first")
    return example, prediction


class TestMetricContract:
    def test_direct_call_returns_float(self):
        example, prediction = _example_and_pred()
        score = skill_fitness_metric(example, prediction)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_miprov2_style_call_returns_float(self):
        example, prediction = _example_and_pred()
        score = skill_fitness_metric(example, prediction, None)
        assert isinstance(score, float)

    def test_gepa_reflection_call_returns_feedback(self):
        # GEPA's GEPAFeedbackMetric protocol: (gold, pred, trace, pred_name,
        # pred_trace); predictor-level calls expect Prediction(score, feedback).
        example, prediction = _example_and_pred()
        result = skill_fitness_metric(example, prediction, None, "predictor", None)
        assert isinstance(result, dspy.Prediction)
        assert 0.0 <= result.score <= 1.0
        assert result.feedback

    def test_empty_output_scores_zero(self):
        example, _ = _example_and_pred()
        assert skill_fitness_metric(example, dspy.Prediction(output="")) == 0.0

    def test_gepa_accepts_metric_with_valid_budget(self):
        # Regression: GEPA has no `max_steps` — the old call always raised
        # TypeError and silently fell back to MIPROv2.
        lm = dspy.LM("openai/test", api_base="http://127.0.0.1:1/v1", api_key="x")
        optimizer = dspy.GEPA(
            metric=skill_fitness_metric, max_full_evals=5, reflection_lm=lm,
        )
        assert optimizer is not None
