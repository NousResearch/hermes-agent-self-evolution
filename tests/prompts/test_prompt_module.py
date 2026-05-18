"""Tests for the prompt-section DSPy module."""

import dspy

from evolution.prompts.prompt_module import (
    PromptSectionModule,
    _parse_judge_score,
    behavioral_fitness_metric,
    clean_evolved_section,
    make_llm_judge_metric,
)


class TestPromptSectionModule:
    def test_text_lives_in_predictor_instructions(self):
        text = "Save durable facts to persistent memory."
        module = PromptSectionModule(section_name="MEMORY_GUIDANCE", text=text)

        assert module.section_text == text
        assert module._inner_predict().signature.instructions == text

    def test_setter_updates_predictor_instructions(self):
        module = PromptSectionModule(section_name="MEMORY_GUIDANCE", text="old text")
        module.section_text = "new text"

        assert module.section_text == "new text"
        assert module._inner_predict().signature.instructions == "new text"


class TestBehavioralFitnessMetric:
    def test_high_keyword_overlap_scores_above_baseline(self):
        example = dspy.Example(
            task_input="When should the agent save memory?",
            expected_behavior="save durable user preferences",
        )
        prediction = dspy.Prediction(output="The agent should save durable user preferences.")
        assert behavioral_fitness_metric(example, prediction) >= 0.7

    def test_no_overlap_scores_low(self):
        example = dspy.Example(
            task_input="x",
            expected_behavior="durable persistent preferences",
        )
        prediction = dspy.Prediction(output="Completely unrelated text.")
        assert behavioral_fitness_metric(example, prediction) <= 0.4

    def test_empty_output_scores_zero(self):
        example = dspy.Example(task_input="x", expected_behavior="anything")
        prediction = dspy.Prediction(output="")
        assert behavioral_fitness_metric(example, prediction) == 0.0


class TestParseJudgeScore:
    def test_extracts_simple_float(self):
        assert _parse_judge_score("0.8") == 0.8

    def test_extracts_score_from_prose(self):
        assert _parse_judge_score("I would rate this 0.45 because ...") == 0.45

    def test_clamps_out_of_range(self):
        assert _parse_judge_score("1") == 1.0
        assert _parse_judge_score("0") == 0.0

    def test_empty_returns_zero(self):
        assert _parse_judge_score("") == 0.0
        assert _parse_judge_score("no number here") == 0.0


class _FakeJudgeLM:
    def __init__(self, score: str):
        self.score = score


class TestLLMJudgeMetric:
    def test_falls_back_to_overlap_on_judge_error(self, monkeypatch):
        metric = make_llm_judge_metric(judge_lm=_FakeJudgeLM("0.9"), fallback_weight=0.0)
        example = dspy.Example(task_input="x", expected_behavior="durable user preferences")
        prediction = dspy.Prediction(output="save durable user preferences")
        score = metric(example, prediction)
        assert 0.0 <= score <= 1.0

    def test_empty_output_returns_zero(self):
        metric = make_llm_judge_metric(judge_lm=_FakeJudgeLM("0.9"))
        example = dspy.Example(task_input="x", expected_behavior="anything")
        prediction = dspy.Prediction(output="")
        assert metric(example, prediction) == 0.0


class TestCleanEvolvedSection:
    def test_truncates_at_inline_example_marker(self):
        text = "Save durable facts.\n\nExample:\nTask Input: x\nDecision: yes"
        cleaned = clean_evolved_section(text)
        assert cleaned == "Save durable facts."

    def test_budget_drops_trailing_sentences(self):
        text = "Save durable facts. Skip transient task state. Do not save secrets."
        cleaned = clean_evolved_section(text, max_chars=25)
        assert len(cleaned) <= 25
        assert cleaned.startswith("Save")
