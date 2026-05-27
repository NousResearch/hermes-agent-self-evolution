"""Tests for skill evolution budget selection."""

import dspy

from evolution.skills.evolve_skill import _gepa_budget_kwargs, _score_holdout_example


def test_gepa_budget_uses_explicit_full_evals_for_ollama_and_codex_models():
    assert _gepa_budget_kwargs(
        5,
        'ollama_chat/gemma4-e4b:latest',
        'ollama_chat/gemma4-e4b:latest',
    ) == {'max_full_evals': 5}
    assert _gepa_budget_kwargs(
        2,
        'openai-codex/gpt-5.4-mini',
        'openai-codex/gpt-5.4-mini',
    ) == {'max_full_evals': 2}


def test_gepa_budget_keeps_auto_presets_for_other_hosted_models():
    assert _gepa_budget_kwargs(
        5,
        'openai/gpt-4.1-mini',
        'openai/gpt-4.1-mini',
    ) == {'auto': 'light'}
    assert _gepa_budget_kwargs(
        10,
        'openai/gpt-4.1-mini',
        'openai/gpt-4.1-mini',
    ) == {'auto': 'medium'}


def test_holdout_scoring_converts_program_failures_to_zero():
    class BrokenProgram:
        def __call__(self, **kwargs):
            raise RuntimeError('format failed')

    example = dspy.Example(
        task_input='Do the task',
        expected_behavior='Do it safely',
    ).with_inputs('task_input')

    assert _score_holdout_example(BrokenProgram(), example, lm=None) == 0.0
