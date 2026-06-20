"""Tests for eval dataset parsing helpers."""

import pytest

from evolution.core.dataset_builder import (
    _case_to_eval_example,
    _parse_generated_test_cases,
)


def test_parse_generated_cases_accepts_python_literal_alias_fields():
    raw = """
    [{'id': 1,
      'scenario': 'Successful 3-step workflow execution',
      'input': 'Draft a market analysis with research, draft, and review phases.',
      'expected_outcome': 'The plan must split work into research, drafting, and review steps.',
      'difficulty': 'hard',
      'category': 'workflow'}]
    """

    cases = _parse_generated_test_cases(raw)
    example = _case_to_eval_example(cases[0])

    assert example is not None
    assert example.task_input == 'Draft a market analysis with research, draft, and review phases.'
    assert example.expected_behavior == 'The plan must split work into research, drafting, and review steps.'
    assert example.difficulty == 'hard'
    assert example.category == 'workflow'


def test_parse_generated_cases_accepts_json_with_trailing_commas():
    raw = '[{"task_input": "Do the thing", "expected_behavior": "Do it safely",},]'

    cases = _parse_generated_test_cases(raw)
    example = _case_to_eval_example(cases[0])

    assert example is not None
    assert example.task_input == 'Do the thing'
    assert example.expected_behavior == 'Do it safely'


def test_parse_generated_cases_raises_when_no_array():
    with pytest.raises(ValueError, match='Could not find JSON array'):
        _parse_generated_test_cases('no list here')
