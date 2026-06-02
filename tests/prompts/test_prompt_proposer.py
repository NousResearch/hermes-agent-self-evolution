"""Tests for PromptSectionProposer — sentinel-preserving GEPA proposal fn."""
from __future__ import annotations

from unittest.mock import MagicMock

import dspy

from evolution.prompts.prompt_module import (
    _close_sentinel,
    _open_sentinel,
    _render_instructions,
)
from evolution.prompts.prompt_proposer import (
    PromptSectionProposer,
    extract_and_rebuild,
)


SECTION = "MEMORY_GUIDANCE"


def _wrapped(body: str) -> str:
    return _render_instructions(SECTION, body)


def test_extract_and_rebuild_round_trips_sentinels():
    candidate = _wrapped("a refined body")
    rebuilt = extract_and_rebuild(candidate, SECTION)
    # The rebuilt instructions still carry intact sentinels around the new body.
    assert _open_sentinel(SECTION) in rebuilt
    assert _close_sentinel(SECTION) in rebuilt
    assert "a refined body" in rebuilt


def test_proposer_only_acts_on_its_component():
    proposer = PromptSectionProposer(
        section_name=SECTION, baseline_chars=100,
    )
    # A request that doesn't include our component returns empty.
    out = proposer(
        candidate={"passthrough.predict": _wrapped("x")},
        reflective_dataset={},
        components_to_update=["something.else"],
    )
    assert out == {}


def test_proposer_rebuilds_sentinel_region(monkeypatch):
    proposer = PromptSectionProposer(section_name=SECTION, baseline_chars=100)

    # Stub the LM proposal: return a full-instructions string with the
    # sentinel region edited.
    fake_pred = MagicMock()
    fake_pred.improved_instruction = _wrapped("LM-revised memory guidance")
    proposer.propose = MagicMock(return_value=fake_pred)

    out = proposer(
        candidate={"passthrough.predict": _wrapped("original")},
        reflective_dataset={"passthrough.predict": [{"Feedback": "be clearer"}]},
        components_to_update=["passthrough.predict"],
    )
    assert "passthrough.predict" in out
    assert "LM-revised memory guidance" in out["passthrough.predict"]
    assert _open_sentinel(SECTION) in out["passthrough.predict"]


def test_proposer_raises_on_sentinel_loss():
    from evolution.prompts.prompt_module import SentinelParseError

    proposer = PromptSectionProposer(section_name=SECTION, baseline_chars=100)
    fake_pred = MagicMock()
    fake_pred.improved_instruction = "the model dropped the sentinels entirely"
    proposer.propose = MagicMock(return_value=fake_pred)

    import pytest
    with pytest.raises(SentinelParseError):
        proposer(
            candidate={"passthrough.predict": _wrapped("original")},
            reflective_dataset={"passthrough.predict": [{"Feedback": "x"}]},
            components_to_update=["passthrough.predict"],
        )
    assert proposer.sentinel_failures == 1
