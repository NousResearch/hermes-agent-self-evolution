"""Tests for the semantic-preservation constraint (PLAN.md guardrail #4).

This gate exists because of a real failure: GEPA drifted a gate-evidence-ledger
skill into an app-reverse-engineering prompt. Term retention must catch that,
while passing legitimate same-domain rewrites (a tightened wording, an added
example) that keep the skill's vocabulary.
"""

import pytest

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator


GATE_BASELINE = """# Gate & Evidence Ledger

Turn "it works" into a traceable claim. Every gate you pass gets a ledger row;
every claim you make maps to a row. This mirrors the discipline of a numbered
gate pipeline: requirement to evidence to verdict, never "tests passed."

## The ledger row

One row per requirement. Never merge two requirements into one row.
Each row records the requirement identifier, the evidence that satisfies it,
and the verdict of pass or fail with the command output that produced it.
"""

# The actual failure mode: same skill rewritten into a different domain.
DRIFTED = """# App Reverse Engineering Master Prompt

You are an AI assistant whose job is to produce a rigorous master prompt for
reverse-engineering an existing app and rebuilding a private iOS clone. The
prompt must cover Android APK extraction, bytecode analysis, iOS private
distribution, xcodebuild signing, and TestFlight upload, with no mistakes.
"""

# A legitimate same-domain edit: keeps domain vocabulary, tightens wording.
SAME_DOMAIN = """# Gate & Evidence Ledger

Turn "it works" into a traceable claim. Every gate you pass gets a ledger row;
every claim you make maps to a row with a requirement identifier and evidence.
A verdict is pass or fail only when command output backs it — never "tests
passed" without the output. One row per requirement, never merged.
"""


@pytest.fixture
def validator():
    return ConstraintValidator(EvolutionConfig())


def _result(validator, text, baseline=GATE_BASELINE):
    return [
        r for r in validator.validate_all(text, "skill", baseline_text=baseline)
        if r.constraint_name == "semantic_preservation"
    ][0]


class TestSemanticPreservation:
    def test_same_domain_edit_passes(self, validator):
        r = _result(validator, SAME_DOMAIN)
        assert r.passed, r.message

    def test_drift_is_rejected(self, validator):
        r = _result(validator, DRIFTED)
        assert not r.passed, r.message
        assert "Drift" in r.message

    def test_drift_reports_dropped_terms(self, validator):
        r = _result(validator, DRIFTED)
        # The domain words that vanished should be named in the message.
        assert any(w in r.message for w in ("ledger", "verdict", "evidence", "requirement"))

    def test_identical_text_passes(self, validator):
        r = _result(validator, GATE_BASELINE)
        assert r.passed, r.message

    def test_ratios_are_quantified(self, validator):
        r = _result(validator, SAME_DOMAIN)
        assert "%" in r.message  # reports retention percentage

    def test_empty_baseline_skips_gate(self, validator):
        """Empty baseline string means no anchor: the gate is skipped, not failed.

        validate_all only runs semantic preservation when baseline_text is
        truthy, so an empty baseline yields no semantic result at all — not a
        spurious rejection.
        """
        results = validator.validate_all("whatever", "skill", baseline_text="")
        names = {r.constraint_name for r in results}
        assert "semantic_preservation" not in names

    def test_baseline_without_content_terms_passes(self, validator):
        """A non-empty baseline whose words are all <=3 chars has no anchor."""
        # "gate" is 4 chars; use only sub-4-char words so content_terms is empty.
        r = _result(validator, "abc def", baseline="the a cat")
        assert r.passed, r.message

    def test_short_baseline_is_not_an_automatic_fail(self, validator):
        # A baseline of one 4-letter word: evolved text that keeps it passes.
        r = _result(validator, "keep gate here", baseline="gate")
        assert r.passed

    def test_disabled_flag_skips_the_gate(self):
        cfg = EvolutionConfig(semantic_preservation_enabled=False)
        v = ConstraintValidator(cfg)
        results = v.validate_all(DRIFTED, "skill", baseline_text=GATE_BASELINE)
        names = {r.constraint_name for r in results}
        assert "semantic_preservation" not in names


class TestThreshold:
    def test_threshold_is_configurable(self):
        # Raise the bar to 100%: even a legit edit now fails, proving the
        # threshold is actually read rather than hardcoded.
        cfg = EvolutionConfig(min_semantic_similarity=1.0)
        v = ConstraintValidator(cfg)
        r = _result(v, SAME_DOMAIN)
        assert not r.passed

    def test_default_threshold_accepts_minor_edit(self, validator):
        # A small wording change keeps enough terms to clear the 0.5 default.
        r = _result(validator, GATE_BASELINE + "\n\nRevised: keep every ledger row.")
        assert r.passed
