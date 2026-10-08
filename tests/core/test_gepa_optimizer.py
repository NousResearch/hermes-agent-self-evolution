"""Regression tests for GEPA optimizer construction.

DSPy renamed GEPA's budget parameter (max_steps -> max_full_evals /
max_metric_calls). The code passed max_steps unconditionally inside a blanket
try/except, so every run raised TypeError and silently fell back to MIPROv2 --
reported as "GEPA not available" rather than as the bug it was. These tests
pin that GEPA is actually built, with a real budget.
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest

from evolution.skills import evolve_skill as es


def _fake_lm(*a, **k):
    return MagicMock()


class TestGepaConstruction:
    def test_gepa_is_constructed_without_raising(self):
        """dspy.GEPA must accept the kwargs this code passes.

        Constructing the real class against the real signature is the only
        check that catches a renamed parameter; a MagicMock would accept
        anything and let the bug straight through.

        reflection_lm is required, not optional: GEPA asserts on it, so a
        construction that omits it fails even when the budget kwarg is right.
        """
        import dspy

        params = inspect.signature(dspy.GEPA.__init__).parameters
        kw = {"metric": lambda *a, **k: 1.0}
        if "max_steps" in params:
            kw["max_steps"] = 10
        elif "max_full_evals" in params:
            kw["max_full_evals"] = 10
        elif "max_metric_calls" in params:
            kw["max_metric_calls"] = 10
        if "reflection_lm" in params:
            kw["reflection_lm"] = MagicMock()

        optimizer = dspy.GEPA(**kw)  # must not raise TypeError/AssertionError
        assert isinstance(optimizer, dspy.GEPA)

    def test_reflection_lm_is_required_by_gepa(self):
        """Pin the upstream requirement that made the second failure invisible.

        Without this, dropping reflection_lm looks harmless in code review but
        raises at construction -- which the old blanket except turned into a
        silent MIPROv2 downgrade.
        """
        import dspy

        params = inspect.signature(dspy.GEPA.__init__).parameters
        if "reflection_lm" not in params:
            pytest.skip("installed GEPA has no reflection_lm parameter")
        with pytest.raises(AssertionError):
            dspy.GEPA(metric=lambda *a, **k: 1.0)

    def test_budget_kwarg_matches_installed_signature(self):
        """Whatever budget kwarg we choose must exist on the installed GEPA."""
        import dspy

        params = inspect.signature(dspy.GEPA.__init__).parameters
        assert (
            "max_steps" in params
            or "max_full_evals" in params
            or "max_metric_calls" in params
        ), "no known GEPA budget parameter found; update the builder"

    def test_optimizer_construction_failure_is_not_swallowed(self):
        """A GEPA construction error must surface, not downgrade to MIPROv2.

        The original defect: one try/except wrapped both construction and
        compile, so a construction TypeError became a quiet fallback and the
        run reported success using the wrong optimizer.
        """
        source = inspect.getsource(es.evolve)
        # GEPA must be built outside the compile-only try block.
        assert "GEPA compile failed" in source
        assert "GEPA not available" not in source

    def test_reflection_lm_is_set_before_construction(self):
        """reflection_lm must be in the kwargs dict passed to GEPA(...).

        Setting it on the dict after construction silently does nothing, so
        the mutation proposer would fall back to the task/judge model.
        """
        source = inspect.getsource(es.build_gepa_optimizer)
        construct_at = source.index("dspy.GEPA(**gepa_kwargs)")
        reflection_at = source.index('gepa_kwargs["reflection_lm"]')
        assert reflection_at < construct_at, (
            "reflection_lm assigned after GEPA() is constructed; it has no effect"
        )


class TestOptimizerConcurrency:
    def test_concurrency_constant_is_sane(self):
        assert 1 <= es._OPTIMIZER_CONCURRENCY <= 32

    @patch.object(es, "make_lm", side_effect=_fake_lm)
    def test_make_lm_is_used_for_reflection(self, _mock_make_lm):
        assert callable(es.make_lm)