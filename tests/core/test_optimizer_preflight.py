"""Preflight: catch optimizer signature breakage in seconds, not 15 minutes.

Three bugs in the GEPA setup path each cost a full dataset build to surface
(renamed budget parameter silently falling back to MIPROv2, missing required
reflection_lm, and a 3-arg metric against GEPA's 5-arg bind). preflight_optimizer
exists so that class of failure is found before any LLM spend.
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest

from evolution.skills import evolve_skill as es


class TestBuildGepaOptimizer:
    def test_builds_against_real_installed_signature(self):
        """The real dspy.GEPA, real signature checks -- no mocks.

        This is the check that would have caught all three bugs.
        """
        optimizer = es.build_gepa_optimizer(
            10, "openrouter/~deepseek/deepseek-pro-latest"
        )
        import dspy

        assert isinstance(optimizer, dspy.GEPA)

    def test_metric_binds_five_positional_args(self):
        """GEPA's own arity contract, checked directly."""
        inspect.signature(es._gepa_metric).bind(None, None, None, None, None)

    def test_unknown_budget_parameter_raises_loudly(self):
        """A DSPy version with no known budget param must not silently no-op."""
        import dspy

        # A stand-in GEPA whose __init__ exposes no budget parameter. Patching
        # inspect.signature instead would leak globally -- es.inspect IS the
        # inspect module, so it would break every other introspecting test.
        class _FakeGepaNoBudget:
            def __init__(self, *args, **kwargs):
                pass

        real_gepa = dspy.GEPA
        try:
            dspy.GEPA = _FakeGepaNoBudget
            with pytest.raises(RuntimeError, match="budget parameter"):
                es.build_gepa_optimizer(10, "openrouter/~deepseek/deepseek-pro-latest")
        finally:
            dspy.GEPA = real_gepa

    def test_budget_value_is_passed_through(self):
        """iterations must reach whichever budget kwarg this version has.

        The factory introspects the class it is about to construct, so the
        stand-in must expose the real budget parameter -- a bare MagicMock or a
        kwargs-only __init__ has no discoverable signature and the factory
        raises before reaching the call under test.
        """
        import dspy

        params = inspect.signature(dspy.GEPA.__init__).parameters
        expected = next(
            k for k in ("max_steps", "max_full_evals", "max_metric_calls") if k in params
        )
        captured = {}

        def _make_init(param_name):
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
            # Advertise the parameter so the factory's introspection sees it.
            __init__.__signature__ = inspect.Signature(
                [
                    inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    inspect.Parameter(
                        param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=7
                    ),
                    inspect.Parameter(
                        "metric", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
                    ),
                    inspect.Parameter(
                        "reflection_lm",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                    ),
                    inspect.Parameter(
                        "num_threads", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
                    ),
                ]
            )
            return __init__

        real_gepa = dspy.GEPA
        try:
            dspy.GEPA = type("FakeGEPA", (), {"__init__": _make_init(expected)})
            es.build_gepa_optimizer(7, "openrouter/~deepseek/deepseek-pro-latest")
        finally:
            dspy.GEPA = real_gepa

        assert captured[expected] == 7


class TestPreflight:
    def test_returns_true_when_optimizer_builds(self):
        assert es.preflight_optimizer(
            10, "openrouter/~deepseek/deepseek-pro-latest"
        ) is True

    def test_returns_false_instead_of_raising(self):
        """Preflight reports; it must not take the whole run down."""
        with patch.object(
            es, "build_gepa_optimizer", side_effect=TypeError("bad kwarg")
        ):
            assert es.preflight_optimizer(10, "some-model") is False

    def test_runs_before_dataset_build(self):
        """Ordering guard: preflight must precede the expensive mining.

        Asserted against source because the ordering is the entire point --
        each of the three bugs cost a ~15 minute build to discover.
        """
        source = inspect.getsource(es.evolve)
        assert "preflight_optimizer" in source
        assert source.index("preflight_optimizer") < source.index(
            "Building evaluation dataset"
        ), "preflight must run before the dataset build"