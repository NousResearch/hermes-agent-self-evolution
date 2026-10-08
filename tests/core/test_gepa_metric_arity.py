"""The GEPA metric arity contract.

GEPA asserts on the metric signature: it binds five positional args
(gold, pred, trace, pred_name, pred_trace) and raises TypeError otherwise. The
shared skill_fitness_metric has the DSPy-standard 3-arg shape, so the evolve
path must adapt it. This is the third bug in the same family -- a renamed
parameter, a missing required reflection_lm, and now arity -- all of which
surfaced only at runtime, after a full dataset build.
"""

import inspect

import pytest


def _gepa_metric_factory():
    """Rebuild the adapter the same way evolve_skill does."""
    from evolution.core.fitness import skill_fitness_metric

    def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return skill_fitness_metric(gold, pred, trace)

    return gepa_metric


class TestGepaMetricArity:
    def test_adapter_binds_five_positional_args(self):
        """This is the exact check GEPA performs on the metric."""
        metric = _gepa_metric_factory()
        inspect.signature(metric).bind(None, None, None, None, None)  # must not raise

    def test_adapter_rejects_fewer_args_like_the_shared_metric(self):
        """Without the adapter, the 3-arg metric fails GEPA's bind()."""
        from evolution.core.fitness import skill_fitness_metric

        with pytest.raises(TypeError):
            inspect.signature(skill_fitness_metric).bind(None, None, None, None, None)

    def test_gepa_constructs_with_the_adapter(self):
        """End-to-end: real GEPA, real signature check, adapter metric."""
        import dspy
        from unittest.mock import MagicMock

        params = inspect.signature(dspy.GEPA.__init__).parameters
        kw = {"metric": _gepa_metric_factory()}
        if "max_steps" in params:
            kw["max_steps"] = 10
        elif "max_full_evals" in params:
            kw["max_full_evals"] = 10
        elif "max_metric_calls" in params:
            kw["max_metric_calls"] = 10
        if "reflection_lm" in params:
            kw["reflection_lm"] = MagicMock()

        optimizer = dspy.GEPA(**kw)
        assert isinstance(optimizer, dspy.GEPA)

    def test_adapter_forwards_trace_positionally(self):
        """trace must reach skill_fitness_metric, not be dropped."""
        from evolution.core import fitness

        seen = {}

        def spy(example, prediction, trace=None):
            seen["trace"] = trace
            return 0.5

        original = fitness.skill_fitness_metric
        try:
            fitness.skill_fitness_metric = spy
            metric = _gepa_metric_factory()
            sentinel = object()
            metric("gold", "pred", sentinel)
            assert seen["trace"] is sentinel
        finally:
            fitness.skill_fitness_metric = original