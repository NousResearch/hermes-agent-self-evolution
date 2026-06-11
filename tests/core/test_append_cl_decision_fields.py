"""Unit tests for `append_cl_decision_fields`."""

from __future__ import annotations

import math

from evolution.core.quality_gate import (
    CL_PRIMARY_GROWTH_FREE_THRESHOLD,
    CL_PRIMARY_GROWTH_SLOPE,
    CL_PRIMARY_SYNTH_TOLERANCE,
    append_cl_decision_fields,
)


def _call(payload: dict, **overrides) -> None:
    kwargs = dict(
        cached_baseline_cl_per_example=[1.0, 0.0, 1.0],
        evolved_cl_per_example=[1.0, 1.0, 1.0],
        avg_baseline=0.60,
        avg_evolved=0.65,
        growth_pct=0.25,
        cl_eval_cost_usd=0.0123,
        preflight_holdout_score=0.7,
        preflight_cl_score=0.4,
        closed_loop_agent_model="openai/gpt-4.1-mini",
    )
    kwargs.update(overrides)
    append_cl_decision_fields(payload, **kwargs)


class TestAppendClDecisionFields:
    def test_all_fields_added(self):
        payload: dict = {}
        _call(payload)
        assert set(payload.keys()) == {
            "baseline_closed_loop_per_example",
            "evolved_closed_loop_per_example",
            "evolved_closed_loop_errored_tasks",
            "cl_tasks_gained",
            "cl_required_gain",
            "cl_noise_floor_passes",
            "noise_aware_gate",
            "synthetic_sanity_check",
            "evolved_cl_eval_cost_usd",
            "band_trigger_score",
            "validator_agent_model",
        }
        # Default call supplies no noise floor → legacy behavior recorded.
        assert payload["cl_noise_floor_passes"] == 0.0
        assert payload["noise_aware_gate"] is False
        assert payload["cl_tasks_gained"] == 3 - 2
        assert payload["evolved_cl_eval_cost_usd"] == 0.0123
        assert payload["band_trigger_score"] == {"holdout": 0.7, "closed_loop": 0.4}
        assert payload["validator_agent_model"] == "openai/gpt-4.1-mini"
        sanity = payload["synthetic_sanity_check"]
        assert sanity["tolerance"] == CL_PRIMARY_SYNTH_TOLERANCE
        assert sanity["baseline_mean"] == 0.60
        assert sanity["evolved_mean"] == 0.65
        assert sanity["passed"] is True

    def test_errored_tasks_is_empty_list(self):
        payload: dict = {}
        _call(payload)
        assert payload["evolved_closed_loop_errored_tasks"] == []

    def test_cl_required_gain_uses_constants_not_magic_numbers(self):
        # Pin the formula so silent constant→literal substitutions break this test.
        payload: dict = {}
        _call(payload, growth_pct=CL_PRIMARY_GROWTH_FREE_THRESHOLD + 0.5)
        expected = max(1, math.ceil(CL_PRIMARY_GROWTH_SLOPE * 0.5))
        assert payload["cl_required_gain"] == expected

    def test_synthetic_sanity_check_passed_reflects_tolerance(self):
        # Exactly at the boundary: passes.
        boundary_payload: dict = {}
        _call(
            boundary_payload,
            avg_baseline=0.50,
            avg_evolved=0.50 - CL_PRIMARY_SYNTH_TOLERANCE,
        )
        assert boundary_payload["synthetic_sanity_check"]["passed"] is True

        # Just past the boundary: fails.
        over_payload: dict = {}
        _call(
            over_payload,
            avg_baseline=0.50,
            avg_evolved=0.50 - CL_PRIMARY_SYNTH_TOLERANCE - 0.001,
        )
        assert over_payload["synthetic_sanity_check"]["passed"] is False
