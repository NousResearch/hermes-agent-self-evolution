"""Tests for the auto-triage ranker.

Pure computation over fixed points at fixed timestamps, so every ranking here
is reproducible. The arithmetic is asserted directly rather than through
approximate ordering, because the ranking is what decides where money gets
spent on an optimizer run.
"""

import math

import pytest

from evolution.monitor.metrics import (
    BENCHMARK_SCORE,
    SECONDS_PER_DAY,
    SKILL_SUCCESS_RATE,
    TOOL_SELECTION_ACCURACY,
    USER_CORRECTION,
    MetricPoint,
    MetricStore,
)
from evolution.monitor.triage import (
    AutoTriage,
    TargetType,
    TriageConfig,
    rank_points,
)

T0 = 1_700_000_000.0
DAY = SECONDS_PER_DAY


def point(metric, target, value, days_ago=1.0, samples=1, **metadata):
    return MetricPoint(
        metric=metric,
        target=target,
        value=value,
        timestamp=T0 - days_ago * DAY,
        samples=samples,
        source="test",
        metadata=dict(metadata),
    )


def corrections(target, count, days_ago=1.0, **metadata):
    return [
        point(USER_CORRECTION, target, 1.0, days_ago=days_ago + i, **metadata)
        for i in range(count)
    ]


def by_target(entries):
    return {entry.target: entry for entry in entries}


class TestRankingOrder:
    def test_usage_frequency_breaks_a_tie_on_potential_improvement(self):
        entries = rank_points(
            [
                point(SKILL_SUCCESS_RATE, "busy", 0.5, samples=100),
                point(SKILL_SUCCESS_RATE, "quiet", 0.5, samples=10),
            ],
            now=T0,
        )
        assert [e.target for e in entries] == ["busy", "quiet"]
        assert entries[0].score == pytest.approx(0.5)
        # Ten times the traffic is not ten times the priority. Linear
        # normalization put the quiet target at 0.05, an order of magnitude
        # down, for two targets that are equally broken.
        assert entries[1].score == pytest.approx(0.5 * math.log1p(10) / math.log1p(100))
        assert entries[1].score > 0.2

    def test_potential_improvement_breaks_a_tie_on_usage(self):
        entries = rank_points(
            [
                point(SKILL_SUCCESS_RATE, "healthy", 0.9, samples=100),
                point(SKILL_SUCCESS_RATE, "struggling", 0.4, samples=100),
            ],
            now=T0,
        )
        assert [e.target for e in entries] == ["struggling", "healthy"]

    def test_the_score_is_improvement_times_frequency(self):
        entries = rank_points(
            [
                point(SKILL_SUCCESS_RATE, "alpha", 0.6, samples=50),
                point(SKILL_SUCCESS_RATE, "beta", 0.9, samples=100),
            ],
            now=T0,
        )
        ranked = by_target(entries)
        assert ranked["alpha"].potential_improvement == pytest.approx(0.4)
        assert ranked["alpha"].usage_weight == pytest.approx(
            math.log1p(50) / math.log1p(100)
        )
        assert ranked["alpha"].score == pytest.approx(
            0.4 * ranked["alpha"].usage_weight
        )
        assert ranked["beta"].usage_weight == pytest.approx(1.0)

    def test_a_busy_healthy_target_loses_to_a_quiet_broken_one_only_if_the_gap_is_big(self):
        entries = rank_points(
            [
                point(SKILL_SUCCESS_RATE, "busy_ok", 0.95, samples=1000),
                point(SKILL_SUCCESS_RATE, "quiet_bad", 0.10, samples=100),
            ],
            now=T0,
        )
        ranked = by_target(entries)
        assert ranked["busy_ok"].score == pytest.approx(0.05)
        assert ranked["quiet_bad"].score == pytest.approx(
            0.9 * math.log1p(100) / math.log1p(1000)
        )
        assert entries[0].target == "quiet_bad"

    def test_ordering_is_deterministic_for_identical_scores(self):
        points = [
            point(SKILL_SUCCESS_RATE, "zeta", 0.5, samples=10),
            point(SKILL_SUCCESS_RATE, "alpha", 0.5, samples=10),
        ]
        first = [e.target for e in rank_points(points, now=T0)]
        second = [e.target for e in rank_points(list(reversed(points)), now=T0)]
        assert first == second == ["alpha", "zeta"]

    def test_the_weighted_mean_drives_the_score_not_the_last_reading(self):
        entries = rank_points(
            [
                point(SKILL_SUCCESS_RATE, "arxiv", 0.2, days_ago=3, samples=99),
                point(SKILL_SUCCESS_RATE, "arxiv", 1.0, days_ago=1, samples=1),
            ],
            now=T0,
        )
        assert entries[0].current_value == pytest.approx(0.208)


class TestExplanations:
    def test_every_entry_names_both_ranking_factors(self):
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.5, samples=20)], now=T0
        )[0]
        names = [f.name for f in entry.factors]
        assert names[:2] == ["potential improvement", "usage frequency"]

    def test_explain_states_the_arithmetic(self):
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.5, samples=20)], now=T0
        )[0]
        text = entry.explain()
        assert "score 0.500" in text
        assert "potential improvement" in text
        assert " x " in text

    def test_explain_flags_a_trigger(self):
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.2, samples=50)], now=T0
        )[0]
        assert "TRIGGERED" in entry.explain()

    def test_details_describe_each_factor(self):
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.5, samples=20)], now=T0
        )[0]
        joined = " ".join(entry.details())
        assert "leaving 0.50" in joined
        assert "20 observations" in joined

    def test_entries_serialise_with_their_explanation(self):
        blob = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.5, samples=20)], now=T0
        )[0].to_dict()
        assert blob["target"] == "arxiv"
        assert blob["target_type"] == "skill"
        assert blob["actionable"] is True
        assert "explanation" in blob
        assert blob["factors"][0]["name"] == "potential improvement"


class TestTrendPressure:
    def _declining(self, target, values, samples=50):
        # Spread evenly back from today so any length fits inside the window.
        step = 25 // max(1, len(values) - 1)
        return [
            point(
                SKILL_SUCCESS_RATE,
                target,
                value,
                days_ago=step * (len(values) - 1 - i),
                samples=samples,
            )
            for i, value in enumerate(values)
        ]

    def test_a_declining_target_outranks_an_equally_bad_stable_one(self):
        entries = rank_points(
            self._declining("eroding", [0.90, 0.86, 0.82, 0.78, 0.74, 0.70])
            + self._declining("steady", [0.8] * 6),
            now=T0,
        )
        ranked = by_target(entries)
        assert ranked["eroding"].score == pytest.approx(0.2 * 1.0 * 1.5)
        assert ranked["steady"].score == pytest.approx(0.2)
        assert entries[0].target == "eroding"

    def test_the_decline_shows_up_as_a_named_factor(self):
        entry = by_target(
            rank_points(
                self._declining("eroding", [0.90, 0.86, 0.82, 0.78, 0.74, 0.70]),
                now=T0,
            )
        )["eroding"]
        names = [f.name for f in entry.factors]
        assert "declining trend" in names
        assert entry.trend.significant

    def test_a_significant_decline_triggers_even_below_the_failure_threshold(self):
        entry = rank_points(
            self._declining("slipping", [0.98, 0.96, 0.94, 0.92, 0.90, 0.88]), now=T0
        )[0]
        assert entry.potential_improvement < 0.3
        assert entry.triggered
        assert "significant decline" in entry.trigger_reason

    def test_an_improving_target_gets_no_boost(self):
        entry = rank_points(self._declining("improving", [0.5, 0.6, 0.7]), now=T0)[0]
        assert [f.name for f in entry.factors] == [
            "potential improvement",
            "usage frequency",
        ]
        assert not entry.trend.significant


class TestUsageWeighting:
    """Volume should count without deciding the answer on its own."""

    def _falling(self, target, values, samples):
        step = 25 // max(1, len(values) - 1)
        return [
            point(
                SKILL_SUCCESS_RATE,
                target,
                value,
                days_ago=step * (len(values) - 1 - i),
                samples=samples,
            )
            for i, value in enumerate(values)
        ]

    def _live_example(self):
        """The ranking that exposed the defect.

        ``collapsing`` lost 36 points of success rate, ``slipping`` lost 8, and
        ``chatty`` is healthy but soaks up most of the traffic in the window.
        Under linear normalization chatty's volume squashed both weights and
        the milder decline came out on top.
        """
        return (
            self._falling(
                "collapsing",
                [0.910, 0.838, 0.766, 0.694, 0.622, 0.550],
                samples=50,
            )
            + self._falling(
                "slipping",
                [0.820, 0.804, 0.788, 0.772, 0.756, 0.740],
                samples=100,
            )
            + [point(SKILL_SUCCESS_RATE, "chatty", 0.99, samples=10_000)]
        )

    def test_the_steeper_collapse_outranks_the_milder_decline(self):
        entries = rank_points(self._live_example(), now=T0)
        ranked = by_target(entries)
        assert ranked["collapsing"].score > ranked["slipping"].score
        assert entries[0].target == "collapsing"

    def test_linear_normalization_is_what_got_this_backwards(self):
        # Same points, compression switched off: the old arithmetic exactly.
        entries = rank_points(
            self._live_example(), TriageConfig(usage_compression=0.0), now=T0
        )
        ranked = by_target(entries)
        assert ranked["slipping"].score > ranked["collapsing"].score
        assert ranked["collapsing"].usage_weight == pytest.approx(300 / 10_000)

    def test_volume_still_orders_equally_broken_targets(self):
        entries = rank_points(
            [
                point(SKILL_SUCCESS_RATE, "busy", 0.5, samples=500),
                point(SKILL_SUCCESS_RATE, "medium", 0.5, samples=50),
                point(SKILL_SUCCESS_RATE, "quiet", 0.5, samples=5),
            ],
            now=T0,
        )
        assert [e.target for e in entries] == ["busy", "medium", "quiet"]

    def test_the_busiest_target_still_weighs_one(self):
        entries = rank_points(
            [
                point(SKILL_SUCCESS_RATE, "busy", 0.5, samples=900),
                point(SKILL_SUCCESS_RATE, "quiet", 0.5, samples=9),
            ],
            now=T0,
        )
        assert by_target(entries)["busy"].usage_weight == pytest.approx(1.0)

    def test_a_quiet_target_is_not_crushed_by_one_hot_neighbour(self):
        entries = rank_points(
            [
                point(SKILL_SUCCESS_RATE, "hot", 0.9, samples=100_000),
                point(SKILL_SUCCESS_RATE, "quiet", 0.9, samples=100),
            ],
            now=T0,
        )
        quiet = by_target(entries)["quiet"]
        # Linear normalization left this at 0.001, three orders of magnitude
        # down, which is how a real problem disappears off the bottom.
        assert quiet.usage_weight == pytest.approx(
            math.log1p(100) / math.log1p(100_000)
        )
        assert quiet.usage_weight > 0.3

    def test_compression_is_configurable(self):
        # A rate of 0.2 rather than 0.5 so the quiet target is a demonstrated
        # breach and survives the score floor at every compression setting.
        # Otherwise linear weighting sinks it below min_score and it drops out
        # of the ranking entirely, which is the behaviour under test.
        points = [
            point(SKILL_SUCCESS_RATE, "busy", 0.2, samples=1000),
            point(SKILL_SUCCESS_RATE, "quiet", 0.2, samples=10),
        ]
        weights = []
        for compression in (0.0, 0.01, 1.0, 100.0):
            entries = rank_points(
                points, TriageConfig(usage_compression=compression), now=T0
            )
            weights.append(by_target(entries)["quiet"].usage_weight)
        # Flatter curves lift the quiet target, and 0 is exactly linear.
        assert weights[0] == pytest.approx(0.01)
        assert weights == sorted(weights)
        assert weights[-1] < 1.0

    def test_correction_only_targets_use_the_same_curve(self):
        entries = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.9, samples=1000)]
            + corrections("web_search", 8),
            now=T0,
        )
        entry = by_target(entries)["web_search"]
        assert entry.usage_weight == pytest.approx(math.log1p(8) / math.log1p(1000))


class TestFailureRateLabelling:
    def test_headroom_and_failure_rate_agree_at_a_ceiling_of_one(self):
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.62, samples=50)], now=T0
        )[0]
        assert entry.headroom == pytest.approx(0.38)
        assert entry.failure_rate == pytest.approx(0.38)

    def test_a_lower_ceiling_separates_the_two_numbers(self):
        config = TriageConfig(ceiling=0.8, failure_threshold=0.3, min_samples=5)
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.45, samples=50)], config, now=T0
        )[0]
        assert entry.headroom == pytest.approx(0.35)
        assert entry.potential_improvement == pytest.approx(0.35)
        # 45% success is a 55% failure rate. The old message called the 35%
        # headroom a "failure rate" and understated it by twenty points.
        assert entry.failure_rate == pytest.approx(0.55)

    def test_the_trigger_message_describes_the_quantity_it_tested(self):
        config = TriageConfig(ceiling=0.8, failure_threshold=0.3, min_samples=5)
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.35, samples=50)], config, now=T0
        )[0]
        assert entry.triggered
        # The condition is headroom >= threshold, so headroom is what the
        # sentence leads with, and the real failure rate is named separately.
        assert "headroom 45%" in entry.trigger_reason
        assert "at or above threshold 30%" in entry.trigger_reason
        assert "failure rate 65%" in entry.trigger_reason

    def test_both_numbers_serialise(self):
        config = TriageConfig(ceiling=0.8)
        blob = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.45, samples=50)], config, now=T0
        )[0].to_dict()
        assert blob["headroom"] == pytest.approx(0.35)
        assert blob["failure_rate"] == pytest.approx(0.55)
        assert blob["potential_improvement"] == pytest.approx(0.35)

    def test_a_correction_only_entry_has_no_failure_rate(self):
        entry = rank_points(corrections("web_search", 6), now=T0)[0]
        assert entry.current_value is None
        assert entry.failure_rate is None
        assert entry.headroom is None


class TestRankingConfidence:
    def _series(self, target, values, samples=60):
        return [
            point(SKILL_SUCCESS_RATE, target, value, days_ago=35 - 5 * i, samples=samples)
            for i, value in enumerate(values)
        ]

    OSCILLATION = [0.90, 0.35, 0.85, 0.30, 0.88, 0.33, 0.60]
    STEADY_EROSION = [0.91, 0.86, 0.78, 0.71, 0.62, 0.55]

    def test_a_noisy_trend_is_visibly_uncertain(self):
        config = TriageConfig(window_days=40)
        entry = rank_points(self._series("noisy", self.OSCILLATION), config, now=T0)[0]
        assert entry.trend_p_value == pytest.approx(0.582, abs=0.01)
        assert entry.trend_r_squared == pytest.approx(0.06, abs=0.01)
        assert not entry.trend.significant
        assert "p=0.582" in entry.explain()

    def test_a_clean_decline_is_visibly_certain(self):
        config = TriageConfig(window_days=40)
        entry = rank_points(self._series("eroding", self.STEADY_EROSION), config, now=T0)[0]
        assert entry.trend_p_value < 0.05
        assert entry.trend_r_squared > 0.95
        assert "R²=" in entry.explain()

    def test_the_noisy_target_does_not_get_the_decline_multiplier(self):
        config = TriageConfig(window_days=40)
        noisy = rank_points(self._series("noisy", self.OSCILLATION), config, now=T0)[0]
        clean = rank_points(self._series("eroding", self.STEADY_EROSION), config, now=T0)[0]
        assert "declining trend" not in [f.name for f in noisy.factors]
        assert "declining trend" in [f.name for f in clean.factors]
        # The noisy target still fires, but on its 40% headroom, which is
        # measured. It must not fire on a trend that is not there.
        assert "significant decline" not in noisy.trigger_reason
        assert "at or above threshold" in noisy.trigger_reason
        assert "significant decline" in clean.trigger_reason

    def test_confidence_travels_in_the_serialised_entry(self):
        config = TriageConfig(window_days=40)
        blob = rank_points(self._series("noisy", self.OSCILLATION), config, now=T0)[
            0
        ].to_dict()
        assert blob["trend_p_value"] == pytest.approx(0.582, abs=0.01)
        assert blob["trend_r_squared"] == pytest.approx(0.06, abs=0.01)
        assert blob["trend"]["statistically_significant"] is False

    def test_a_single_reading_claims_no_confidence(self):
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.5, samples=20)], now=T0
        )[0]
        assert entry.trend_p_value is None
        assert entry.trend_r_squared is None
        assert entry.confidence_note() == ""
        assert "p=" not in entry.explain()

    def test_a_correction_only_entry_claims_no_confidence(self):
        entry = rank_points(corrections("web_search", 6), now=T0)[0]
        assert entry.trend is None
        assert entry.trend_p_value is None
        assert entry.confidence_note() == ""


class TestCorrections:
    def test_corrections_boost_a_measured_target(self):
        base = [point(TOOL_SELECTION_ACCURACY, "search_files", 0.8, samples=100)]
        quiet = [point(TOOL_SELECTION_ACCURACY, "read_file", 0.8, samples=100)]
        entries = rank_points(base + quiet + corrections("search_files", 5), now=T0)

        ranked = by_target(entries)
        assert ranked["search_files"].score == pytest.approx(0.2 * 1.25)
        assert ranked["read_file"].score == pytest.approx(0.2)
        assert entries[0].target == "search_files"

    def test_correction_pressure_saturates(self):
        entries = rank_points(
            [point(TOOL_SELECTION_ACCURACY, "search_files", 0.8, samples=100)]
            + corrections("search_files", 40),
            now=T0,
        )
        # 40 corrections cannot buy more than the saturated 1.5x multiplier.
        assert entries[0].score == pytest.approx(0.2 * 1.5)

    def test_a_target_known_only_from_corrections_still_ranks(self):
        entries = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.9, samples=100)]
            + corrections("web_search", 6),
            now=T0,
        )
        entry = by_target(entries)["web_search"]
        assert entry.metric == USER_CORRECTION
        assert entry.current_value is None
        assert entry.corrections == 6
        assert entry.score == pytest.approx(0.6 * math.log1p(6) / math.log1p(100))

    def test_enough_corrections_alone_fire_a_trigger(self):
        entry = rank_points(corrections("web_search", 5), now=T0)[0]
        assert entry.triggered
        assert "5 user corrections" in entry.trigger_reason

    def test_a_couple_of_corrections_do_not_fire_a_trigger(self):
        entry = rank_points(corrections("web_search", 2), now=T0)[0]
        assert not entry.triggered


class TestThresholdTriggering:
    def test_a_rate_sitting_exactly_on_the_threshold_does_not_fire(self):
        """Landing on the line is not evidence of crossing it.

        A rate of 0.75 against a 0.25 threshold puts the point estimate exactly
        at the boundary, so its interval always straddles it however many
        samples arrive. Spending an optimization run on that is spending it on
        a coin toss.
        """
        config = TriageConfig(failure_threshold=0.25, min_samples=5)
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.75, samples=20)], config, now=T0
        )[0]
        assert entry.potential_improvement == pytest.approx(0.25)
        assert not entry.triggered

    def test_a_demonstrated_breach_fires(self):
        """Worse than the threshold by more than the noise, so it fires."""
        config = TriageConfig(failure_threshold=0.25, min_samples=5)
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.55, samples=100)], config, now=T0
        )[0]
        assert entry.triggered
        assert "at or above threshold" in entry.trigger_reason

    def test_the_same_breach_on_a_thin_sample_does_not_fire(self):
        """Identical rate, too few observations to rule out noise."""
        config = TriageConfig(failure_threshold=0.25, min_samples=5)
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.55, samples=8)], config, now=T0
        )[0]
        assert entry.potential_improvement == pytest.approx(0.45)
        assert not entry.triggered

    def test_just_below_the_threshold_does_not_fire(self):
        config = TriageConfig(failure_threshold=0.25, min_samples=5)
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.80, samples=20)], config, now=T0
        )[0]
        assert not entry.triggered
        assert entry.trigger_reason == ""

    def test_a_thin_sample_cannot_fire_a_trigger(self):
        config = TriageConfig(failure_threshold=0.25, min_samples=5)
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.10, samples=3)], config, now=T0
        )[0]
        assert entry.potential_improvement == pytest.approx(0.9)
        assert not entry.triggered

    def test_the_sample_floor_is_inclusive(self):
        config = TriageConfig(failure_threshold=0.25, min_samples=5)
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.10, samples=5)], config, now=T0
        )[0]
        assert entry.triggered

    def test_a_custom_threshold_is_honoured(self):
        points = [point(SKILL_SUCCESS_RATE, "arxiv", 0.85, samples=50)]
        assert not rank_points(points, TriageConfig(failure_threshold=0.3), now=T0)[0].triggered
        # 0.85 over 50 samples reaches 0.93 at the top of its interval, so a
        # 0.10 threshold is not demonstrated; 0.60 over 200 is.
        assert not rank_points(points, TriageConfig(failure_threshold=0.1), now=T0)[0].triggered
        clear = [point(SKILL_SUCCESS_RATE, "arxiv", 0.60, samples=200)]
        assert rank_points(clear, TriageConfig(failure_threshold=0.1), now=T0)[0].triggered

    def test_triggers_helper_returns_only_fired_entries(self, tmp_path):
        store = MetricStore(tmp_path / "m.jsonl", clock=lambda: T0)
        store.extend(
            [
                point(SKILL_SUCCESS_RATE, "broken", 0.2, samples=50),
                point(SKILL_SUCCESS_RATE, "fine", 0.99, samples=50),
            ]
        )
        fired = AutoTriage(store, TriageConfig()).triggers(now=T0)
        assert [e.target for e in fired] == ["broken"]


class TestFilteringAndScope:
    def test_history_outside_the_window_is_ignored(self):
        entries = rank_points(
            [point(SKILL_SUCCESS_RATE, "ancient", 0.1, days_ago=90, samples=100)],
            TriageConfig(window_days=30),
            now=T0,
        )
        assert entries == []

    def test_weak_candidates_are_dropped(self):
        entries = rank_points(
            [
                point(SKILL_SUCCESS_RATE, "busy", 0.5, samples=1000),
                point(SKILL_SUCCESS_RATE, "negligible", 0.99, samples=5),
            ],
            now=T0,
        )
        assert [e.target for e in entries] == ["busy"]

    def test_a_triggered_candidate_survives_the_score_floor(self):
        config = TriageConfig(min_score=0.2)
        entries = rank_points(
            [
                point(SKILL_SUCCESS_RATE, "busy", 0.99, samples=100_000),
                point(SKILL_SUCCESS_RATE, "rare_but_broken", 0.05, samples=5),
            ],
            config,
            now=T0,
        )
        ranked = by_target(entries)
        rare = ranked["rare_but_broken"]
        assert rare.score < config.min_score
        assert rare.triggered
        # The healthy target scores below the floor too, and nothing triggered
        # it, so it is the one that gets dropped.
        assert "busy" not in ranked

    def test_limit_truncates_the_ranking(self, tmp_path):
        store = MetricStore(tmp_path / "m.jsonl", clock=lambda: T0)
        store.extend(
            [
                point(SKILL_SUCCESS_RATE, f"skill{i}", 0.5, samples=10 * (i + 1))
                for i in range(4)
            ]
        )
        assert len(AutoTriage(store).rank(now=T0, limit=2)) == 2

    def test_autotriage_uses_the_store_clock_when_now_is_absent(self, tmp_path):
        store = MetricStore(tmp_path / "m.jsonl", clock=lambda: T0)
        store.extend([point(SKILL_SUCCESS_RATE, "arxiv", 0.5, days_ago=2, samples=20)])
        assert [e.target for e in AutoTriage(store).rank()] == ["arxiv"]


class TestTargetTyping:
    def test_skill_metrics_produce_skill_targets(self):
        entry = rank_points([point(SKILL_SUCCESS_RATE, "arxiv", 0.5, samples=20)], now=T0)[0]
        assert entry.target_type is TargetType.SKILL
        assert entry.actionable

    def test_tool_metrics_produce_tool_targets(self):
        entry = rank_points(
            [point(TOOL_SELECTION_ACCURACY, "read_file", 0.5, samples=20)], now=T0
        )[0]
        assert entry.target_type is TargetType.TOOL

    def test_benchmarks_are_ranked_but_never_actionable(self):
        entry = rank_points([point(BENCHMARK_SCORE, "tblite", 0.4, samples=20)], now=T0)[0]
        assert entry.target_type is TargetType.BENCHMARK
        assert not entry.actionable
        assert "advisory" in entry.explain()

    def test_metadata_declares_the_target_type(self):
        entry = rank_points(
            corrections("MEMORY_GUIDANCE", 6, target_type="prompt"), now=T0
        )[0]
        assert entry.target_type is TargetType.PROMPT
        assert entry.actionable

    def test_an_unrecognised_declared_type_is_ignored(self):
        entry = rank_points(corrections("read_file", 6, target_type="nonsense"), now=T0)[0]
        assert entry.target_type is TargetType.TOOL

    def test_a_correction_inherits_the_type_seen_earlier_in_history(self):
        # The success-rate reading is older than the window, so only the
        # corrections rank - but the target is still a skill, not a tool.
        entry = rank_points(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.9, days_ago=80, samples=40)]
            + corrections("arxiv", 6),
            TriageConfig(window_days=30),
            now=T0,
        )[0]
        assert entry.target == "arxiv"
        assert entry.target_type is TargetType.SKILL

    def test_an_extra_metric_can_declare_its_own_type(self):
        config = TriageConfig(extra_metric_types={"tool_crash_free_rate": TargetType.CODE})
        entry = rank_points(
            [point("tool_crash_free_rate", "file_tools", 0.4, samples=30)],
            config,
            now=T0,
        )[0]
        assert entry.target_type is TargetType.CODE

    def test_actionable_only_hides_advisory_entries(self, tmp_path):
        store = MetricStore(tmp_path / "m.jsonl", clock=lambda: T0)
        store.extend(
            [
                point(BENCHMARK_SCORE, "tblite", 0.4, samples=20),
                point(SKILL_SUCCESS_RATE, "arxiv", 0.4, samples=20),
            ]
        )
        entries = AutoTriage(store).rank(now=T0, actionable_only=True)
        assert [e.target for e in entries] == ["arxiv"]

    def test_declining_helper_surfaces_deteriorating_series(self, tmp_path):
        store = MetricStore(tmp_path / "m.jsonl", clock=lambda: T0)
        store.extend(
            [
                point(SKILL_SUCCESS_RATE, "eroding", value, days_ago=21 - 7 * i, samples=50)
                for i, value in enumerate([0.9, 0.8, 0.7])
            ]
            + [point(SKILL_SUCCESS_RATE, "steady", 0.7, samples=50)]
        )
        assert [e.target for e in AutoTriage(store).declining(now=T0)] == ["eroding"]


class TestBookkeepingIsNotATarget:
    def test_the_loops_own_records_do_not_become_candidates(self):
        entries = rank_points(
            [
                point("optimization_run", "arxiv", 1.0, samples=1),
                point(SKILL_SUCCESS_RATE, "arxiv", 0.5, samples=20),
            ],
            now=T0,
        )
        assert [(e.metric, e.target) for e in entries] == [(SKILL_SUCCESS_RATE, "arxiv")]

    def test_an_empty_history_ranks_nothing(self):
        assert rank_points([], now=T0) == []
