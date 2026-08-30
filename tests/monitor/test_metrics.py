"""Tests for the performance history store.

Entirely offline: a JSONL file in tmp_path, an injected clock, and fixed
timestamps. Nothing here reads the wall clock, so a trend that is declining
today is still declining when this suite runs in a year.
"""

import contextlib
import json
import os

import pytest

from evolution.monitor.metrics import (
    BENCHMARK_SCORE,
    HIGHER_IS_BETTER,
    SECONDS_PER_DAY,
    SKILL_SUCCESS_RATE,
    TOOL_SELECTION_ACCURACY,
    TRACKED_METRICS,
    USER_CORRECTION,
    MetricPoint,
    MetricStore,
    TrendDirection,
    compute_trend,
    summarize,
)

# A fixed instant to hang every fixture off: 2023-11-14T22:13:20Z.
T0 = 1_700_000_000.0

DAY = SECONDS_PER_DAY


def point(metric, target, value, days_ago=0.0, samples=1, source="test", **metadata):
    return MetricPoint(
        metric=metric,
        target=target,
        value=value,
        timestamp=T0 - days_ago * DAY,
        samples=samples,
        source=source,
        metadata=dict(metadata),
    )


@pytest.fixture
def store(tmp_path):
    """A store whose clock is frozen at T0."""
    return MetricStore(tmp_path / "history" / "metrics.jsonl", clock=lambda: T0)


class TestMetricPoint:
    def test_round_trips_through_a_dict(self):
        original = point(SKILL_SUCCESS_RATE, "arxiv", 0.75, samples=12, note="hello")
        restored = MetricPoint.from_dict(original.to_dict())
        assert restored == original

    def test_round_trips_through_a_json_line(self):
        original = point(BENCHMARK_SCORE, "tblite", 0.62, source="benchmark")
        restored = MetricPoint.from_json_line(original.to_json_line())
        assert restored.metric == BENCHMARK_SCORE
        assert restored.target == "tblite"
        assert restored.value == pytest.approx(0.62)
        assert restored.source == "benchmark"

    def test_serialised_line_carries_a_human_readable_timestamp(self):
        blob = json.loads(point(SKILL_SUCCESS_RATE, "arxiv", 1.0).to_json_line())
        assert blob["at"].startswith("2023-11-14T")
        assert blob["timestamp"] == pytest.approx(T0)

    def test_can_be_rebuilt_from_the_iso_field_alone(self):
        blob = point(SKILL_SUCCESS_RATE, "arxiv", 0.5).to_dict()
        del blob["timestamp"]
        assert MetricPoint.from_dict(blob).timestamp == pytest.approx(T0)

    def test_empty_metric_is_rejected(self):
        with pytest.raises(ValueError):
            MetricPoint(metric="  ", target="arxiv", value=1.0, timestamp=T0)

    def test_empty_target_is_rejected(self):
        with pytest.raises(ValueError):
            MetricPoint(metric=SKILL_SUCCESS_RATE, target="", value=1.0, timestamp=T0)

    def test_negative_sample_count_is_rejected(self):
        with pytest.raises(ValueError):
            MetricPoint(
                metric=SKILL_SUCCESS_RATE,
                target="arxiv",
                value=1.0,
                timestamp=T0,
                samples=-1,
            )

    def test_values_are_coerced_to_numbers(self):
        p = MetricPoint(
            metric=SKILL_SUCCESS_RATE,
            target="arxiv",
            value="0.5",
            timestamp="1700000000",
            samples="4",
        )
        assert isinstance(p.value, float) and p.value == 0.5
        assert isinstance(p.samples, int) and p.samples == 4
        assert p.timestamp == pytest.approx(T0)

    def test_corrections_are_the_only_signal_where_higher_is_worse(self):
        assert HIGHER_IS_BETTER[USER_CORRECTION] is False
        for metric in TRACKED_METRICS:
            if metric != USER_CORRECTION:
                assert HIGHER_IS_BETTER[metric] is True


class TestStoreWriteAndReload:
    def test_missing_file_reads_as_empty_history(self, store):
        assert store.load() == []

    def test_append_creates_parent_directories(self, store):
        store.append(point(SKILL_SUCCESS_RATE, "arxiv", 0.9))
        assert store.path.exists()
        assert store.path.parent.is_dir()

    def test_record_uses_the_injected_clock(self, store):
        written = store.record(SKILL_SUCCESS_RATE, "arxiv", 0.9)
        assert written.timestamp == pytest.approx(T0)

    def test_explicit_timestamp_beats_the_clock(self, store):
        written = store.record(
            SKILL_SUCCESS_RATE, "arxiv", 0.9, timestamp=T0 - 5 * DAY
        )
        assert written.timestamp == pytest.approx(T0 - 5 * DAY)

    def test_append_and_reload_round_trip(self, store):
        store.record(SKILL_SUCCESS_RATE, "arxiv", 0.4, samples=8, metadata={"run": 1})
        store.record(TOOL_SELECTION_ACCURACY, "read_file", 0.95, samples=200)

        reloaded = store.load()
        assert [p.metric for p in reloaded] == [
            SKILL_SUCCESS_RATE,
            TOOL_SELECTION_ACCURACY,
        ]
        assert reloaded[0].metadata == {"run": 1}
        assert reloaded[1].samples == 200

    def test_extend_writes_every_point(self, store):
        store.extend(
            [point(SKILL_SUCCESS_RATE, "arxiv", 0.5, days_ago=n) for n in range(4)]
        )
        assert len(store.load()) == 4

    def test_a_truncated_line_does_not_blind_the_rest(self, store):
        store.record(SKILL_SUCCESS_RATE, "arxiv", 0.4)
        with store.path.open("a", encoding="utf-8") as handle:
            handle.write('{"metric": "skill_suc\n')
        store.record(SKILL_SUCCESS_RATE, "arxiv", 0.6)

        points = store.load()
        assert len(points) == 2
        assert store.skipped_lines == 1

    def test_blank_lines_are_not_counted_as_damage(self, store):
        store.record(SKILL_SUCCESS_RATE, "arxiv", 0.4)
        with store.path.open("a", encoding="utf-8") as handle:
            handle.write("\n\n")
        assert len(store.load()) == 1
        assert store.skipped_lines == 0


class TestQueries:
    @pytest.fixture
    def filled(self, store):
        store.extend(
            [
                point(SKILL_SUCCESS_RATE, "arxiv", 0.9, days_ago=40, samples=10),
                point(SKILL_SUCCESS_RATE, "arxiv", 0.7, days_ago=10, samples=10),
                point(SKILL_SUCCESS_RATE, "debugging", 0.5, days_ago=5, samples=30),
                point(TOOL_SELECTION_ACCURACY, "read_file", 0.8, days_ago=2, samples=90),
                point(BENCHMARK_SCORE, "tblite", 0.61, days_ago=1, source="benchmark"),
            ]
        )
        return store

    def test_filter_by_metric(self, filled):
        assert len(filled.query(metric=SKILL_SUCCESS_RATE)) == 3

    def test_filter_by_several_metrics(self, filled):
        found = filled.query(metric=[BENCHMARK_SCORE, TOOL_SELECTION_ACCURACY])
        assert {p.metric for p in found} == {BENCHMARK_SCORE, TOOL_SELECTION_ACCURACY}

    def test_filter_by_target(self, filled):
        assert len(filled.query(target="arxiv")) == 2

    def test_filter_by_source(self, filled):
        assert len(filled.query(source="benchmark")) == 1

    def test_results_come_back_oldest_first(self, filled):
        stamps = [p.timestamp for p in filled.query()]
        assert stamps == sorted(stamps)

    def test_since_is_inclusive(self, filled):
        cutoff = T0 - 10 * DAY
        found = filled.query(metric=SKILL_SUCCESS_RATE, since=cutoff)
        assert len(found) == 2

    def test_until_excludes_later_points(self, filled):
        found = filled.query(until=T0 - 30 * DAY)
        assert len(found) == 1

    def test_window_drops_history_older_than_the_window(self, filled):
        recent = filled.window(30, now=T0, metric=SKILL_SUCCESS_RATE, target="arxiv")
        assert len(recent) == 1
        assert recent[0].value == pytest.approx(0.7)

    def test_latest_returns_the_newest_point(self, filled):
        assert filled.latest(SKILL_SUCCESS_RATE, "arxiv").value == pytest.approx(0.7)

    def test_latest_is_none_for_an_unknown_series(self, filled):
        assert filled.latest(SKILL_SUCCESS_RATE, "nonexistent") is None

    def test_pairs_lists_every_series(self, filled):
        assert (SKILL_SUCCESS_RATE, "debugging") in filled.pairs()
        assert len(filled.pairs()) == 4

    def test_targets_can_be_scoped_to_one_metric(self, filled):
        assert filled.targets(metric=SKILL_SUCCESS_RATE) == ["arxiv", "debugging"]


class TestAggregation:
    def test_weighted_mean_respects_sample_counts(self):
        points = [
            point(SKILL_SUCCESS_RATE, "arxiv", 1.0, days_ago=2, samples=1),
            point(SKILL_SUCCESS_RATE, "arxiv", 0.0, days_ago=1, samples=99),
        ]
        aggregate = summarize(SKILL_SUCCESS_RATE, "arxiv", points)
        assert aggregate.mean == pytest.approx(0.5)
        assert aggregate.weighted_mean == pytest.approx(0.01)
        assert aggregate.samples == 100
        assert aggregate.count == 2

    def test_last_value_is_chronological_not_file_order(self):
        points = [
            point(SKILL_SUCCESS_RATE, "arxiv", 0.2, days_ago=1),
            point(SKILL_SUCCESS_RATE, "arxiv", 0.9, days_ago=9),
        ]
        assert summarize(SKILL_SUCCESS_RATE, "arxiv", points).last_value == pytest.approx(0.2)

    def test_empty_aggregate_is_flagged(self):
        aggregate = summarize(SKILL_SUCCESS_RATE, "arxiv", [])
        assert aggregate.empty
        assert aggregate.samples == 0

    def test_store_summarize_honours_the_window(self, store):
        store.extend(
            [
                point(SKILL_SUCCESS_RATE, "arxiv", 0.1, days_ago=60, samples=5),
                point(SKILL_SUCCESS_RATE, "arxiv", 0.9, days_ago=1, samples=5),
            ]
        )
        aggregate = store.summarize(SKILL_SUCCESS_RATE, "arxiv", window_days=30, now=T0)
        assert aggregate.count == 1
        assert aggregate.weighted_mean == pytest.approx(0.9)


class TestTrends:
    def _series(self, values, target="arxiv", metric=SKILL_SUCCESS_RATE, step_days=7):
        return [
            point(metric, target, value, days_ago=(len(values) - 1 - i) * step_days)
            for i, value in enumerate(values)
        ]

    def test_rising_series_is_rising(self):
        trend = compute_trend(self._series([0.4, 0.6, 0.8]))
        assert trend.direction is TrendDirection.RISING
        assert trend.slope_per_day > 0
        assert not trend.significant

    def test_declining_series_is_declining_and_significant(self):
        # Six readings, not three. Three points on a straight line are one
        # ordering in six, p = 0.33, and cannot carry a significance claim; the
        # same shape over six points is one in 720. Direction is descriptive and
        # is reported for both, but only the longer run is evidence.
        trend = compute_trend(self._series([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]))
        assert trend.direction is TrendDirection.DECLINING
        assert trend.change == pytest.approx(-0.5)
        assert trend.significant
        assert trend.is_deterioration

    def test_three_collinear_points_describe_a_decline_without_claiming_evidence(self):
        trend = compute_trend(self._series([0.9, 0.8, 0.7]))
        assert trend.direction is TrendDirection.DECLINING
        assert not trend.significant

    def test_flat_series_is_flat(self):
        trend = compute_trend(self._series([0.8, 0.8, 0.8]))
        assert trend.direction is TrendDirection.FLAT
        assert trend.slope_per_day == pytest.approx(0.0)
        assert not trend.significant

    def test_a_tiny_wobble_stays_flat(self):
        trend = compute_trend(self._series([0.80, 0.81, 0.79]))
        assert trend.direction is TrendDirection.FLAT

    def test_too_few_points_is_unknown_not_a_guess(self):
        trend = compute_trend(self._series([0.9, 0.1]))
        assert trend.direction is TrendDirection.UNKNOWN
        assert not trend.significant
        assert "need 3 points" in trend.note

    def test_slope_is_per_day_not_per_point(self):
        # Ten points, one a day, dropping 0.05 a day.
        values = [1.0 - 0.05 * i for i in range(10)]
        trend = compute_trend(self._series(values, step_days=1))
        assert trend.slope_per_day == pytest.approx(-0.05, abs=1e-9)
        assert trend.span_days == pytest.approx(9.0)

    def test_a_small_decline_is_not_significant(self):
        trend = compute_trend(self._series([0.85, 0.83, 0.82]))
        assert trend.direction is TrendDirection.DECLINING
        assert trend.change == pytest.approx(-0.03)
        assert not trend.significant

    def test_rising_corrections_count_as_deterioration(self):
        trend = compute_trend(
            self._series([1.0, 3.0, 6.0], target="read_file", metric=USER_CORRECTION)
        )
        assert trend.direction is TrendDirection.RISING
        assert trend.is_deterioration
        # Three points with visible scatter leave one degree of freedom, and
        # the slope clears alpha only just short of it (p = 0.073). The
        # direction is a deterioration; the evidence is not yet conclusive.
        assert trend.p_value == pytest.approx(0.073, abs=0.005)
        assert not trend.significant

    def test_a_sustained_rise_in_corrections_is_significant(self):
        trend = compute_trend(
            self._series(
                [1.0, 2.0, 3.0, 5.0, 6.0, 8.0], target="read_file", metric=USER_CORRECTION
            )
        )
        assert trend.direction is TrendDirection.RISING
        assert trend.is_deterioration
        assert trend.significant

    def test_simultaneous_points_report_change_without_a_slope(self):
        points = [
            point(SKILL_SUCCESS_RATE, "arxiv", value, days_ago=3)
            for value in (0.9, 0.5, 0.4)
        ]
        trend = compute_trend(points)
        assert trend.slope_per_day == pytest.approx(0.0)
        assert trend.change == pytest.approx(-0.5)
        assert trend.direction is TrendDirection.DECLINING

    def test_describe_is_human_readable(self):
        trend = compute_trend(self._series([0.9, 0.8, 0.7]))
        described = trend.describe()
        assert "declining" in described
        assert "/day" in described

    def test_trend_serialises(self):
        blob = compute_trend(self._series([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])).to_dict()
        assert blob["direction"] == "declining"
        assert blob["significant"] is True

    def test_store_trend_reads_from_disk(self, store):
        store.extend(self._series([0.9, 0.8, 0.7]))
        trend = store.trend(SKILL_SUCCESS_RATE, "arxiv", window_days=60, now=T0)
        assert trend.direction is TrendDirection.DECLINING
        assert trend.n == 3


class TestTrendSignificanceIsARealTest:
    """Significance has to be evidence, not a magnitude threshold.

    The two named series here are the regression cases. Under the old rule -
    ``is_deterioration and abs(change) >= 0.05`` - the oscillation reported a
    significant decline and would have fired an optimization run on noise.
    """

    def _series(self, values, target="arxiv", metric=SKILL_SUCCESS_RATE, step_days=7):
        return [
            point(metric, target, value, days_ago=(len(values) - 1 - i) * step_days)
            for i, value in enumerate(values)
        ]

    # The value bounces between roughly 0.9 and 0.3 and lands in the middle.
    # There is no trend; a magnitude rule sees one because the last reading is
    # below the first.
    OSCILLATION = [0.90, 0.35, 0.85, 0.30, 0.88, 0.33, 0.60]

    # Six readings, each below the one before it. Same story every week.
    STEADY_EROSION = [0.91, 0.86, 0.78, 0.71, 0.62, 0.55]

    def test_pure_oscillation_is_not_significant(self):
        trend = compute_trend(self._series(self.OSCILLATION))
        assert trend.direction is TrendDirection.DECLINING
        assert trend.change < -0.05  # the old magnitude rule fired here
        assert trend.p_value == pytest.approx(0.582, abs=0.01)
        assert trend.r_squared == pytest.approx(0.06, abs=0.01)
        assert not trend.statistically_significant
        assert not trend.significant

    def test_steady_erosion_is_significant(self):
        trend = compute_trend(self._series(self.STEADY_EROSION))
        assert trend.direction is TrendDirection.DECLINING
        assert trend.p_value < 0.05
        assert trend.r_squared > 0.95
        assert trend.statistically_significant
        assert trend.practically_significant
        assert trend.significant

    def test_the_oscillation_never_ranks_above_the_erosion(self):
        noisy = compute_trend(self._series(self.OSCILLATION))
        clean = compute_trend(self._series(self.STEADY_EROSION))
        assert clean.p_value < noisy.p_value
        assert clean.r_squared > noisy.r_squared
        assert clean.significant and not noisy.significant

    def test_a_real_but_trivial_drift_is_not_worth_a_cycle(self):
        # A perfect line, so the slope is unarguable, but it loses two and a
        # half points over five weeks. Statistically significant, practically
        # not, and an optimization run costs the same either way.
        trend = compute_trend(self._series([0.900, 0.895, 0.890, 0.885, 0.880, 0.875]))
        assert trend.statistically_significant
        assert not trend.practically_significant
        assert not trend.significant

    def test_alpha_is_configurable(self):
        values = [1.0, 3.0, 6.0]
        strict = compute_trend(self._series(values, metric=USER_CORRECTION))
        lenient = compute_trend(self._series(values, metric=USER_CORRECTION), alpha=0.10)
        assert not strict.significant
        assert lenient.significant
        assert lenient.alpha == 0.10

    def test_the_practical_floor_still_applies(self):
        values = [0.900, 0.895, 0.890, 0.885, 0.880, 0.875]
        assert not compute_trend(self._series(values)).significant
        assert compute_trend(self._series(values), significant_change=0.01).significant

    def test_the_fit_carries_its_uncertainty(self):
        trend = compute_trend(self._series(self.OSCILLATION))
        assert trend.stderr > 0
        assert trend.slope_ci is not None
        assert trend.slope_ci.contains(trend.slope_per_day)
        # The interval straddles zero, which is the same statement as p > alpha.
        assert trend.slope_ci.contains(0.0)
        assert trend.fitted

    def test_describe_states_the_evidence(self):
        described = compute_trend(self._series(self.STEADY_EROSION)).describe()
        assert "p=" in described
        assert "R²=" in described
        assert "CI on slope" in described

    def test_serialisation_carries_the_inference(self):
        blob = compute_trend(self._series(self.OSCILLATION)).to_dict()
        assert blob["significant"] is False
        assert blob["statistically_significant"] is False
        assert blob["p_value"] == pytest.approx(0.582, abs=0.01)
        assert blob["r_squared"] == pytest.approx(0.06, abs=0.01)
        assert blob["slope_ci"]["low"] < 0 < blob["slope_ci"]["high"]
        assert blob["alpha"] == 0.05
        assert blob["practical_threshold"] == 0.05

    def test_too_little_history_reports_no_inference_at_all(self):
        trend = compute_trend(self._series([0.9, 0.4]))
        assert trend.direction is TrendDirection.UNKNOWN
        assert not trend.fitted
        assert trend.confidence_note() == ""
        assert trend.to_dict()["slope_ci"] is None

    def test_a_shared_timestamp_cannot_be_significant(self):
        # Three readings at the same instant: real movement, no time axis to
        # fit it against, so there is nothing to test the slope with.
        points = [
            point(SKILL_SUCCESS_RATE, "arxiv", value, days_ago=3)
            for value in (0.9, 0.5, 0.4)
        ]
        trend = compute_trend(points)
        assert trend.change == pytest.approx(-0.5)
        assert trend.direction is TrendDirection.DECLINING
        assert not trend.significant
        assert not trend.fitted
        assert trend.slope_ci is None
        assert trend.confidence_note() == ""

    def test_an_improving_series_is_never_significant_however_clean(self):
        trend = compute_trend(self._series([0.55, 0.62, 0.71, 0.78, 0.86, 0.91]))
        assert trend.statistically_significant
        assert not trend.is_deterioration
        assert not trend.significant

    def test_confidence_note_is_short_enough_for_a_table(self):
        note = compute_trend(self._series(self.STEADY_EROSION)).confidence_note()
        assert note.startswith("p=")
        assert "R²=" in note
        assert len(note) < 30


class TestArchiving:
    def test_old_points_move_to_the_archive_and_nothing_is_lost(self, store):
        store.extend(
            [
                point(SKILL_SUCCESS_RATE, "arxiv", 0.4, days_ago=200),
                point(SKILL_SUCCESS_RATE, "arxiv", 0.5, days_ago=150),
                point(SKILL_SUCCESS_RATE, "arxiv", 0.6, days_ago=2),
            ]
        )
        moved = store.archive_before(T0 - 100 * DAY)

        assert moved == 2
        assert len(store.load()) == 1
        archived = store.archive_path.read_text().strip().splitlines()
        assert len(archived) == 2

    def test_archiving_nothing_is_a_no_op(self, store):
        store.record(SKILL_SUCCESS_RATE, "arxiv", 0.6, timestamp=T0)
        assert store.archive_before(T0 - 100 * DAY) == 0
        assert not store.archive_path.exists()
        assert len(store.load()) == 1

    def test_archiving_an_empty_store_is_safe(self, store):
        assert store.archive_before(T0) == 0


class TestNothingUnusableGetsIntoTheHistory:
    """The store is the loop's only evidence, so it refuses what would spoil it."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_value_is_refused(self, bad):
        with pytest.raises(ValueError):
            MetricPoint(
                metric=SKILL_SUCCESS_RATE, target="arxiv", value=bad, timestamp=T0
            )

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_timestamp_is_refused(self, bad):
        with pytest.raises(ValueError):
            MetricPoint(
                metric=SKILL_SUCCESS_RATE, target="arxiv", value=1.0, timestamp=bad
            )

    def test_a_nan_never_reaches_the_file_as_bare_json(self, store):
        """json.dumps writes NaN unquoted, which no other JSON reader accepts."""
        with pytest.raises(ValueError):
            store.record(SKILL_SUCCESS_RATE, "arxiv", float("nan"), timestamp=T0)
        assert not store.path.exists() or "NaN" not in store.path.read_text()

    @pytest.mark.parametrize("bad", [1.9, 0.5, -0.5])
    def test_a_fractional_sample_count_is_refused_not_rounded(self, bad):
        with pytest.raises(ValueError):
            MetricPoint(
                metric=SKILL_SUCCESS_RATE,
                target="arxiv",
                value=1.0,
                timestamp=T0,
                samples=bad,
            )

    def test_a_whole_sample_count_still_coerces(self):
        """4.0 and "4" have always been accepted, and still are."""
        for given in (4, 4.0, "4"):
            p = MetricPoint(
                metric=SKILL_SUCCESS_RATE,
                target="arxiv",
                value=1.0,
                timestamp=T0,
                samples=given,
            )
            assert p.samples == 4 and isinstance(p.samples, int)

    def test_a_fractional_stored_count_is_skipped_not_truncated(self, store):
        store.path.parent.mkdir(parents=True, exist_ok=True)
        store.path.write_text(
            json.dumps(
                {
                    "metric": SKILL_SUCCESS_RATE,
                    "target": "arxiv",
                    "value": 0.5,
                    "timestamp": T0,
                    "samples": 1.9,
                }
            )
            + "\n"
        )
        assert store.load() == []
        assert store.skipped_lines == 1

    @pytest.mark.parametrize("record", ["[1, 2, 3]", "null", '"just a string"', "42"])
    def test_a_record_that_is_not_an_object_costs_one_line_not_the_history(
        self, store, record
    ):
        """A valid-JSON non-object used to raise AttributeError past load()."""
        store.extend([point(SKILL_SUCCESS_RATE, "arxiv", 0.9, days_ago=1)])
        with store.path.open("a", encoding="utf-8") as handle:
            handle.write(record + "\n")
        store.extend([point(SKILL_SUCCESS_RATE, "arxiv", 0.8)])

        loaded = store.load()
        assert [p.value for p in loaded] == [0.9, 0.8]
        assert store.skipped_lines == 1


class TestTheAggregateSaysWhichWindowItCovers:
    def test_to_dict_carries_both_ends_of_the_window(self):
        points = [
            point(SKILL_SUCCESS_RATE, "arxiv", 0.4, days_ago=9),
            point(SKILL_SUCCESS_RATE, "arxiv", 0.6, days_ago=1),
        ]
        blob = summarize(SKILL_SUCCESS_RATE, "arxiv", points).to_dict()

        assert blob["first_timestamp"] == T0 - 9 * DAY
        assert blob["last_timestamp"] == T0 - 1 * DAY

    def test_to_dict_still_carries_everything_it_used_to(self):
        blob = summarize(
            SKILL_SUCCESS_RATE, "arxiv", [point(SKILL_SUCCESS_RATE, "arxiv", 0.6)]
        ).to_dict()
        assert set(blob) >= {
            "metric",
            "target",
            "count",
            "samples",
            "mean",
            "weighted_mean",
            "minimum",
            "maximum",
            "total",
            "last_value",
            "sources",
        }

    def test_an_empty_window_serialises_without_timestamps(self):
        blob = summarize(SKILL_SUCCESS_RATE, "arxiv", []).to_dict()
        assert blob["first_timestamp"] is None
        assert blob["last_timestamp"] is None


class TestRotationCannotEatAWrite:
    """archive_before reads, writes elsewhere, then replaces. All three must hold."""

    def test_every_writer_takes_the_lock(self, store, monkeypatch):
        """A lock only one of the writers respects protects nothing."""
        import evolution.monitor.metrics as metrics_module

        real = metrics_module._exclusive
        held = []

        @contextlib.contextmanager
        def recording(path):
            held.append(path)
            with real(path):
                yield

        monkeypatch.setattr(metrics_module, "_exclusive", recording)
        store.append(point(SKILL_SUCCESS_RATE, "arxiv", 0.4, days_ago=200))
        store.extend([point(SKILL_SUCCESS_RATE, "arxiv", 0.6, days_ago=2)])
        store.archive_before(T0 - 100 * DAY)

        assert held == [store.path, store.path, store.path]

    def test_the_lock_is_held_across_the_whole_rotation(self, store, monkeypatch):
        """Not just the replace: the snapshot it is built from has to be covered too."""
        fcntl = pytest.importorskip("fcntl")
        store.extend(
            [
                point(SKILL_SUCCESS_RATE, "arxiv", 0.4, days_ago=200),
                point(SKILL_SUCCESS_RATE, "arxiv", 0.6, days_ago=2),
            ]
        )
        lock_path = store.path.with_name(store.path.name + ".lock")
        real_replace = os.replace
        rival_got_in = []

        def replace_while_a_rival_tries_the_lock(src, dst, *args, **kwargs):
            with lock_path.open("a+b") as rival:
                try:
                    fcntl.flock(rival.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    rival_got_in.append(True)
                except OSError:
                    pass
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(os, "replace", replace_while_a_rival_tries_the_lock)
        store.archive_before(T0 - 100 * DAY)

        assert not rival_got_in, "a second writer could have appended into the snapshot"

    def test_the_writer_lock_is_really_exclusive(self, store):
        """Two descriptions on one lock file must not both hold it."""
        from evolution.monitor.metrics import _exclusive

        fcntl = pytest.importorskip("fcntl")
        store.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = store.path.with_name(store.path.name + ".lock")

        with _exclusive(store.path):
            with lock_path.open("a+b") as rival:
                with pytest.raises(OSError):
                    fcntl.flock(rival.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_a_rotation_that_died_before_the_replace_does_not_double_the_archive(
        self, store, monkeypatch
    ):
        store.extend(
            [
                point(SKILL_SUCCESS_RATE, "arxiv", 0.4, days_ago=200),
                point(SKILL_SUCCESS_RATE, "arxiv", 0.5, days_ago=150),
                point(SKILL_SUCCESS_RATE, "arxiv", 0.6, days_ago=2),
            ]
        )

        def die(*args, **kwargs):
            raise OSError("killed between the archive and the replace")

        monkeypatch.setattr(os, "replace", die)
        with pytest.raises(OSError):
            store.archive_before(T0 - 100 * DAY)
        monkeypatch.undo()

        # The retry the operator runs after the crash.
        assert store.archive_before(T0 - 100 * DAY) == 2
        archived = store.archive_path.read_text().strip().splitlines()
        assert len(archived) == 2
        assert len(store.load()) == 1

    def test_two_identical_observations_both_survive_rotation(self, store):
        """Deduplicating a retry must not deduplicate real repeated measurements."""
        twice = [
            point(SKILL_SUCCESS_RATE, "arxiv", 0.4, days_ago=200),
            point(SKILL_SUCCESS_RATE, "arxiv", 0.4, days_ago=200),
        ]
        store.extend([*twice, point(SKILL_SUCCESS_RATE, "arxiv", 0.6, days_ago=2)])

        assert store.archive_before(T0 - 100 * DAY) == 2
        archived = store.archive_path.read_text().strip().splitlines()
        assert len(archived) == 2
