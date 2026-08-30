"""Tests for the cross-tool regression guard.

The headline case is the one PLAN.md warns about: a candidate whose aggregate
accuracy improves while one tool's selection rate falls. That must be rejected,
and the report must be able to say which tool absorbed the lost selections.

The second half of this file is about the evidence behind that decision. Every
comparison here is paired - baseline and candidate see the identical examples -
so the guard runs an exact McNemar test per tool rather than holding two rates
up against a number, and it says out loud when the sample was too small for the
tolerance it claims to enforce.
"""

import pytest

from evolution.core.gates import GateChain, GateResult, GateStatus
from evolution.core.stats import Interval, PairedBinary
from evolution.tools.cross_tool import (
    ConfusionMatrix,
    CrossToolGuard,
    CrossToolReport,
    ToolRate,
    align_outcomes,
    paired_for_tool,
)
from evolution.tools.selection_eval import NO_TOOL, ToolSelectionExample, score_selection


def outcomes(pairs):
    """Build scored outcomes from ``(expected, predicted)`` pairs."""
    return [
        score_selection(
            ToolSelectionExample(task=f"task {i}", correct_tool=expected), predicted
        )
        for i, (expected, predicted) in enumerate(pairs)
    ]


def report(pairs, tools=None):
    return CrossToolReport.from_outcomes(outcomes(pairs), tools=tools)


def named_outcomes(triples):
    """Scored outcomes from explicit ``(task, expected, predicted)`` triples."""
    return [
        score_selection(ToolSelectionExample(task=task, correct_tool=expected), predicted)
        for task, expected, predicted in triples
    ]


def named_report(triples, tools=None):
    return CrossToolReport.from_outcomes(named_outcomes(triples), tools=tools)


def hits(tool, correct, wrong, thief="terminal"):
    """``correct`` right answers then ``wrong`` misroutes, for one tool."""
    return [(tool, tool)] * correct + [(tool, thief)] * wrong


class TestConfusionMatrix:
    def test_records_and_reads_rows(self):
        matrix = ConfusionMatrix()
        matrix.record("read_file", "read_file")
        matrix.record("read_file", "search_files", 2)
        assert matrix.row("read_file") == {"read_file": 1, "search_files": 2}
        assert matrix.correct("read_file") == 1
        assert matrix.opportunities("read_file") == 3
        assert matrix.total() == 3

    def test_column_shows_who_lost_selections(self):
        matrix = ConfusionMatrix()
        matrix.record("read_file", "search_files", 3)
        matrix.record("terminal", "search_files", 1)
        matrix.record("search_files", "search_files", 5)
        assert matrix.column("search_files") == {"read_file": 3, "terminal": 1}

    def test_misroutes_are_ordered_worst_first(self):
        matrix = ConfusionMatrix()
        matrix.record("read_file", "read_file", 4)
        matrix.record("read_file", "terminal", 1)
        matrix.record("read_file", "search_files", 3)
        assert matrix.misroutes("read_file") == [("search_files", 3), ("terminal", 1)]

    def test_top_confusions_excludes_the_diagonal(self):
        matrix = ConfusionMatrix()
        matrix.record("a", "a", 10)
        matrix.record("a", "b", 2)
        matrix.record("b", "c", 3)
        assert matrix.top_confusions() == [("b", "c", 3), ("a", "b", 2)]

    def test_top_confusions_respects_the_limit(self):
        matrix = ConfusionMatrix()
        for i in range(5):
            matrix.record(f"t{i}", "thief", i + 1)
        assert len(matrix.top_confusions(limit=2)) == 2

    def test_unknown_row_is_empty_not_an_error(self):
        assert ConfusionMatrix().row("ghost") == {}
        assert ConfusionMatrix().opportunities("ghost") == 0

    def test_serialises(self):
        matrix = ConfusionMatrix()
        matrix.record("a", "b")
        assert matrix.to_dict() == {"a": {"b": 1}}


class TestCrossToolReport:
    def test_rates_per_tool(self):
        built = report(
            [
                ("read_file", "read_file"),
                ("read_file", "search_files"),
                ("search_files", "search_files"),
                ("search_files", "search_files"),
            ]
        )
        assert built.rate("read_file") == 0.5
        assert built.rate("search_files") == 1.0
        assert built.overall_accuracy == 0.75

    def test_confusion_matrix_is_populated(self):
        built = report([("read_file", "search_files"), ("read_file", "read_file")])
        assert built.confusion.row("read_file") == {"search_files": 1, "read_file": 1}

    def test_unmeasured_tools_are_seeded_at_zero(self):
        built = report([("read_file", "read_file")], tools=["read_file", "terminal"])
        assert built.opportunities("terminal") == 0
        assert built.rate("terminal") == 0.0
        assert built.measured_tools == ["read_file"]

    def test_no_tool_is_always_a_row(self):
        assert NO_TOOL in report([("read_file", "read_file")]).rates

    def test_no_tool_is_scored_like_any_other_tool(self):
        built = report([(NO_TOOL, NO_TOOL), (NO_TOOL, "read_file")])
        assert built.rate(NO_TOOL) == 0.5
        assert built.confusion.row(NO_TOOL)["read_file"] == 1

    def test_unknown_tool_reads_as_zero(self):
        built = report([("read_file", "read_file")])
        assert built.rate("ghost") == 0.0
        assert built.opportunities("ghost") == 0

    def test_empty_report(self):
        built = report([])
        assert built.n == 0 and built.overall_accuracy == 0.0

    def test_param_accuracy_flows_through(self):
        example = ToolSelectionExample(
            task="t", correct_tool="read_file", correct_params={"path": "a.py"}
        )
        built = CrossToolReport.from_outcomes(
            [score_selection(example, "read_file", {"path": "b.py"})]
        )
        assert built.param_accuracy == 0.0
        assert built.overall_accuracy == 1.0

    def test_serialises(self):
        blob = report([("read_file", "read_file")]).to_dict()
        assert blob["rates"]["read_file"]["rate"] == 1.0
        assert blob["confusion"]["read_file"] == {"read_file": 1}


class TestGuardRejectsStealing:
    """The case PLAN.md calls out: aggregate up, one tool down."""

    def baseline(self):
        # read_file 4/4, search_files 1/4. Overall 5/8.
        return report(
            [("read_file", "read_file")] * 4
            + [("search_files", "search_files")]
            + [("search_files", "read_file")] * 3,
            tools=["read_file", "search_files"],
        )

    def candidate(self):
        # search_files climbs to 4/4, read_file drops to 2/4. Overall 6/8.
        return report(
            [("read_file", "read_file")] * 2
            + [("read_file", "search_files")] * 2
            + [("search_files", "search_files")] * 4,
            tools=["read_file", "search_files"],
        )

    def test_the_aggregate_really_did_improve(self):
        assert self.candidate().overall_accuracy > self.baseline().overall_accuracy

    def test_candidate_is_rejected_anyway(self):
        verdict = CrossToolGuard().compare(self.baseline(), self.candidate())
        assert verdict.accepted is False
        assert [r.tool for r in verdict.regressions] == ["read_file"]

    def test_reason_admits_the_aggregate_improved(self):
        verdict = CrossToolGuard().compare(self.baseline(), self.candidate())
        assert "despite the aggregate improving" in verdict.reason

    def test_the_thief_is_named(self):
        verdict = CrossToolGuard().compare(self.baseline(), self.candidate())
        assert verdict.regressions[0].stolen_by == {"search_files": 2}
        assert "lost to search_files +2" in verdict.regressions[0].describe()

    def test_the_improvement_is_still_recorded(self):
        verdict = CrossToolGuard().compare(self.baseline(), self.candidate())
        assert [i.tool for i in verdict.improvements] == ["search_files"]

    def test_verdict_becomes_a_blocking_gate(self):
        gate = CrossToolGuard().gate(self.baseline(), self.candidate())
        assert isinstance(gate, GateResult)
        assert gate.status is GateStatus.FAILED
        assert gate.blocking and not gate.passed
        assert "read_file" in gate.details

    def test_gate_chain_stops_on_the_rejection(self):
        guard = CrossToolGuard()
        later_gate_ran = []

        def later():
            later_gate_ran.append(True)
            return GateResult("later", GateStatus.PASSED, "ok")

        chain = GateChain().run(lambda: guard.gate(self.baseline(), self.candidate()), later)
        assert chain.passed is False
        assert later_gate_ran == []
        assert [r.name for r in chain.blockers] == ["cross_tool"]


class TestGuardAcceptance:
    def test_identical_reports_are_accepted(self):
        built = report([("read_file", "read_file"), ("terminal", "read_file")])
        verdict = CrossToolGuard().compare(built, built)
        assert verdict.accepted
        assert verdict.regressions == []
        assert verdict.overall_delta == 0.0

    def test_a_pure_improvement_is_accepted(self):
        baseline = report([("read_file", "terminal"), ("read_file", "read_file")])
        candidate = report([("read_file", "read_file"), ("read_file", "read_file")])
        verdict = CrossToolGuard().compare(baseline, candidate)
        assert verdict.accepted
        assert verdict.improvements[0].delta == pytest.approx(0.5)

    def test_holding_steady_while_shrinking_text_is_allowed(self):
        built = report([("read_file", "read_file")] * 4)
        assert CrossToolGuard().compare(built, built).accepted

    def test_require_overall_improvement_rejects_a_flat_candidate(self):
        built = report([("read_file", "read_file")] * 4)
        verdict = CrossToolGuard(require_overall_improvement=True).compare(built, built)
        assert verdict.accepted is False
        assert "did not improve" in verdict.reason
        assert verdict.regressions == []


class TestTolerance:
    def baseline(self):
        return report([("read_file", "read_file")] * 4, tools=["read_file"])

    def candidate(self):
        # 3/4: a 25 point drop.
        return report(
            [("read_file", "read_file")] * 3 + [("read_file", "terminal")],
            tools=["read_file"],
        )

    def test_zero_tolerance_is_the_default(self):
        assert CrossToolGuard().tolerance == 0.0
        assert not CrossToolGuard().compare(self.baseline(), self.candidate()).accepted

    def test_a_configured_tolerance_absorbs_a_small_drop(self):
        assert CrossToolGuard(tolerance=0.3).compare(self.baseline(), self.candidate()).accepted

    def test_a_drop_exactly_at_tolerance_is_accepted(self):
        assert CrossToolGuard(tolerance=0.25).compare(self.baseline(), self.candidate()).accepted

    def test_a_drop_just_past_tolerance_is_rejected(self):
        guard = CrossToolGuard(tolerance=0.2499)
        assert not guard.compare(self.baseline(), self.candidate()).accepted

    def test_tolerance_is_reported_in_the_reason(self):
        verdict = CrossToolGuard(tolerance=0.1).compare(self.baseline(), self.baseline())
        assert "10.0% tolerance" in verdict.reason


class TestMinOpportunities:
    def test_thin_tools_are_ignored_and_listed(self):
        baseline = report(
            [("read_file", "read_file")] * 4 + [("terminal", "terminal")],
            tools=["read_file", "terminal"],
        )
        candidate = report(
            [("read_file", "read_file")] * 4 + [("terminal", "read_file")],
            tools=["read_file", "terminal"],
        )
        guard = CrossToolGuard(min_opportunities=3)
        verdict = guard.compare(baseline, candidate)
        assert verdict.accepted
        assert "terminal (1 example(s))" in verdict.ignored

    def test_the_same_drop_is_caught_at_the_default_threshold(self):
        baseline = report(
            [("read_file", "read_file")] * 4 + [("terminal", "terminal")],
            tools=["read_file", "terminal"],
        )
        candidate = report(
            [("read_file", "read_file")] * 4 + [("terminal", "read_file")],
            tools=["read_file", "terminal"],
        )
        assert not CrossToolGuard().compare(baseline, candidate).accepted

    def test_tools_with_no_examples_are_listed_plainly(self):
        built = report([("read_file", "read_file")], tools=["read_file", "ghost"])
        verdict = CrossToolGuard().compare(built, built)
        assert "ghost" in verdict.ignored
        assert NO_TOOL in verdict.ignored


class TestVerdictReporting:
    def test_summary_reads_as_a_sentence(self):
        baseline = report([("read_file", "read_file")] * 2)
        candidate = report([("read_file", "terminal")] * 2)
        verdict = CrossToolGuard().compare(baseline, candidate)
        assert verdict.summary().startswith("cross-tool REJECTED")
        assert "100.0% -> 0.0%" in verdict.summary()

    def test_accepted_summary(self):
        built = report([("read_file", "read_file")])
        assert CrossToolGuard().compare(built, built).summary().startswith("cross-tool accepted")

    def test_serialises_the_evidence(self):
        baseline = report([("read_file", "read_file")] * 2)
        candidate = report([("read_file", "terminal")] * 2)
        blob = CrossToolGuard().compare(baseline, candidate).to_dict()
        assert blob["accepted"] is False
        assert blob["regressions"][0]["tool"] == "read_file"
        assert blob["regressions"][0]["delta"] == -1.0
        assert blob["overall_delta"] == -1.0

    def test_passing_verdict_is_a_passing_gate(self):
        built = report([("read_file", "read_file")])
        gate = CrossToolGuard().compare(built, built).to_gate_result()
        assert gate.passed and gate.name == "cross_tool"
        assert gate.baseline == 1.0 and gate.score == 1.0

    def test_regression_describes_itself(self):
        baseline = report([("read_file", "read_file")] * 4, tools=["read_file"])
        candidate = report(
            [("read_file", "read_file")] * 2 + [("read_file", "terminal")] * 2,
            tools=["read_file"],
        )
        regression = CrossToolGuard().compare(baseline, candidate).regressions[0]
        assert regression.describe().startswith("read_file: 100.0% -> 50.0% (-50.0%")
        assert regression.to_dict()["opportunities"] == 4

    def test_a_tool_that_vanishes_from_the_candidate_counts_as_zero(self):
        baseline = report([("read_file", "read_file")] * 2, tools=["read_file"])
        candidate = CrossToolReport()
        verdict = CrossToolGuard().compare(baseline, candidate)
        assert not verdict.accepted
        assert verdict.regressions[0].candidate_rate == 0.0


# ──────────────────────────────────────────────────────────────────────────
# Paired evidence
# ──────────────────────────────────────────────────────────────────────────


class TestOutcomeVectors:
    """Counts cannot be paired. The per-example record is the whole point."""

    def test_the_report_keeps_one_outcome_per_example(self):
        built = report(
            [("read_file", "read_file"), ("read_file", "terminal"), ("read_file", "read_file")]
        )
        assert built.outcome_vector("read_file") == (True, False, True)

    def test_the_vector_is_in_dataset_order(self):
        built = named_report(
            [("b", "read_file", "terminal"), ("a", "read_file", "read_file")]
        )
        assert built.outcome_vector("read_file") == (False, True)
        assert built.example_keys("read_file") == ("b", "a")

    def test_repeated_task_text_still_gets_distinct_keys(self):
        built = named_report([("same", "read_file", "read_file")] * 3)
        assert built.example_keys("read_file") == ("same", "same#2", "same#3")

    def test_counts_and_the_vector_agree(self):
        built = report(hits("read_file", 3, 1))
        rate = built.rates["read_file"]
        assert rate.opportunities == len(rate.outcomes) == 4
        assert rate.correct == sum(rate.outcomes) == 3
        assert rate.paired_ready

    def test_a_seeded_tool_with_no_examples_has_no_vector(self):
        built = report([("read_file", "read_file")], tools=["read_file", "ghost"])
        assert built.outcome_vector("ghost") == ()
        assert built.rates["ghost"].paired_ready is False

    def test_the_vector_is_serialised_so_p_values_can_be_recomputed(self):
        blob = report(hits("read_file", 2, 1)).to_dict()
        assert blob["rates"]["read_file"]["outcomes"] == [1, 1, 0]


class TestAlignment:
    def test_matching_reports_align_positionally(self):
        baseline = report(hits("read_file", 4, 0))
        candidate = report(hits("read_file", 2, 2))
        assert align_outcomes(baseline, candidate, "read_file") == (
            [True, True, True, True],
            [True, True, False, False],
        )

    def test_a_shuffled_candidate_still_pairs_by_example(self):
        baseline = named_report(
            [("a", "read_file", "read_file"), ("b", "read_file", "read_file")]
        )
        candidate = named_report(
            [("b", "read_file", "terminal"), ("a", "read_file", "read_file")]
        )
        assert align_outcomes(baseline, candidate, "read_file") == (
            [True, True],
            [True, False],
        )

    def test_only_the_shared_examples_are_paired(self):
        baseline = named_report(
            [("a", "read_file", "read_file"), ("b", "read_file", "read_file")]
        )
        candidate = named_report([("a", "read_file", "terminal")])
        assert align_outcomes(baseline, candidate, "read_file") == ([True], [False])

    def test_no_shared_example_means_no_pairing(self):
        baseline = named_report([("a", "read_file", "read_file")])
        candidate = named_report([("z", "read_file", "read_file")])
        assert align_outcomes(baseline, candidate, "read_file") is None

    def test_a_tool_absent_from_one_side_cannot_be_paired(self):
        baseline = report(hits("read_file", 2, 0))
        assert align_outcomes(baseline, CrossToolReport(), "read_file") is None

    def test_hand_built_vectors_pair_positionally(self):
        baseline = CrossToolReport(
            rates={"read_file": ToolRate("read_file", 3, 3, (True, True, True))}
        )
        candidate = CrossToolReport(
            rates={"read_file": ToolRate("read_file", 3, 1, (True, False, False))}
        )
        assert align_outcomes(baseline, candidate, "read_file") == (
            [True, True, True],
            [True, False, False],
        )

    def test_hand_built_vectors_of_unequal_length_are_a_misalignment(self):
        baseline = CrossToolReport(rates={"t": ToolRate("t", 3, 3, (True, True, True))})
        candidate = CrossToolReport(rates={"t": ToolRate("t", 2, 2, (True, True))})
        assert align_outcomes(baseline, candidate, "t") is None

    def test_paired_for_tool_builds_the_test(self):
        baseline = report(hits("read_file", 4, 0))
        candidate = report(hits("read_file", 2, 2))
        paired = paired_for_tool(baseline, candidate, "read_file")
        assert isinstance(paired, PairedBinary)
        assert (paired.n, paired.baseline_only, paired.candidate_only) == (4, 2, 0)

    def test_paired_for_tool_returns_none_when_it_cannot_pair(self):
        assert paired_for_tool(CrossToolReport(), CrossToolReport(), "ghost") is None


class TestSignificanceChangesTheVerdict:
    """A tolerance is a magnitude. Significance is evidence. The gate wants both."""

    def baseline(self, n=40):
        return report(hits("read_file", n, 0), tools=["read_file"])

    def candidate(self, n=40, wrong=5):
        return report(hits("read_file", n - wrong, wrong), tools=["read_file"])

    def test_a_significant_regression_inside_the_tolerance_is_rejected(self):
        """5 of 40 flipped one way: -12.5% is under a 15% tolerance, p = 0.031."""
        verdict = CrossToolGuard(tolerance=0.15).compare(self.baseline(), self.candidate())
        regression = verdict.regressions[0]
        assert verdict.accepted is False
        assert regression.breaches_tolerance is False
        assert regression.significant_regression is True
        assert regression.p_worse == pytest.approx(0.03125)

    def test_the_reason_names_the_significant_tools(self):
        verdict = CrossToolGuard(tolerance=0.15).compare(self.baseline(), self.candidate())
        assert "significant at alpha=0.05 (read_file)" in verdict.reason
        assert [r.tool for r in verdict.significant_regressions] == ["read_file"]

    def test_a_large_drop_that_is_not_significant_is_still_rejected(self):
        """The point estimate rule stands on its own: statistics only add rejections."""
        baseline = report(hits("read_file", 4, 0), tools=["read_file"])
        candidate = report(hits("read_file", 2, 2), tools=["read_file"])
        verdict = CrossToolGuard().compare(baseline, candidate)
        regression = verdict.regressions[0]
        assert verdict.accepted is False
        assert regression.breaches_tolerance is True
        assert regression.significant_regression is False

    def test_the_same_ten_point_drop_reads_differently_at_two_sample_sizes(self):
        """-10% over 40 examples is p=0.062; over 10 it is p=0.500."""
        wide = CrossToolGuard(tolerance=0.2).compare(
            report(hits("read_file", 40, 0), tools=["read_file"]),
            report(hits("read_file", 36, 4), tools=["read_file"]),
        )
        narrow = CrossToolGuard(tolerance=0.2).compare(
            report(hits("read_file", 10, 0), tools=["read_file"]),
            report(hits("read_file", 9, 1), tools=["read_file"]),
        )
        assert wide.comparison("read_file").delta == pytest.approx(-0.1)
        assert narrow.comparison("read_file").delta == pytest.approx(-0.1)
        assert wide.comparison("read_file").p_worse == pytest.approx(0.0625)
        assert narrow.comparison("read_file").p_worse == pytest.approx(0.5)

    def test_disagreement_in_both_directions_is_not_a_regression(self):
        """Three flips each way is noise, and a paired test says so."""
        baseline = named_report(
            [(f"t{i}", "read_file", "read_file" if i < 6 else "terminal") for i in range(12)],
            tools=["read_file"],
        )
        candidate = named_report(
            [
                (f"t{i}", "read_file", "terminal" if i < 3 else "read_file")
                for i in range(12)
            ][:6]
            + [
                (f"t{i}", "read_file", "read_file" if i < 9 else "terminal")
                for i in range(6, 12)
            ],
            tools=["read_file"],
        )
        comparison = CrossToolGuard().compare(baseline, candidate).comparison("read_file")
        assert comparison.delta == pytest.approx(0.0)
        assert comparison.p_worse == pytest.approx(0.65625)
        assert comparison.regressed is False

    def test_a_significant_improvement_is_labelled(self):
        baseline = report(hits("read_file", 30, 10), tools=["read_file"])
        candidate = report(hits("read_file", 40, 0), tools=["read_file"])
        verdict = CrossToolGuard().compare(baseline, candidate)
        assert verdict.improvements[0].significant_improvement is True

    def test_alpha_is_configurable(self):
        strict = CrossToolGuard(tolerance=0.15, alpha=0.01)
        assert strict.compare(self.baseline(), self.candidate()).accepted is True


class TestNoMultiplicityCorrection:
    """Accepting means 'no tool regressed', an intersection-union claim.

    Each per-tool test runs at alpha with no Bonferroni. Correcting would raise
    every tool's bar as the catalogue grows, making a safety gate looser the
    more tools it has to protect.
    """

    def pair(self, tool_count):
        clean = []
        for index in range(tool_count):
            clean.extend(hits(f"tool{index}", 8, 0))
        baseline = report(clean + hits("read_file", 40, 0))
        candidate = report(clean + hits("read_file", 35, 5))
        return baseline, candidate

    def test_one_tool_regressing_rejects_a_small_catalogue(self):
        baseline, candidate = self.pair(1)
        assert CrossToolGuard(tolerance=0.15).compare(baseline, candidate).accepted is False

    def test_the_same_regression_still_rejects_a_large_catalogue(self):
        baseline, candidate = self.pair(20)
        verdict = CrossToolGuard(tolerance=0.15).compare(baseline, candidate)
        assert verdict.accepted is False
        assert [r.tool for r in verdict.significant_regressions] == ["read_file"]

    def test_the_p_value_threshold_does_not_move_with_tool_count(self):
        small = CrossToolGuard(tolerance=0.15).compare(*self.pair(1))
        large = CrossToolGuard(tolerance=0.15).compare(*self.pair(20))
        assert small.comparison("read_file").p_worse == large.comparison("read_file").p_worse


class TestPowerIsReported:
    def small(self):
        return (
            report(hits("read_file", 10, 0), tools=["read_file"]),
            report(hits("read_file", 10, 0), tools=["read_file"]),
        )

    def test_a_tolerance_the_sample_cannot_detect_is_flagged(self):
        verdict = CrossToolGuard(tolerance=0.05).compare(*self.small())
        assert verdict.accepted
        assert verdict.underpowered == ["read_file"]
        assert verdict.comparison("read_file").min_detectable_shift == pytest.approx(0.5)

    def test_the_summary_says_the_tolerance_is_not_enforced_by_evidence(self):
        verdict = CrossToolGuard(tolerance=0.05).compare(*self.small())
        assert "not enforced by evidence" in verdict.summary()
        assert "5.0% tolerance" in verdict.power_note()

    def test_a_sample_large_enough_is_not_flagged(self):
        baseline = report(hits("read_file", 100, 0), tools=["read_file"])
        verdict = CrossToolGuard(tolerance=0.05).compare(baseline, baseline)
        assert verdict.underpowered == []
        assert verdict.power_note() == ""

    def test_a_zero_tolerance_needs_no_power_warning(self):
        """Any drop at all breaches zero, so the gate is already as strict as it gets."""
        verdict = CrossToolGuard(tolerance=0.0).compare(*self.small())
        assert verdict.underpowered == []
        assert "not enforced by evidence" not in verdict.summary()

    def test_the_power_note_reaches_the_gate_details(self):
        gate = CrossToolGuard(tolerance=0.05).gate(*self.small())
        assert gate.passed
        assert "not enforced by evidence" in gate.details

    def test_an_underpowered_regression_says_both_things(self):
        baseline = report(hits("read_file", 10, 0), tools=["read_file"])
        candidate = report(hits("read_file", 8, 2), tools=["read_file"])
        regression = CrossToolGuard(tolerance=0.05).compare(baseline, candidate).regressions[0]
        assert regression.breaches_tolerance is True
        assert regression.underpowered is True
        assert "underpowered" in regression.describe()

    def test_a_tool_that_cannot_be_paired_is_listed(self):
        baseline = report(hits("read_file", 2, 0), tools=["read_file"])
        verdict = CrossToolGuard().compare(baseline, CrossToolReport())
        assert verdict.unpaired == ["read_file"]
        assert "no paired evidence" in verdict.summary()
        assert verdict.regressions[0].p_worse is None
        assert verdict.regressions[0].delta_interval() is None


class TestIntervalsInTheRecords:
    def test_a_regression_carries_its_interval_and_p_value(self):
        baseline = report(hits("read_file", 40, 0), tools=["read_file"])
        candidate = report(hits("read_file", 35, 5), tools=["read_file"])
        regression = CrossToolGuard().compare(baseline, candidate).regressions[0]
        interval = regression.delta_interval()
        assert isinstance(interval, Interval)
        assert interval.contains(regression.delta)
        assert interval.high < 0  # a one-sided story: the whole interval is a loss
        assert "95% CI" in regression.describe() and "p=0.031" in regression.describe()

    def test_an_improvement_carries_the_same_evidence(self):
        baseline = report(hits("read_file", 30, 10), tools=["read_file"])
        candidate = report(hits("read_file", 40, 0), tools=["read_file"])
        improvement = CrossToolGuard().compare(baseline, candidate).improvements[0]
        assert improvement.delta_interval().low > 0
        assert "95% CI" in improvement.describe()

    def test_every_measured_tool_gets_a_comparison(self):
        built = report(
            hits("read_file", 4, 0) + hits("search_files", 3, 1),
            tools=["read_file", "search_files", "ghost"],
        )
        verdict = CrossToolGuard().compare(built, built)
        assert [c.tool for c in verdict.comparisons] == ["read_file", "search_files"]
        assert verdict.comparison("ghost") is None

    def test_comparisons_serialise_with_their_statistics(self):
        baseline = report(hits("read_file", 40, 0), tools=["read_file"])
        candidate = report(hits("read_file", 35, 5), tools=["read_file"])
        blob = CrossToolGuard(tolerance=0.15).compare(baseline, candidate).to_dict()
        row = blob["comparisons"][0]
        assert row["tool"] == "read_file"
        assert row["baseline_rate"] == 1.0 and row["candidate_rate"] == 0.875
        assert row["delta"] == -0.125
        # When every disagreement points the same way the point estimate sits on
        # the interval's edge rather than inside it, so the bound is inclusive.
        assert row["delta_ci"]["low"] <= row["delta"] <= row["delta_ci"]["high"]
        assert row["delta_ci"]["high"] > row["delta_ci"]["low"]
        assert row["p_worse"] == pytest.approx(0.03125)
        assert row["significant_regression"] is True
        assert row["underpowered"] is False
        assert blob["underpowered"] == [] and blob["unpaired"] == []

    def test_an_unpaired_record_serialises_without_inventing_a_p_value(self):
        baseline = report(hits("read_file", 2, 0), tools=["read_file"])
        blob = CrossToolGuard().compare(baseline, CrossToolReport()).to_dict()
        assert blob["regressions"][0]["p_worse"] is None
        assert blob["regressions"][0]["delta_ci"] is None
        assert blob["regressions"][0]["paired"] is None


class TestAccuracyAgainstChance:
    """40% is poor against two tools and excellent against thirty."""

    def test_chance_counts_every_option_including_no_tool(self):
        built = report(
            [("read_file", "read_file")], tools=["read_file", "search_files", "terminal"]
        )
        assert built.num_options == 4
        assert built.chance_accuracy == pytest.approx(0.25)

    def test_the_accuracy_interval_is_a_wilson_interval(self):
        built = report(hits("read_file", 8, 0))
        interval = built.accuracy_interval()
        assert interval.point == 1.0
        assert interval.low == pytest.approx(0.6756, abs=1e-4)
        assert interval.high == 1.0

    def test_a_small_sample_gets_a_wide_interval(self):
        assert report(hits("read_file", 2, 0)).accuracy_interval().low < 0.35

    def test_correct_counts_across_every_tool(self):
        built = report(hits("read_file", 3, 1) + hits("search_files", 1, 1))
        assert built.correct == 4 and built.n == 6

    def test_describe_accuracy_reads_as_a_sentence(self):
        text = report(hits("read_file", 8, 0), tools=["read_file", "terminal"]).describe_accuracy()
        assert "100.0% [67.6%, 100.0%]" in text
        assert "vs 33.3% chance across 3 option(s)" in text

    def test_the_report_serialises_the_baseline_and_the_interval(self):
        blob = report(hits("read_file", 8, 0), tools=["read_file"]).to_dict()
        assert blob["num_options"] == 2
        assert blob["chance_accuracy"] == 0.5
        assert blob["correct"] == 8
        assert blob["accuracy_interval"]["low"] == pytest.approx(0.6756, abs=1e-4)

    def test_the_verdict_carries_both_intervals_and_the_chance_line(self):
        baseline = report(hits("read_file", 8, 0), tools=["read_file"])
        candidate = report(hits("read_file", 6, 2), tools=["read_file"])
        verdict = CrossToolGuard().compare(baseline, candidate)
        assert verdict.baseline_interval.point == 1.0
        assert verdict.candidate_interval.point == 0.75
        assert verdict.chance_accuracy == pytest.approx(0.5)
        assert "vs 50.0% chance across 2 option(s)" in verdict.accuracy_note()

    def test_the_verdict_serialises_them(self):
        built = report(hits("read_file", 4, 0), tools=["read_file"])
        blob = CrossToolGuard().compare(built, built).to_dict()
        assert blob["chance_accuracy"] == 0.5 and blob["num_options"] == 2
        assert blob["baseline_accuracy_interval"]["point"] == 1.0
        assert blob["candidate_accuracy_interval"]["point"] == 1.0

    def test_an_empty_report_does_not_divide_by_zero(self):
        built = CrossToolReport()
        assert built.chance_accuracy == 0.0
        assert built.accuracy_interval().point == 0.0
        assert "0 example(s)" in built.describe_accuracy()


class TestDeltaMatchesItsEvidence:
    """The delta a verdict rejects on must describe the examples it tested.

    Audit finding: with a partial candidate run, the whole-report rates and the
    paired comparison covered different populations, and the headline delta
    could land outside its own confidence interval.
    """

    def _rate(self, tool, outcomes, keys):
        return ToolRate(
            tool=tool,
            opportunities=len(outcomes),
            correct=sum(1 for o in outcomes if o),
            outcomes=tuple(outcomes),
            example_keys=tuple(keys),
        )

    def test_delta_comes_from_the_pairing_not_the_whole_report(self):
        keys = [f"task{i}" for i in range(20)]
        baseline = CrossToolReport(
            rates={"read_file": self._rate("read_file", [True] * 10 + [False] * 10, keys)},
            n=20,
        )
        # The candidate only managed two of the twenty, and got both right.
        candidate = CrossToolReport(
            rates={"read_file": self._rate("read_file", [True, True], keys[:2])},
            n=2,
        )
        verdict = CrossToolGuard(min_opportunities=1).compare(baseline, candidate)
        record = verdict.comparison("read_file")

        assert record is not None
        assert record.paired is not None
        assert record.paired.n == 2
        assert record.delta == pytest.approx(record.paired.delta)
        assert record.population_mismatch
        # The unpaired view is still available, just never gated on.
        assert record.unpaired_delta != record.delta

    def test_the_delta_sits_inside_its_own_interval(self):
        keys = [f"task{i}" for i in range(20)]
        baseline = CrossToolReport(
            rates={"read_file": self._rate("read_file", [True] * 10 + [False] * 10, keys)},
            n=20,
        )
        candidate = CrossToolReport(
            rates={"read_file": self._rate("read_file", [True, True], keys[:2])},
            n=2,
        )
        record = CrossToolGuard(min_opportunities=1).compare(
            baseline, candidate
        ).comparison("read_file")
        interval = record.paired.delta_interval()
        assert interval.low <= record.delta <= interval.high
