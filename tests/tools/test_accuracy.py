"""Tests for the factual-accuracy constraint.

Two things matter equally here. The checker has to catch a description that
promises a parameter, a value or a capability the frozen schema cannot deliver,
and it has to stay silent on the descriptions hermes-agent actually ships -
which quote other tools, name enum values, and document conditional
requirements JSON Schema cannot express. A checker that fires on honest text
gets switched off, and then it catches nothing at all.

Nothing here reaches a model. The entailment tier is exercised with injected
stubs, including one that raises, because the tier has to degrade rather than
take the run down with it.
"""

import dspy
import pytest

from evolution.tools.accuracy import (
    KIND_ENTAILMENT,
    KIND_ENUM_CONTRADICTION,
    KIND_REQUIREDNESS,
    KIND_UNKNOWN_PARAMETER,
    KIND_UNSUPPORTED_CAPABILITY,
    AccuracyFinding,
    AccuracyReport,
    FactualAccuracyChecker,
    ToolSchemaFacts,
    facts_from_catalog,
)
from evolution.tools.tool_catalog import ToolDescriptions, load_catalog


@pytest.fixture
def facts(hermes_repo):
    return facts_from_catalog(load_catalog(hermes_repo))


@pytest.fixture
def checker(facts):
    return FactualAccuracyChecker(facts=facts)


def kinds(findings):
    return sorted(f.kind for f in findings)


class TestSchemaFacts:
    def test_parameters_types_and_required_come_from_the_schema(self, facts):
        read_file = facts["read_file"]
        assert read_file.params == ("path", "offset", "limit")
        assert read_file.required == ("path",)
        assert read_file.types["offset"] == "integer"

    def test_enums_are_carried(self, facts):
        assert facts["search_files"].enums == {"target": ("content", "files")}
        assert facts["search_files"].enum_values == {"content", "files"}

    def test_a_tool_without_enums_has_none(self, facts):
        assert facts["read_file"].enums == {}

    def test_every_tool_knows_the_other_tool_names(self, facts):
        assert "search_files" in facts["read_file"].known_tools

    def test_render_shows_the_frozen_half(self, facts):
        rendered = facts["search_files"].render()
        assert "pattern: string; required" in rendered
        assert "one of content, files" in rendered

    def test_render_says_so_when_there_are_no_parameters(self):
        assert "(no parameters)" in ToolSchemaFacts(tool_name="ghost").render()


class TestRealDescriptionsAreLeftAlone:
    """The fixture repo mirrors the real schemas. None of it may be flagged."""

    def test_the_shipped_catalogue_is_clean(self, hermes_repo, checker):
        catalog = load_catalog(hermes_repo)
        for entry in catalog:
            assert checker.check_tool(entry.tool_name, entry.description) == []
            for param, text in entry.param_descriptions.items():
                assert checker.check_param(entry.tool_name, param, text) == []

    def test_quoting_another_tool_is_not_a_parameter_claim(self, checker):
        found = checker.check_tool(
            "write_file", "Write a file, replacing it. Use 'patch' for targeted edits."
        )
        assert found == []

    def test_quoting_an_enum_value_is_not_a_parameter_claim(self, checker):
        found = checker.check_tool(
            "search_files", "Search inside files with 'content', or find names with 'files'."
        )
        assert found == []

    def test_prose_in_quotes_is_not_a_parameter_claim(self, checker):
        found = checker.check_tool("read_file", "Reads a file and prints 'LINE|TEXT' rows.")
        assert found == []

    def test_naming_real_parameters_is_fine(self, checker):
        found = checker.check_tool(
            "read_file", "Read a file. Use the `offset` and `limit` parameters for big files."
        )
        assert found == []

    def test_a_conditional_requirement_is_not_a_contradiction(self, checker):
        """The real patch tool documents 'REQUIRED when mode=replace' on an optional field."""
        found = checker.check_param(
            "write_file", "cross_profile", "REQUIRED when writing to another profile."
        )
        assert found == []

    def test_a_capability_the_schema_does_carry_is_fine(self, checker):
        found = checker.check_tool(
            "terminal", "Run a command. Set a timeout in seconds to bound the wait."
        )
        assert found == []

    def test_an_unknown_tool_is_not_judged(self, checker):
        assert checker.check_tool("ghost", "Anything at all, `nonsense` included.") == []
        assert checker.check_param("ghost", "x", "`nonsense`") == []


class TestUnknownParameters:
    def test_a_backticked_invention_is_caught(self, checker):
        found = checker.check_tool("read_file", "Read a file. Pass `recursive` to descend.")
        assert kinds(found) == [KIND_UNKNOWN_PARAMETER]
        assert "recursive" in found[0].message
        assert "path, offset, limit" in found[0].message

    def test_snake_case_alone_is_enough_to_qualify(self, checker):
        found = checker.check_tool("read_file", "Read a file, honouring `max_bytes`.")
        assert kinds(found) == [KIND_UNKNOWN_PARAMETER]

    def test_a_quoted_word_beside_the_word_parameter_qualifies(self, checker):
        found = checker.check_tool("read_file", "Read a file. The 'tail' parameter reads the end.")
        assert kinds(found) == [KIND_UNKNOWN_PARAMETER]

    def test_the_same_invention_is_reported_once(self, checker):
        found = checker.check_tool(
            "read_file", "Pass `recursive` to descend. Set `recursive` for trees."
        )
        assert len(found) == 1

    def test_the_finding_carries_its_evidence(self, checker):
        found = checker.check_tool("read_file", "Read a file. Pass `recursive` to descend.")
        assert "recursive" in found[0].evidence
        assert found[0].describe().startswith("read_file: names 'recursive'")

    def test_a_parameter_description_is_checked_too(self, checker):
        found = checker.check_param("read_file", "limit", "How many lines; pair with `stride`.")
        assert kinds(found) == [KIND_UNKNOWN_PARAMETER]
        assert found[0].target == "read_file.limit"

    def test_an_assigned_value_is_not_read_as_a_parameter(self, checker):
        """target='both' is one problem, the enum. Reporting it twice helps nobody."""
        found = checker.check_tool("search_files", "Search. Set target='both' for either.")
        assert kinds(found) == [KIND_ENUM_CONTRADICTION]


class TestEnumContradictions:
    def test_an_assignment_outside_the_enum_is_caught(self, checker):
        found = checker.check_tool("search_files", "Search files with target='both'.")
        assert kinds(found) == [KIND_ENUM_CONTRADICTION]
        assert "content, files" in found[0].message

    def test_an_assignment_inside_the_enum_is_fine(self, checker):
        assert checker.check_tool("search_files", "Defaults to target='content'.") == []

    def test_a_value_offered_beside_the_parameter_name_is_caught(self, checker):
        found = checker.check_tool(
            "search_files", "Search. Use target with 'content', 'files' or 'symbols'."
        )
        assert kinds(found) == [KIND_ENUM_CONTRADICTION]
        assert "symbols" in found[0].message

    def test_a_parameter_description_is_checked_against_its_own_enum(self, checker):
        found = checker.check_param(
            "search_files", "target", "'content' scans text, 'symbols' scans definitions"
        )
        assert kinds(found) == [KIND_ENUM_CONTRADICTION]

    def test_a_parameter_description_listing_the_real_enum_is_fine(self, checker):
        assert (
            checker.check_param(
                "search_files", "target", "'content' scans text, 'files' matches names"
            )
            == []
        )

    def test_a_value_belonging_to_a_neighbouring_parameter_is_not_misattributed(self, checker):
        """The window stops at the next parameter, so path's words stay path's."""
        found = checker.check_tool(
            "search_files", "Give a pattern, then target, then path like 'src'."
        )
        assert found == []

    def test_a_parameter_with_no_enum_constrains_nothing(self, checker):
        assert checker.check_tool("read_file", "Read with path=anything_at_all.") == []


class TestRequiredness:
    def test_calling_a_required_parameter_optional_is_caught(self, checker):
        found = checker.check_tool("read_file", "Read a file. The path is optional.")
        assert kinds(found) == [KIND_REQUIREDNESS]
        assert "'path'" in found[0].message

    def test_may_be_omitted_counts_too(self, checker):
        found = checker.check_tool("read_file", "Read a file; path may be omitted.")
        assert kinds(found) == [KIND_REQUIREDNESS]

    def test_saying_it_is_not_optional_is_not_a_finding(self, checker):
        assert checker.check_tool("read_file", "Read a file. path is not optional.") == []

    def test_an_optional_neighbour_does_not_taint_a_required_one(self, checker):
        found = checker.check_tool("read_file", "Read a file. path, then offset (optional).")
        assert found == []

    def test_claiming_no_arguments_contradicts_a_required_parameter(self, checker):
        found = checker.check_tool("read_file", "Reads the current file. Takes no arguments.")
        assert KIND_REQUIREDNESS in kinds(found)
        assert "path" in found[0].message


class TestUnsupportedCapabilities:
    def test_promising_a_timeout_the_schema_cannot_carry_is_caught(self, checker):
        found = checker.check_tool("read_file", "Read a file. Set a timeout in seconds.")
        assert kinds(found) == [KIND_UNSUPPORTED_CAPABILITY]
        assert "timeout" in found[0].message

    def test_promising_an_encoding_is_caught(self, checker):
        found = checker.check_tool("read_file", "Read a file, specifying the encoding you want.")
        assert kinds(found) == [KIND_UNSUPPORTED_CAPABILITY]

    def test_behaviour_claims_are_not_capability_claims(self, checker):
        """The schema cannot adjudicate what the handler does, so it does not try."""
        assert checker.check_tool("read_file", "Reads recursively and retries on failure.") == []

    def test_a_tool_that_has_the_parameter_is_not_flagged(self, checker):
        assert checker.check_tool("terminal", "Run a command; set the timeout.") == []


class TestEntailmentTier:
    def stub(self, supported, claim="it reads your mail"):
        class Stub:
            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return dspy.Prediction(supported=supported, unsupported_claim=claim)

        return Stub()

    def make(self, facts, predictor, lm=object()):
        return FactualAccuracyChecker(facts=facts, entailment=predictor, lm=lm)

    def test_an_unsupported_verdict_becomes_a_finding(self, facts):
        checker = self.make(facts, self.stub("no"))
        found = checker.check_tool("read_file", "Read a file.", "Read a text file.")
        assert kinds(found) == [KIND_ENTAILMENT]
        assert found[0].source == "entailment"
        assert "it reads your mail" in found[0].message

    def test_a_supported_verdict_is_silent(self, facts):
        checker = self.make(facts, self.stub("yes"))
        assert checker.check_tool("read_file", "Read a file.", "Read a text file.") == []

    def test_the_judge_sees_the_schema_and_both_descriptions(self, facts):
        predictor = self.stub("yes")
        checker = self.make(facts, predictor)
        checker.check_tool("read_file", "New text.", "Old text.")
        call = predictor.calls[0]
        assert call["tool_name"] == "read_file"
        assert "offset: integer" in call["tool_schema"]
        assert call["baseline_description"] == "Old text."
        assert call["evolved_description"] == "New text."

    def test_a_missing_quote_still_reports_the_finding(self, facts):
        checker = self.make(facts, self.stub("no", claim="none"))
        found = checker.check_tool("read_file", "Read a file.", "Read a text file.")
        assert kinds(found) == [KIND_ENTAILMENT]
        assert "did not quote" in found[0].message

    def test_structural_findings_survive_alongside_entailment(self, facts):
        checker = self.make(facts, self.stub("no"))
        found = checker.check_tool(
            "read_file", "Read a file. Pass `recursive`.", "Read a text file."
        )
        assert kinds(found) == [KIND_ENTAILMENT, KIND_UNKNOWN_PARAMETER]

    def test_a_judge_that_raises_does_not_take_the_run_down(self, facts):
        def explode(**kwargs):
            raise RuntimeError("no credit left")

        checker = self.make(facts, explode)
        assert checker.check_tool("read_file", "Read a file.", "Read a text file.") == []
        assert "no credit left" in checker.skipped_reason
        assert checker.entailment_ran is False

    def test_no_predictor_means_structural_only(self, facts):
        checker = FactualAccuracyChecker(facts=facts)
        found = checker.check_tool("read_file", "Pass `recursive`.", "Read a text file.")
        assert kinds(found) == [KIND_UNKNOWN_PARAMETER]
        assert "no entailment predictor" in checker.skipped_reason

    def test_no_configured_lm_means_structural_only(self, facts):
        predictor = self.stub("no")
        checker = FactualAccuracyChecker(facts=facts, entailment=predictor, lm=None)
        with dspy.context(lm=None):
            assert checker.check_tool("read_file", "Read a file.", "Read a text file.") == []
        assert predictor.calls == []
        assert checker.skipped_reason == "no LM configured"

    def test_nothing_to_entail_against_skips_the_call(self, facts):
        predictor = self.stub("no")
        checker = self.make(facts, predictor)
        assert checker.check_tool("read_file", "Read a file.") == []
        assert predictor.calls == []


class TestBundleChecking:
    def bundle(self, description, params=None):
        return {"read_file": ToolDescriptions("read_file", description, dict(params or {}))}

    def test_only_changed_descriptions_are_checked(self, checker):
        baseline = self.bundle("Read a file. Pass `recursive`.")
        report = checker.check_bundle(dict(baseline), baseline)
        assert report.checked == 0
        assert report.passed

    def test_a_changed_description_is_checked(self, checker):
        baseline = self.bundle("Read a text file.")
        candidate = self.bundle("Read a file. Pass `recursive`.")
        report = checker.check_bundle(candidate, baseline)
        assert report.checked == 1
        assert not report.passed
        assert report.for_target("read_file")[0].kind == KIND_UNKNOWN_PARAMETER

    def test_changed_parameters_are_checked(self, checker):
        baseline = self.bundle("Read a text file.", {"limit": "How many lines."})
        candidate = self.bundle("Read a text file.", {"limit": "How many; see `stride`."})
        report = checker.check_bundle(candidate, baseline)
        assert list(report.by_target()) == ["read_file.limit"]

    def test_a_tool_outside_the_allowed_set_is_skipped(self, checker):
        baseline = self.bundle("Read a text file.")
        candidate = self.bundle("Pass `recursive`.")
        report = checker.check_bundle(candidate, baseline, allowed=["terminal"])
        assert report.checked == 0

    def test_a_bundle_with_no_baseline_checks_everything(self, checker):
        report = checker.check_bundle(self.bundle("Pass `recursive`."))
        assert report.checked == 1
        assert not report.passed

    def test_the_summary_says_the_entailment_tier_did_not_run(self, checker):
        report = checker.check_bundle(self.bundle("Read a text file."), self.bundle("Old."))
        assert "structural checks only" in report.summary() or "skipped" in report.summary()

    def test_the_summary_says_when_it_did_run(self, facts):
        class Stub:
            def __call__(self, **kwargs):
                return dspy.Prediction(supported="yes", unsupported_claim="none")

        checker = FactualAccuracyChecker(facts=facts, entailment=Stub(), lm=object())
        report = checker.check_bundle(
            {"read_file": ToolDescriptions("read_file", "New.")},
            {"read_file": ToolDescriptions("read_file", "Old.")},
        )
        assert report.entailment_ran
        assert "entailment check ran" in report.summary()

    def test_the_report_serialises(self, checker):
        report = checker.check_bundle(
            self.bundle("Pass `recursive`."), self.bundle("Read a text file.")
        )
        blob = report.to_dict()
        assert blob["passed"] is False
        assert blob["findings"][0]["kind"] == KIND_UNKNOWN_PARAMETER
        assert blob["findings"][0]["source"] == "structural"


class TestFindingBasics:
    def test_describe_includes_the_target_and_evidence(self):
        finding = AccuracyFinding(
            tool="read_file",
            target="read_file.limit",
            kind=KIND_UNKNOWN_PARAMETER,
            message="names 'stride'",
            evidence="see `stride`",
        )
        assert finding.describe() == "read_file.limit: names 'stride' (in: 'see `stride`')"

    def test_describe_without_evidence(self):
        finding = AccuracyFinding("t", "t", KIND_REQUIREDNESS, "says something wrong")
        assert finding.describe() == "t: says something wrong"

    def test_an_empty_report_passes(self):
        assert AccuracyReport().passed
        assert "0 finding(s)" in AccuracyReport().summary()
