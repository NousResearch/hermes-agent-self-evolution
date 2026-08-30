"""Tests for loading, measuring and rewriting the tool catalogue.

Offline: the only hermes-agent involved is the fake one in conftest.
"""

import ast

import pytest

from evolution.core.artifact_io import schema_skeleton
from evolution.core.config import EvolutionConfig
from evolution.tools.tool_catalog import (
    DEFAULT_MAX_PARAM_DESC,
    DEFAULT_MAX_TOOL_DESC,
    ToolDescriptions,
    UnknownTool,
    bundle_from_dict,
    bundle_to_dict,
    diff_bundles,
    discover_toolsets,
    infer_toolset,
    load_catalog,
    write_bundle,
)


class TestLoadCatalog:
    def test_finds_every_literal_schema(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        assert catalog.names == [
            "read_file",
            "search_files",
            "terminal",
            "vision_analyze",
            "write_file",
        ]

    def test_captures_constant_module_and_path(self, hermes_repo):
        entry = load_catalog(hermes_repo).require("read_file")
        assert entry.constant == "READ_FILE_SCHEMA"
        assert entry.module == "file_tools"
        assert entry.path.name == "file_tools.py"

    def test_toolset_comes_from_the_registration(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        assert catalog.require("read_file").toolset == "file"
        assert catalog.require("read_file").toolset_source == "registry"
        assert catalog.require("terminal").toolset == "terminal"

    def test_toolset_is_inferred_when_unregistered(self, hermes_repo):
        entry = load_catalog(hermes_repo).require("vision_analyze")
        assert entry.toolset == "vision"
        assert entry.toolset_source == "inferred"

    def test_required_and_types_come_from_the_frozen_schema(self, hermes_repo):
        entry = load_catalog(hermes_repo).require("write_file")
        assert entry.required == ("path", "content")
        assert entry.param_types["cross_profile"] == "boolean"

    def test_signature_marks_required_parameters(self, hermes_repo):
        signature = load_catalog(hermes_repo).require("read_file").signature()
        assert signature.startswith("read_file(path: string [required]")
        assert "limit: integer" in signature

    def test_empty_repo_is_an_empty_catalogue_not_a_crash(self, empty_repo):
        catalog = load_catalog(empty_repo)
        assert len(catalog) == 0
        assert catalog.budget_findings() == []

    def test_config_overrides_the_budgets(self, hermes_repo):
        config = EvolutionConfig(max_tool_desc_size=50, max_param_desc_size=10)
        catalog = load_catalog(hermes_repo, config)
        assert catalog.max_tool_desc == 50
        assert len(catalog.budget_findings()) > 2

    def test_default_budgets_match_plan(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        assert (catalog.max_tool_desc, catalog.max_param_desc) == (
            DEFAULT_MAX_TOOL_DESC,
            DEFAULT_MAX_PARAM_DESC,
        )


class TestToolsetDiscovery:
    def test_indexes_by_tool_and_by_constant(self, hermes_repo):
        index = discover_toolsets(hermes_repo)
        assert index.by_tool["read_file"] == "file"
        assert index.by_constant["TERMINAL_SCHEMA"] == "terminal"

    def test_lookup_falls_back_to_the_constant(self, hermes_repo):
        index = discover_toolsets(hermes_repo)
        assert index.lookup("nope", "TERMINAL_SCHEMA") == "terminal"
        assert index.lookup("nope", "NOPE_SCHEMA") is None

    def test_missing_tools_dir_gives_an_empty_index(self, empty_repo):
        assert not discover_toolsets(empty_repo)

    def test_positional_registration_is_read_too(self, tmp_path):
        tools = tmp_path / "tools"
        tools.mkdir()
        (tools / "odd_tools.py").write_text(
            'S = {"name": "odd", "description": "Odd.", '
            '"parameters": {"type": "object", "properties": {}, "required": []}}\n'
            'register("odd", "weird", S, None)\n'
        )
        assert discover_toolsets(tmp_path).by_tool["odd"] == "weird"

    @pytest.mark.parametrize(
        "module,expected",
        [
            ("file_tools", "file"),
            ("browser_tool", "browser"),
            ("registry", "registry"),
            ("_tools", "_tools"),
        ],
    )
    def test_infer_toolset(self, module, expected):
        assert infer_toolset(module) == expected


class TestGrouping:
    def test_by_toolset(self, hermes_repo):
        grouped = load_catalog(hermes_repo).by_toolset()
        assert sorted(grouped) == ["file", "terminal", "vision"]
        assert [e.tool_name for e in grouped["file"]] == [
            "read_file",
            "search_files",
            "write_file",
        ]

    def test_by_module(self, hermes_repo):
        grouped = load_catalog(hermes_repo).by_module()
        assert sorted(grouped) == ["file_tools", "shell_tools", "vision_tools"]

    def test_get_and_contains(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        assert "terminal" in catalog
        assert "nope" not in catalog
        assert catalog.get("nope") is None

    def test_require_raises_for_unknown(self, hermes_repo):
        with pytest.raises(UnknownTool):
            load_catalog(hermes_repo).require("nope")


class TestSelect:
    def test_select_by_name(self, hermes_repo):
        chosen = load_catalog(hermes_repo).select(tools=["terminal", "read_file"])
        assert chosen.names == ["read_file", "terminal"]

    def test_select_by_toolset(self, hermes_repo):
        chosen = load_catalog(hermes_repo).select(toolset="file")
        assert chosen.names == ["read_file", "search_files", "write_file"]

    def test_select_combines_both(self, hermes_repo):
        chosen = load_catalog(hermes_repo).select(tools=["terminal", "read_file"], toolset="file")
        assert chosen.names == ["read_file"]

    def test_unknown_name_is_an_error_not_a_silent_empty_run(self, hermes_repo):
        with pytest.raises(UnknownTool) as excinfo:
            load_catalog(hermes_repo).select(tools=["read_file", "nope"])
        assert "nope" in str(excinfo.value)

    def test_select_preserves_budgets(self, hermes_repo):
        config = EvolutionConfig(max_tool_desc_size=42)
        chosen = load_catalog(hermes_repo, config).select(toolset="file")
        assert chosen.max_tool_desc == 42


class TestBudgets:
    def test_reports_the_over_budget_tool_without_crashing(self, hermes_repo):
        findings = load_catalog(hermes_repo).budget_findings()
        tool_findings = [f for f in findings if f.kind == "tool"]
        assert [f.tool_name for f in tool_findings] == ["terminal"]
        assert tool_findings[0].size > 500
        assert tool_findings[0].overage == tool_findings[0].size - 500

    def test_reports_the_over_budget_parameter(self, hermes_repo):
        findings = load_catalog(hermes_repo).budget_findings()
        param_findings = [f for f in findings if f.kind == "param"]
        assert [f.target for f in param_findings] == ["write_file.cross_profile"]

    def test_exactly_at_budget_is_not_over(self, hermes_repo):
        entry = load_catalog(hermes_repo).require("read_file")
        entry.max_tool_desc = entry.description_size
        assert entry.budget_findings() == []
        entry.max_tool_desc = entry.description_size - 1
        assert len(entry.budget_findings()) == 1

    def test_param_exactly_at_budget_is_not_over(self, hermes_repo):
        entry = load_catalog(hermes_repo).require("read_file")
        widest = max(entry.param_sizes, key=lambda name: entry.param_sizes[name])
        entry.max_param_desc = entry.param_sizes[widest]
        assert entry.budget_findings() == []
        entry.max_param_desc = entry.param_sizes[widest] - 1
        assert [f.param for f in entry.budget_findings()] == [widest]

    def test_total_chars_counts_tool_and_param_text(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        entry = catalog.require("read_file")
        assert entry.total_chars == entry.description_size + sum(entry.param_sizes.values())
        assert catalog.total_description_chars == sum(e.total_chars for e in catalog)

    def test_finding_describes_itself(self, hermes_repo):
        finding = [f for f in load_catalog(hermes_repo).budget_findings() if f.kind == "tool"][0]
        assert "over budget" in finding.describe()
        assert finding.to_dict()["budget"] == 500

    def test_catalog_to_dict_lists_over_budget(self, hermes_repo):
        blob = load_catalog(hermes_repo).to_dict()
        assert len(blob["over_budget"]) == 2
        assert blob["toolsets"]["terminal"] == ["terminal"]


class TestBundles:
    def test_bundle_holds_the_evolvable_text(self, hermes_repo):
        bundle = load_catalog(hermes_repo).bundle()
        assert set(bundle) == {
            "read_file",
            "search_files",
            "terminal",
            "vision_analyze",
            "write_file",
        }
        assert bundle["read_file"].params["path"] == "Path to the file to read"

    def test_bundle_dict_round_trip(self, hermes_repo):
        bundle = load_catalog(hermes_repo).bundle()
        assert bundle_to_dict(bundle_from_dict(bundle_to_dict(bundle))) == bundle_to_dict(bundle)

    def test_copy_is_deep_enough_to_edit_safely(self, hermes_repo):
        bundle = load_catalog(hermes_repo).bundle()
        clone = bundle["read_file"].copy()
        clone.params["path"] = "changed"
        assert bundle["read_file"].params["path"] == "Path to the file to read"

    def test_total_chars(self):
        descriptions = ToolDescriptions("t", "abcd", {"p": "xy"})
        assert descriptions.total_chars == 6


class TestDiffBundles:
    def test_detects_tool_and_param_changes(self, hermes_repo):
        baseline = load_catalog(hermes_repo).bundle()
        candidate = {k: v.copy() for k, v in baseline.items()}
        candidate["read_file"].description = "Read a file."
        candidate["read_file"].params["limit"] = "How many lines."
        changes = diff_bundles(baseline, candidate)
        assert [(c.kind, c.target) for c in changes] == [
            ("tool", "read_file"),
            ("param", "read_file.limit"),
        ]

    def test_unchanged_bundle_has_no_changes(self, hermes_repo):
        baseline = load_catalog(hermes_repo).bundle()
        assert diff_bundles(baseline, {k: v.copy() for k, v in baseline.items()}) == []

    def test_delta_chars(self, hermes_repo):
        baseline = load_catalog(hermes_repo).bundle()
        candidate = {k: v.copy() for k, v in baseline.items()}
        candidate["read_file"].description = "Read."
        change = diff_bundles(baseline, candidate)[0]
        assert change.delta_chars == 5 - len(baseline["read_file"].description)

    def test_invented_tools_are_ignored(self, hermes_repo):
        baseline = load_catalog(hermes_repo).bundle()
        candidate = {k: v.copy() for k, v in baseline.items()}
        candidate["imaginary"] = ToolDescriptions("imaginary", "Not real.")
        assert diff_bundles(baseline, candidate) == []


class TestWriteBundle:
    def test_writes_a_tool_description(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        bundle = catalog.bundle()
        bundle["read_file"].description = "Read a text file with line numbers."
        report = write_bundle(hermes_repo, bundle, baseline=catalog.bundle())

        assert report.changed == 1
        assert load_catalog(hermes_repo).require("read_file").description == (
            "Read a text file with line numbers."
        )

    def test_writes_several_edits_to_one_file(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        baseline = catalog.bundle()
        bundle = catalog.bundle()
        bundle["read_file"].description = "Read a file."
        bundle["read_file"].params["limit"] = "Line cap."
        bundle["write_file"].params["path"] = "Where to write."
        write_bundle(hermes_repo, bundle, baseline=baseline)

        reloaded = load_catalog(hermes_repo)
        assert reloaded.require("read_file").description == "Read a file."
        assert reloaded.require("read_file").param_descriptions["limit"] == "Line cap."
        assert reloaded.require("write_file").param_descriptions["path"] == "Where to write."

    def test_touches_only_the_files_that_changed(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        before = (hermes_repo / "tools" / "shell_tools.py").read_text()
        bundle = catalog.bundle()
        bundle["read_file"].description = "Read a file."
        report = write_bundle(hermes_repo, bundle, baseline=catalog.bundle())

        assert [p.name for p in report.files_written] == ["file_tools.py"]
        assert (hermes_repo / "tools" / "shell_tools.py").read_text() == before

    def test_dry_run_changes_nothing_on_disk(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        before = (hermes_repo / "tools" / "file_tools.py").read_text()
        bundle = catalog.bundle()
        bundle["read_file"].description = "Read a file."
        report = write_bundle(hermes_repo, bundle, dry_run=True, baseline=catalog.bundle())

        assert report.dry_run and report.changed == 1
        assert "would write" in report.summary()
        assert (hermes_repo / "tools" / "file_tools.py").read_text() == before

    def test_structure_stays_frozen(self, hermes_repo):
        path = hermes_repo / "tools" / "file_tools.py"
        before = schema_skeleton(path.read_text())
        catalog = load_catalog(hermes_repo)
        bundle = catalog.bundle()
        bundle["write_file"].description = "Write a file."
        bundle["write_file"].params["cross_profile"] = "Cross-profile opt out."
        write_bundle(hermes_repo, bundle, baseline=catalog.bundle())

        assert schema_skeleton(path.read_text()) == before

    def test_hostile_description_cannot_inject_schema(self, hermes_repo):
        """Text that looks like source must be escaped, not executed."""
        path = hermes_repo / "tools" / "file_tools.py"
        before = schema_skeleton(path.read_text())
        catalog = load_catalog(hermes_repo)
        bundle = catalog.bundle()
        bundle["read_file"].description = (
            'Read a file", "required": ["evil"], "x": "\\n injected \'quotes\' and \\\\ backslash'
        )
        write_bundle(hermes_repo, bundle, baseline=catalog.bundle())

        after_source = path.read_text()
        ast.parse(after_source)
        assert schema_skeleton(after_source) == before
        assert load_catalog(hermes_repo).require("read_file").description == (
            bundle["read_file"].description
        )

    def test_multiline_description_survives(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        bundle = catalog.bundle()
        bundle["search_files"].description = "Line one.\n\nLine three with  spacing.\n"
        write_bundle(hermes_repo, bundle, baseline=catalog.bundle())
        assert load_catalog(hermes_repo).require("search_files").description == (
            "Line one.\n\nLine three with  spacing.\n"
        )

    def test_unchanged_text_is_not_rewritten(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        before = (hermes_repo / "tools" / "file_tools.py").read_text()
        report = write_bundle(hermes_repo, catalog.bundle(), baseline=catalog.bundle())

        assert report.changed == 0
        assert report.files_written == []
        assert (hermes_repo / "tools" / "file_tools.py").read_text() == before

    def test_unknown_tool_is_skipped_with_a_reason(self, hermes_repo):
        bundle = {"ghost": ToolDescriptions("ghost", "Not in this repo.")}
        report = write_bundle(hermes_repo, bundle)
        assert report.skipped == [("ghost", "not present in this hermes-agent checkout")]
        assert report.changed == 0

    def test_unknown_parameter_is_skipped_with_a_reason(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        bundle = catalog.bundle()
        bundle["read_file"].params["invented"] = "Does not exist."
        report = write_bundle(hermes_repo, bundle, baseline=catalog.bundle())
        assert report.skipped == [("read_file.invented", "no such parameter in the schema")]

    def test_report_serialises(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        bundle = catalog.bundle()
        bundle["terminal"].description = "Run a command."
        blob = write_bundle(hermes_repo, bundle, dry_run=True, baseline=catalog.bundle()).to_dict()
        assert blob["dry_run"] is True
        assert blob["changes"][0]["tool_name"] == "terminal"

    def test_writing_without_a_baseline_still_only_writes_diffs(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        bundle = catalog.bundle()
        bundle["terminal"].description = "Run a command."
        report = write_bundle(hermes_repo, bundle)
        assert [c.target for c in report.changes] == ["terminal"]
