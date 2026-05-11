"""Tests for evolution.tools.tool_proposer.

The post-extraction logic is tested as a pure function (no LM mocks).
The __call__ wrapper is tested with a monkey-patched self.propose so we
exercise the counter-increment + re-raise contract on SentinelParseError.
"""

from pathlib import Path

import pytest

from evolution.tools.tool_module import _render_manifest_for_prompt
from evolution.tools.tool_proposer import (
    BudgetAwareToolProposer,
    extract_and_rebuild,
)
from evolution.tools.tool_source import SentinelParseError, ToolManifest

FIXTURES = Path(__file__).parent.parent / "fixtures" / "tool_manifests"


class TestExtractAndRebuild:
    def test_preserved_sentinels_returns_rebuilt_full_instructions(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        candidate = _render_manifest_for_prompt(
            manifest, "search_files", "A new description for searching files."
        )
        rebuilt = extract_and_rebuild(candidate, manifest, "search_files")
        expected = _render_manifest_for_prompt(
            manifest, "search_files", "A new description for searching files."
        )
        assert rebuilt == expected

    def test_stripped_open_sentinel_raises(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        candidate = _render_manifest_for_prompt(manifest, "search_files", "X")
        candidate = candidate.replace("<!-- TARGET:search_files -->", "")
        with pytest.raises(SentinelParseError, match="opening sentinel"):
            extract_and_rebuild(candidate, manifest, "search_files")

    def test_stripped_close_sentinel_raises(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        candidate = _render_manifest_for_prompt(manifest, "search_files", "X")
        candidate = candidate.replace("<!-- /TARGET:search_files -->", "")
        with pytest.raises(SentinelParseError, match="closing sentinel"):
            extract_and_rebuild(candidate, manifest, "search_files")

    def test_duplicated_sentinels_raise(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        candidate = _render_manifest_for_prompt(manifest, "search_files", "X")
        candidate += "\n<!-- TARGET:search_files -->Y<!-- /TARGET:search_files -->"
        with pytest.raises(SentinelParseError, match="multiple"):
            extract_and_rebuild(candidate, manifest, "search_files")

    def test_wrong_target_name_in_marker_raises(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        candidate = "<!-- TARGET:wrong_name -->X<!-- /TARGET:wrong_name -->"
        with pytest.raises(SentinelParseError, match="opening sentinel"):
            extract_and_rebuild(candidate, manifest, "search_files")


class TestBudgetAwareToolProposerErrorHandling:
    def test_sentinel_parse_failure_increments_counter_and_reraises(self, monkeypatch):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        proposer = BudgetAwareToolProposer(
            target_tool_name="search_files",
            manifest=manifest,
            target_description="Find things.",
            baseline_chars=len("Find things."),
        )

        class FakePrediction:
            def __init__(self, improved_instruction: str):
                self.improved_instruction = improved_instruction

        def fake_propose(*args, **kwargs):
            return FakePrediction(improved_instruction="No sentinels here.")

        monkeypatch.setattr(proposer, "propose", fake_propose)

        with pytest.raises(SentinelParseError):
            proposer(
                candidate={proposer.component_name: "ignored_current_instruction"},
                reflective_dataset={proposer.component_name: []},
                components_to_update=[proposer.component_name],
            )
        assert proposer.sentinel_failures == 1

    def test_successful_candidate_returns_rebuilt_instructions(self, monkeypatch):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        proposer = BudgetAwareToolProposer(
            target_tool_name="search_files",
            manifest=manifest,
            target_description="Find things.",
            baseline_chars=len("Find things."),
        )

        new_description = "Find files in the repo by name or path pattern."
        good_candidate = _render_manifest_for_prompt(
            manifest, "search_files", new_description
        )

        class FakePrediction:
            def __init__(self, improved_instruction: str):
                self.improved_instruction = improved_instruction

        monkeypatch.setattr(
            proposer, "propose",
            lambda *args, **kwargs: FakePrediction(improved_instruction=good_candidate),
        )

        updated = proposer(
            candidate={proposer.component_name: "ignored"},
            reflective_dataset={proposer.component_name: []},
            components_to_update=[proposer.component_name],
        )
        assert proposer.sentinel_failures == 0
        assert proposer.component_name in updated
        expected = _render_manifest_for_prompt(manifest, "search_files", new_description)
        assert updated[proposer.component_name] == expected


class TestBudgetAwareToolProposerTemplate:
    def test_template_includes_sentinel_preservation_rule(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        proposer = BudgetAwareToolProposer(
            target_tool_name="search_files",
            manifest=manifest,
            target_description="Find things.",
            baseline_chars=len("Find things."),
        )
        template = proposer.propose.signature.instructions
        assert "Modify only the text between" in template
        assert "<!-- TARGET:" in template
        assert "<!-- /TARGET:" in template

    def test_template_carries_forward_decision_rubric(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        proposer = BudgetAwareToolProposer(
            target_tool_name="search_files",
            manifest=manifest,
            target_description="Find things.",
            baseline_chars=len("Find things."),
        )
        template = proposer.propose.signature.instructions
        assert "(a)" in template
        assert "(b)" in template
        assert "(c)" in template

    def test_template_includes_grounding_citation_requirement(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        proposer = BudgetAwareToolProposer(
            target_tool_name="search_files",
            manifest=manifest,
            target_description="Find things.",
            baseline_chars=len("Find things."),
        )
        template = proposer.propose.signature.instructions
        assert "quote or paraphrase" in template

    def test_template_length_budget_is_against_description_not_full_manifest(self):
        """The length budget must reflect the description's size (~12 chars), not
        the rendered manifest's size (~1000+ chars). A common confusion mode for
        budget-aware proposers is to anchor the budget on the full instructions.
        """
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        baseline = "Find things."
        proposer = BudgetAwareToolProposer(
            target_tool_name="search_files",
            manifest=manifest,
            target_description=baseline,
            baseline_chars=len(baseline),
        )
        template = proposer.propose.signature.instructions
        # The current description's length must be quoted on its own anchor line.
        assert f"current description is {len(baseline)} characters" in template
        # The full rendered manifest (which the proposer renders for evaluation) is
        # ~1000+ chars. That number must NOT appear in the budget framing.
        rendered = _render_manifest_for_prompt(manifest, "search_files", baseline)
        manifest_len = len(rendered)
        assert manifest_len > 500, "fixture sanity: manifest should be >500 chars"
        assert str(manifest_len) not in template, (
            "budget framing should not reference the full manifest length"
        )

    def test_component_name_matches_named_predictors(self):
        """Cross-module invariant: BudgetAwareToolProposer.component_name must match
        what ToolModule.named_predictors() actually exposes. If DSPy renames the
        inner Predict attribute of ChainOfThought, this test catches it loudly
        instead of degrading the proposer to a silent no-op.
        """
        from evolution.tools.tool_module import ToolModule
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        module = ToolModule("search_files", manifest, "Find things.")
        predictor_names = {name for name, _ in module.named_predictors()}
        assert BudgetAwareToolProposer.component_name in predictor_names, (
            f"component_name={BudgetAwareToolProposer.component_name!r} not in "
            f"ToolModule.named_predictors()={predictor_names}; DSPy internals may have changed"
        )
