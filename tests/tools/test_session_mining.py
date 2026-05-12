"""Tests for evolution.tools.session_mining — Hermes session log mining."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from evolution.core.dataset_builder import EvalExample
from evolution.core.external_importers import HermesSessionImporter
from evolution.tools.session_mining import (
    CATEGORY_AGREED,
    CATEGORY_MISSELECTION,
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    HermesToolImporter,
    ToolRelevanceFilter,
    _extract_tool_name,
    build_tool_dataset_from_sessions,
)
from evolution.tools.tool_source import ToolManifest

FIXTURES = Path(__file__).parent.parent / "fixtures" / "tool_manifests"


@pytest.fixture
def manifest():
    """Manifest containing search_files, grep_in_terminal, read_file,
    cat_in_terminal, list_directory, ls_in_terminal, compute_sha256."""
    return ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")


@pytest.fixture
def session_dir(tmp_path):
    """Patches HermesSessionImporter.SESSION_DIR to tmp_path for the test."""
    with patch.object(HermesSessionImporter, "SESSION_DIR", tmp_path):
        yield tmp_path


def _write_session(dir_path: Path, name: str, messages: list[dict], session_id: str = "") -> None:
    payload = {"messages": messages}
    if session_id:
        payload["session_id"] = session_id
    (dir_path / f"{name}.json").write_text(json.dumps(payload))


class TestExtractToolName:
    """The shape-tolerant tool-name extractor."""

    def test_handles_openai_nested_shape(self):
        assert _extract_tool_name({"function": {"name": "search_files"}}) == "search_files"

    def test_handles_flat_shape(self):
        assert _extract_tool_name({"name": "search_files"}) == "search_files"

    def test_prefers_nested_when_both_present(self):
        # The nested form is the canonical OpenAI shape; prefer it.
        result = _extract_tool_name({"function": {"name": "real"}, "name": "stale"})
        assert result == "real"

    def test_returns_none_when_neither_shape_present(self):
        assert _extract_tool_name({"id": "abc"}) is None

    def test_falls_back_when_nested_missing_name(self):
        assert _extract_tool_name({"function": {}, "name": "fallback"}) == "fallback"


class TestHermesToolImporterExtraction:
    """Core extraction logic — tool_calls parsing, manifest filter, drop counters."""

    def test_extracts_single_tool_call(self, manifest, session_dir):
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "Find Python test files"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "search_files"}}]},
            {"role": "tool", "content": "results"},
        ], session_id="abc")

        candidates, drops = HermesToolImporter.extract_candidates(manifest)

        assert len(candidates) == 1
        assert candidates[0]["task_input"] == "Find Python test files"
        assert candidates[0]["invoked_tool"] == "search_files"
        assert candidates[0]["session_id"] == "abc"
        assert candidates[0]["source"] == "hermes"

    def test_picks_last_tool_call_in_chain(self, manifest, session_dir):
        # First call is "get oriented"; last is the one that resolved intent.
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "Show me the failing test output"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "read_file"}},
                {"function": {"name": "search_files"}},
                {"function": {"name": "grep_in_terminal"}},
            ]},
        ])
        candidates, _ = HermesToolImporter.extract_candidates(manifest)
        assert len(candidates) == 1
        assert candidates[0]["invoked_tool"] == "grep_in_terminal"

    def test_handles_flat_tool_call_shape(self, manifest, session_dir):
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "Look for the bug in auth.py"},
            {"role": "assistant", "tool_calls": [{"name": "read_file"}]},
        ])
        candidates, _ = HermesToolImporter.extract_candidates(manifest)
        assert candidates[0]["invoked_tool"] == "read_file"

    def test_drops_non_manifest_invocations(self, manifest, session_dir):
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "Do something with terraform"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "tf_apply"}}]},
        ])
        candidates, drops = HermesToolImporter.extract_candidates(manifest)
        assert candidates == []
        assert drops["non_manifest"] == 1

    def test_drops_when_no_tool_calls(self, manifest, session_dir):
        # Assistant answered with text only — no selection signal.
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "What does this code do?"},
            {"role": "assistant", "content": "This iterates over the list."},
        ])
        candidates, drops = HermesToolImporter.extract_candidates(manifest)
        assert candidates == []
        assert drops["no_tool_calls"] == 1

    def test_drops_short_tasks(self, manifest, session_dir):
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "search_files"}}]},
        ])
        candidates, drops = HermesToolImporter.extract_candidates(manifest)
        assert candidates == []
        assert drops["short_task"] == 1

    def test_drops_slash_commands(self, manifest, session_dir):
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "/plugin install foo-bar"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "search_files"}}]},
        ])
        candidates, drops = HermesToolImporter.extract_candidates(manifest)
        assert candidates == []
        assert drops["slash_command"] == 1

    def test_drops_tasks_with_secrets(self, manifest, session_dir):
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "Use ANTHROPIC_API_KEY=sk-ant-api03-xyz and read main.py"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "read_file"}}]},
        ])
        candidates, drops = HermesToolImporter.extract_candidates(manifest)
        assert candidates == []
        assert drops["secret"] == 1

    def test_stops_at_next_user_message(self, manifest, session_dir):
        # User 1's tool call would be missed if we kept scanning into User 2's turn.
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "Initial question without a tool"},
            {"role": "assistant", "content": "Sure, I'll need more info."},
            {"role": "user", "content": "Please search for foo across the repo"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "search_files"}}]},
        ])
        candidates, drops = HermesToolImporter.extract_candidates(manifest)
        assert len(candidates) == 1
        assert candidates[0]["task_input"] == "Please search for foo across the repo"
        # First user message gets the "no_tool_calls" drop.
        assert drops["no_tool_calls"] == 1

    def test_handles_intervening_tool_result(self, manifest, session_dir):
        # tool_result messages between user and assistant shouldn't terminate the scan.
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "Read the config file and tell me the port"},
            {"role": "tool", "content": "earlier result"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "read_file"}}]},
        ])
        candidates, _ = HermesToolImporter.extract_candidates(manifest)
        assert len(candidates) == 1
        assert candidates[0]["invoked_tool"] == "read_file"

    def test_skips_text_only_assistant_turns_before_tool_call(self, manifest, session_dir):
        # Assistant talks first, then makes a tool call — still attributable to the user.
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "Help me find a config issue"},
            {"role": "assistant", "content": "Let me look."},
            {"role": "assistant", "tool_calls": [{"function": {"name": "read_file"}}]},
        ])
        candidates, _ = HermesToolImporter.extract_candidates(manifest)
        assert len(candidates) == 1
        assert candidates[0]["invoked_tool"] == "read_file"

    def test_respects_limit(self, manifest, session_dir):
        _write_session(session_dir, "s1", [
            msg
            for i in range(5)
            for msg in (
                {"role": "user", "content": f"Read file number {i} please"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "read_file"}}]},
            )
        ])
        candidates, _ = HermesToolImporter.extract_candidates(manifest, limit=3)
        assert len(candidates) == 3

    def test_empty_session_dir_returns_empty(self, manifest, session_dir):
        candidates, drops = HermesToolImporter.extract_candidates(manifest)
        assert candidates == []
        assert all(v == 0 for v in drops.values())

    def test_handles_missing_session_dir(self, manifest, tmp_path):
        with patch.object(HermesSessionImporter, "SESSION_DIR", tmp_path / "nope"):
            candidates, drops = HermesToolImporter.extract_candidates(manifest)
        assert candidates == []
        assert all(v == 0 for v in drops.values())


def _make_filter_with_scorer(manifest: ToolManifest, scorer_response: str) -> ToolRelevanceFilter:
    """Build a ToolRelevanceFilter whose scorer returns a fixed JSON response."""
    rf = ToolRelevanceFilter.__new__(ToolRelevanceFilter)
    rf.model = "test-model"
    rf.manifest = manifest
    rf.manifest_names = {t.name for t in manifest.tools}
    rf.seed = 42
    rf._manifest_summary = "- search_files: Find things.\n- grep_in_terminal: Grep."
    rf.scorer = MagicMock(return_value=SimpleNamespace(scoring=scorer_response))
    return rf


def _make_filter_with_responses(manifest: ToolManifest, responses: list[str]) -> ToolRelevanceFilter:
    """Build a filter whose scorer returns a different response per call."""
    rf = _make_filter_with_scorer(manifest, scorer_response="")
    rf.scorer = MagicMock(side_effect=[SimpleNamespace(scoring=r) for r in responses])
    return rf


@pytest.fixture
def mock_dspy_for_filter():
    """Mock dspy.LM and dspy.context for ToolRelevanceFilter tests."""
    with patch("evolution.tools.session_mining.dspy") as mock:
        mock.context.return_value.__enter__ = MagicMock(return_value=None)
        mock.context.return_value.__exit__ = MagicMock(return_value=False)
        yield mock


class TestToolRelevanceFilterBandDecision:
    """The confidence-band decision table — the load-bearing rule."""

    def test_agreement_kept_with_agreed_category(self, manifest, mock_dspy_for_filter):
        rf = _make_filter_with_scorer(manifest, json.dumps({
            "relevant": True,
            "correct_tool": "search_files",
            "confidence": 0.5,  # confidence ignored on agreement
        }))
        candidates = [{"task_input": "Find Python tests", "invoked_tool": "search_files", "source": "hermes"}]

        examples, drops = rf.filter_and_score(candidates, max_examples=10)

        assert len(examples) == 1
        assert examples[0].expected_behavior == "search_files"
        assert examples[0].category == CATEGORY_AGREED
        assert examples[0].source == "hermes"
        assert all(v == 0 for v in drops.values())

    def test_high_confidence_disagreement_flips_label(self, manifest, mock_dspy_for_filter):
        rf = _make_filter_with_scorer(manifest, json.dumps({
            "relevant": True,
            "correct_tool": "search_files",
            "confidence": 0.9,  # ≥ 0.85
        }))
        candidates = [{"task_input": "Find Python tests", "invoked_tool": "grep_in_terminal", "source": "hermes"}]

        examples, drops = rf.filter_and_score(candidates, max_examples=10)

        assert len(examples) == 1
        assert examples[0].expected_behavior == "search_files"  # flipped
        assert examples[0].category == CATEGORY_MISSELECTION
        assert all(v == 0 for v in drops.values())

    def test_noisy_middle_disagreement_dropped(self, manifest, mock_dspy_for_filter):
        rf = _make_filter_with_scorer(manifest, json.dumps({
            "relevant": True,
            "correct_tool": "search_files",
            "confidence": 0.7,  # 0.6 ≤ conf < 0.85
        }))
        candidates = [{"task_input": "Find Python tests", "invoked_tool": "grep_in_terminal", "source": "hermes"}]

        examples, drops = rf.filter_and_score(candidates, max_examples=10)

        assert examples == []
        assert drops["noisy_middle"] == 1

    def test_low_confidence_disagreement_dropped(self, manifest, mock_dspy_for_filter):
        rf = _make_filter_with_scorer(manifest, json.dumps({
            "relevant": True,
            "correct_tool": "search_files",
            "confidence": 0.3,  # < 0.6
        }))
        candidates = [{"task_input": "Find Python tests", "invoked_tool": "grep_in_terminal", "source": "hermes"}]

        examples, drops = rf.filter_and_score(candidates, max_examples=10)

        assert examples == []
        assert drops["low_confidence"] == 1

    def test_judge_threshold_exactly_at_high_bound_keeps(self, manifest, mock_dspy_for_filter):
        # confidence == 0.85 → keep (boundary is inclusive at high end)
        rf = _make_filter_with_scorer(manifest, json.dumps({
            "relevant": True,
            "correct_tool": "search_files",
            "confidence": HIGH_CONFIDENCE_THRESHOLD,
        }))
        candidates = [{"task_input": "Find Python tests", "invoked_tool": "grep_in_terminal", "source": "hermes"}]

        examples, drops = rf.filter_and_score(candidates, max_examples=10)

        assert len(examples) == 1
        assert examples[0].category == CATEGORY_MISSELECTION

    def test_judge_threshold_exactly_at_low_bound_drops_as_noisy(self, manifest, mock_dspy_for_filter):
        # confidence == 0.6 → drop into noisy_middle (boundary inclusive at low side of middle band)
        rf = _make_filter_with_scorer(manifest, json.dumps({
            "relevant": True,
            "correct_tool": "search_files",
            "confidence": LOW_CONFIDENCE_THRESHOLD,
        }))
        candidates = [{"task_input": "Find Python tests", "invoked_tool": "grep_in_terminal", "source": "hermes"}]

        examples, drops = rf.filter_and_score(candidates, max_examples=10)

        assert examples == []
        assert drops["noisy_middle"] == 1


class TestToolRelevanceFilterEdgeCases:
    """Drop reasons that aren't the confidence band itself."""

    def test_irrelevant_judgment_dropped(self, manifest, mock_dspy_for_filter):
        rf = _make_filter_with_scorer(manifest, json.dumps({"relevant": False}))
        candidates = [{"task_input": "Random text", "invoked_tool": "search_files", "source": "hermes"}]

        examples, drops = rf.filter_and_score(candidates, max_examples=10)
        assert examples == []
        assert drops["judge_irrelevant"] == 1

    def test_malformed_json_counted_as_judge_error(self, manifest, mock_dspy_for_filter):
        rf = _make_filter_with_scorer(manifest, "the judge could not parse this task")
        candidates = [{"task_input": "Find files", "invoked_tool": "search_files", "source": "hermes"}]

        examples, drops = rf.filter_and_score(candidates, max_examples=10)
        assert examples == []
        assert drops["judge_error"] == 1

    def test_judge_exception_counted_as_judge_error(self, manifest, mock_dspy_for_filter):
        rf = _make_filter_with_scorer(manifest, "")
        rf.scorer = MagicMock(side_effect=RuntimeError("judge LM timed out"))
        candidates = [{"task_input": "Find files", "invoked_tool": "search_files", "source": "hermes"}]

        examples, drops = rf.filter_and_score(candidates, max_examples=10)
        assert examples == []
        assert drops["judge_error"] == 1

    def test_correct_tool_outside_manifest_dropped(self, manifest, mock_dspy_for_filter):
        # The judge hallucinates a tool name not in the manifest.
        rf = _make_filter_with_scorer(manifest, json.dumps({
            "relevant": True,
            "correct_tool": "imaginary_tool",
            "confidence": 0.95,
        }))
        candidates = [{"task_input": "Find Python tests", "invoked_tool": "search_files", "source": "hermes"}]

        examples, drops = rf.filter_and_score(candidates, max_examples=10)
        assert examples == []
        assert drops["unknown_correct_tool"] == 1

    def test_max_examples_cap_stops_collection(self, manifest, mock_dspy_for_filter):
        rf = _make_filter_with_scorer(manifest, json.dumps({
            "relevant": True,
            "correct_tool": "search_files",
            "confidence": 0.9,
        }))
        candidates = [
            {"task_input": f"Task number {i}", "invoked_tool": "search_files", "source": "hermes"}
            for i in range(20)
        ]

        examples, _ = rf.filter_and_score(candidates, max_examples=5)
        assert len(examples) == 5

    def test_cost_ceiling_caps_judge_calls(self, manifest, mock_dspy_for_filter):
        # All judgements come back irrelevant (so nothing is collected), but the
        # filter should still cap LM calls at max_examples * 2.
        rf = _make_filter_with_scorer(manifest, json.dumps({"relevant": False}))
        candidates = [
            {"task_input": f"Task number {i}", "invoked_tool": "search_files", "source": "hermes"}
            for i in range(100)
        ]

        rf.filter_and_score(candidates, max_examples=10)
        assert rf.scorer.call_count == 20  # 10 * 2

    def test_garbage_confidence_treated_as_zero(self, manifest, mock_dspy_for_filter):
        # LLM might emit "high" or null; we coerce to 0.0 and drop into low_confidence.
        rf = _make_filter_with_responses(manifest, [
            json.dumps({"relevant": True, "correct_tool": "search_files", "confidence": "high"}),
            json.dumps({"relevant": True, "correct_tool": "search_files", "confidence": None}),
        ])
        candidates = [
            {"task_input": "Find files", "invoked_tool": "grep_in_terminal", "source": "hermes"},
            {"task_input": "Find more files", "invoked_tool": "grep_in_terminal", "source": "hermes"},
        ]

        examples, drops = rf.filter_and_score(candidates, max_examples=10)
        assert examples == []
        assert drops["low_confidence"] == 2


class TestBuildToolDatasetFromSessions:
    """End-to-end orchestrator: sessions → candidates → judge → splits."""

    def test_empty_session_dir_returns_empty_dataset(self, manifest, session_dir, mock_dspy_for_filter):
        dataset, drops = build_tool_dataset_from_sessions(
            manifest=manifest,
            target_tool="search_files",
            output_path=None,
            model="test-model",
            max_examples=10,
        )
        assert dataset.train == [] and dataset.val == [] and dataset.holdout == []

    def test_full_pipeline_produces_agreed_and_misselection_examples(
        self, manifest, session_dir, mock_dspy_for_filter, tmp_path
    ):
        # Session 1: agent picked search_files for a search-y task — judge agrees.
        _write_session(session_dir, "agree", [
            {"role": "user", "content": "Find Python test files in src/"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "search_files"}}]},
        ], session_id="sess-agree")
        # Session 2: agent picked grep_in_terminal for a search-y task — judge says
        # search_files would have been correct.
        _write_session(session_dir, "misselect", [
            {"role": "user", "content": "Locate all files matching test_*.py"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "grep_in_terminal"}}]},
        ], session_id="sess-miss")

        # Patch ToolRelevanceFilter to issue per-candidate judgements without an LM.
        # Candidates come back in session walk order; both should pass through.
        def fake_filter_and_score(self, candidates, max_examples=200):
            examples = []
            for cand in candidates:
                if cand["invoked_tool"] == "search_files":
                    examples.append(EvalExample(
                        task_input=cand["task_input"],
                        expected_behavior="search_files",
                        category=CATEGORY_AGREED,
                        source=cand["source"],
                    ))
                else:
                    examples.append(EvalExample(
                        task_input=cand["task_input"],
                        expected_behavior="search_files",
                        category=CATEGORY_MISSELECTION,
                        source=cand["source"],
                    ))
            return examples, {
                "judge_irrelevant": 0, "judge_error": 0, "noisy_middle": 0,
                "low_confidence": 0, "unknown_correct_tool": 0,
            }

        out_dir = tmp_path / "dataset_out"
        with patch.object(ToolRelevanceFilter, "filter_and_score", fake_filter_and_score):
            dataset, drops = build_tool_dataset_from_sessions(
                manifest=manifest,
                target_tool="search_files",
                output_path=out_dir,
                model="test-model",
                max_examples=10,
            )

        all_examples = dataset.all_examples
        assert len(all_examples) == 2
        categories = {ex.category for ex in all_examples}
        assert categories == {CATEGORY_AGREED, CATEGORY_MISSELECTION}
        for ex in all_examples:
            assert ex.source == "hermes"
        # All-stage drops dict has all 10 keys (5 importer + 5 judge).
        assert set(drops.keys()) >= {
            "short_task", "slash_command", "secret", "no_tool_calls", "non_manifest",
            "judge_irrelevant", "judge_error", "noisy_middle", "low_confidence",
            "unknown_correct_tool",
        }
        # JSONL written.
        assert (out_dir / "train.jsonl").exists() or (out_dir / "val.jsonl").exists() or (out_dir / "holdout.jsonl").exists()

    def test_non_manifest_drops_surface_in_returned_dict(self, manifest, session_dir, mock_dspy_for_filter):
        _write_session(session_dir, "wrong_manifest", [
            {"role": "user", "content": "Use terraform to apply the config"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "tf_apply"}}]},
        ])

        dataset, drops = build_tool_dataset_from_sessions(
            manifest=manifest,
            target_tool="search_files",
            output_path=None,
            model="test-model",
            max_examples=10,
        )
        assert dataset.all_examples == []
        assert drops["non_manifest"] == 1

    def test_no_candidates_returns_importer_drops_only(self, manifest, session_dir, mock_dspy_for_filter):
        # All sessions are too short.
        _write_session(session_dir, "short", [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "search_files"}}]},
        ])

        dataset, drops = build_tool_dataset_from_sessions(
            manifest=manifest,
            target_tool="search_files",
            output_path=None,
            model="test-model",
            max_examples=10,
        )
        assert dataset.all_examples == []
        assert drops["short_task"] == 1
        # Judge-stage keys aren't present when the importer surfaces zero candidates.
        assert "judge_irrelevant" not in drops
