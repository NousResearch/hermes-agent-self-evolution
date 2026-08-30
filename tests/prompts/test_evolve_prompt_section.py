"""Tests for the Phase 3 entry point: gate ladder, holdout statistics, CLI, write-back.

Offline throughout. The optimizer, the harness, and the benchmark runners are
all replaced with stubs, which is the only way to exercise the accept/reject
logic that matters here - the decision about whether a prompt change is allowed
to reach disk is the product, and it has to be testable without an LM.

The holdout tests below are built from hand-chosen score pairs rather than
random data, because the interesting cases are the ones where the point
estimate and the evidence disagree: a +12% lift the test cannot separate from
noise, a category that falls off a cliff behind an improving aggregate, and a
suite too small to say anything at all.
"""

import json

import dspy
import pytest
from click.testing import CliRunner
from rich.console import Console

from evolution.core.gates import GateChain, GateResult, GateStatus
from evolution.prompts import evolve_prompt_section as ep
from evolution.prompts.behavioral_eval import (
    CATEGORY_SECTIONS,
    BehavioralOutcome,
    BehavioralReport,
    BehavioralSuite,
)
from evolution.prompts.sections import (
    CACHE_BLOCK_TOKENS,
    ActiveSessionReport,
    EvolvableSection,
    SectionInventory,
    constant_strings,
    load_sections,
)

PROMPT_SOURCE = '''"""Prompt builder."""

CONTEXT_FILE_MAX_CHARS = 20_000

DEFAULT_AGENT_IDENTITY = (
    "You are Hermes Agent, created by Nous Research. You are helpful and "
    "direct, and you admit uncertainty rather than guessing."
)

MEMORY_GUIDANCE = (
    "You have persistent memory across sessions. Save durable facts with the "
    "memory tool: preferences, environment details, stable conventions.\\n"
    "Do NOT save PR numbers, commit SHAs, or completed-work logs."
)

SESSION_SEARCH_GUIDANCE = (
    "When the user references a past conversation, use session_search first."
)

SKILLS_GUIDANCE = (
    "Save a non-trivial workflow as a skill. Patch an outdated skill on sight."
)

KANBAN_GUIDANCE = (
    "# Kanban protocol\\n"
    "Leave me alone."
)

PLATFORM_HINTS = {
    "cli": "You are a CLI AI Agent. Try not to use markdown.",
}
'''


@pytest.fixture(autouse=True)
def restore_dspy_settings():
    """evolve() configures a global default LM. Put the old one back.

    No LM is ever called here, but leaving a configured default behind would
    change how unrelated test modules behave depending on run order.
    """
    previous = dspy.settings.lm
    yield
    dspy.configure(lm=previous)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "prompt_builder.py").write_text(PROMPT_SOURCE, encoding="utf-8")
    (tmp_path / "batch_runner.py").write_text("# fake runner\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def builder(repo):
    return repo / "agent" / "prompt_builder.py"


def gate(name, status, message="", score=None, baseline=None):
    return GateResult(name=name, status=status, message=message, score=score, baseline=baseline)


def flat(output: str) -> str:
    """Collapse rich's console wrapping so assertions can span line breaks."""
    return " ".join(output.split())


def holdout_reports(categories, baseline_scores, candidate_scores):
    """Two reports over the same scenarios in the same order.

    Scenario ids come from the position, which is what the paired comparison
    checks: index i has to mean the same scenario on both sides.
    """
    ids = [f"s{i:02d}" for i in range(len(categories))]

    def build(scores):
        return BehavioralReport(
            outcomes=[
                BehavioralOutcome(
                    scenario_id=scenario_id,
                    category=category,
                    section=CATEGORY_SECTIONS.get(category, ""),
                    score=score,
                    passed=score >= 0.6,
                )
                for scenario_id, category, score in zip(ids, categories, scores)
            ]
        )

    return build(baseline_scores), build(candidate_scores)


def uniform_holdout(spec, base=0.40):
    """Reports where every scenario in a category moves by the same amount.

    *spec* maps category name to the delta applied to that category's scenarios,
    with the count taken from a ``(delta, count)`` pair.
    """
    categories, baseline, candidate = [], [], []
    for category, (delta, count) in spec.items():
        for _ in range(count):
            categories.append(category)
            baseline.append(base)
            candidate.append(base + delta)
    return holdout_reports(categories, baseline, candidate)


# ──────────────────────────────────────────────────────────────────────────
# Gate ladder
# ──────────────────────────────────────────────────────────────────────────


class TestGateLadder:
    def test_zero_tolerance_is_the_phase_default(self):
        assert ep.ZERO_REGRESSION_TOLERANCE == 0.0

    def test_unavailable_benchmarks_pass_permissively(self, repo, monkeypatch):
        monkeypatch.setattr(
            ep,
            "run_benchmark_gate",
            lambda *a, **k: gate(a[1], GateStatus.UNAVAILABLE, "not found"),
        )
        monkeypatch.setattr(
            ep, "run_pytest_gate", lambda *a, **k: gate("pytest", GateStatus.PASSED, "ok")
        )
        chain = ep.run_gate_ladder(repo, {"MEMORY_GUIDANCE": "Short guidance."})
        assert chain.passed
        assert [r.status for r in chain.results][1:] == [
            GateStatus.UNAVAILABLE,
            GateStatus.UNAVAILABLE,
        ]

    def test_strict_mode_blocks_on_an_unavailable_benchmark(self, repo, monkeypatch):
        monkeypatch.setattr(
            ep,
            "run_benchmark_gate",
            lambda *a, **k: gate(a[1], GateStatus.UNAVAILABLE, "not found"),
        )
        monkeypatch.setattr(
            ep, "run_pytest_gate", lambda *a, **k: gate("pytest", GateStatus.PASSED, "ok")
        )
        chain = ep.run_gate_ladder(
            repo, {"MEMORY_GUIDANCE": "Short guidance."}, strict=True
        )
        assert not chain.passed
        assert chain.blockers[0].status is GateStatus.UNAVAILABLE

    def test_zero_tolerance_rejects_a_small_regression(self, repo, monkeypatch):
        calls = {"n": 0}

        def fake_benchmark(repo_arg, name, baseline=None, regression_threshold=0.02, **kw):
            calls["n"] += 1
            if baseline is None:
                return gate(name, GateStatus.PASSED, "baseline", score=0.80)
            score = 0.79
            if score - baseline < -abs(regression_threshold):
                return gate(name, GateStatus.FAILED, "regressed", score=score, baseline=baseline)
            return gate(name, GateStatus.PASSED, "held", score=score, baseline=baseline)

        monkeypatch.setattr(ep, "run_benchmark_gate", fake_benchmark)
        monkeypatch.setattr(
            ep, "run_pytest_gate", lambda *a, **k: gate("pytest", GateStatus.PASSED, "ok")
        )
        chain = ep.run_gate_ladder(repo, {"MEMORY_GUIDANCE": "Short."})
        assert not chain.passed
        assert chain.blockers[0].name == "tblite"

    def test_a_one_percent_drop_would_pass_at_the_old_two_percent_tolerance(
        self, repo, monkeypatch
    ):
        def fake_benchmark(repo_arg, name, baseline=None, regression_threshold=0.02, **kw):
            if baseline is None:
                return gate(name, GateStatus.PASSED, "baseline", score=0.80)
            score = 0.79
            if score - baseline < -abs(regression_threshold):
                return gate(name, GateStatus.FAILED, "regressed", score=score)
            return gate(name, GateStatus.PASSED, "held", score=score)

        monkeypatch.setattr(ep, "run_benchmark_gate", fake_benchmark)
        monkeypatch.setattr(
            ep, "run_pytest_gate", lambda *a, **k: gate("pytest", GateStatus.PASSED, "ok")
        )
        chain = ep.run_gate_ladder(
            repo, {"MEMORY_GUIDANCE": "Short."}, regression_threshold=0.02
        )
        assert chain.passed

    def test_a_failing_pytest_gate_stops_the_ladder(self, repo, monkeypatch):
        monkeypatch.setattr(
            ep, "run_pytest_gate", lambda *a, **k: gate("pytest", GateStatus.FAILED, "3 failed")
        )
        monkeypatch.setattr(
            ep,
            "run_benchmark_gate",
            lambda *a, **k: pytest.fail("benchmarks must not run after pytest fails"),
        )
        chain = ep.run_gate_ladder(
            repo, {"MEMORY_GUIDANCE": "Short."}, benchmarks=()
        )
        assert not chain.passed

    def test_the_candidate_is_staged_while_gates_run(self, repo, builder, monkeypatch):
        seen = {}

        def spy_pytest(repo_arg, **kwargs):
            seen["source"] = builder.read_text(encoding="utf-8")
            return gate("pytest", GateStatus.PASSED, "ok")

        monkeypatch.setattr(ep, "run_pytest_gate", spy_pytest)
        monkeypatch.setattr(
            ep, "run_benchmark_gate", lambda *a, **k: gate(a[1], GateStatus.UNAVAILABLE, "-")
        )
        original = builder.read_text(encoding="utf-8")
        ep.run_gate_ladder(repo, {"MEMORY_GUIDANCE": "STAGED CANDIDATE TEXT."})
        assert "STAGED CANDIDATE TEXT." in seen["source"]
        assert builder.read_text(encoding="utf-8") == original

    def test_a_backup_is_left_behind(self, repo, builder, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ep, "run_pytest_gate", lambda *a, **k: gate("pytest", GateStatus.PASSED, "ok")
        )
        monkeypatch.setattr(
            ep, "run_benchmark_gate", lambda *a, **k: gate(a[1], GateStatus.UNAVAILABLE, "-")
        )
        backup = tmp_path / "out" / "prompt_builder.py.bak"
        original = builder.read_text(encoding="utf-8")
        ep.run_gate_ladder(repo, {"MEMORY_GUIDANCE": "STAGED."}, backup_path=backup)
        assert backup.read_text(encoding="utf-8") == original

    def test_pytest_gate_can_be_skipped(self, repo, monkeypatch):
        monkeypatch.setattr(
            ep, "run_pytest_gate", lambda *a, **k: pytest.fail("pytest should be skipped")
        )
        monkeypatch.setattr(
            ep, "run_benchmark_gate", lambda *a, **k: gate(a[1], GateStatus.UNAVAILABLE, "-")
        )
        chain = ep.run_gate_ladder(repo, {"MEMORY_GUIDANCE": "S."}, run_pytest=False)
        assert chain.passed and len(chain.results) == 2

    def test_default_pytest_subset_is_narrow(self):
        assert "-k" in ep.DEFAULT_PYTEST_SUBSET
        assert "prompt" in ep.DEFAULT_PYTEST_SUBSET


# ──────────────────────────────────────────────────────────────────────────
# SectionOutcome bookkeeping
# ──────────────────────────────────────────────────────────────────────────


class TestSectionOutcome:
    def test_growth_and_improvement_maths(self):
        outcome = ep.SectionOutcome(
            name="MEMORY_GUIDANCE",
            baseline_text="y" * 100,
            evolved_text="y" * 110,
            baseline_score=0.4,
            evolved_score=0.6,
        )
        assert outcome.growth == pytest.approx(0.10)
        assert outcome.improvement == pytest.approx(0.2)
        assert outcome.holdout_improvement is None

    def test_holdout_improvement_when_measured(self):
        outcome = ep.SectionOutcome(
            name="M", baseline_text="a", evolved_text="b",
            holdout_baseline=0.5, holdout_evolved=0.75,
        )
        assert outcome.holdout_improvement == pytest.approx(0.25)

    def test_to_dict_is_json_serialisable(self):
        outcome = ep.SectionOutcome(name="M", baseline_text="a", evolved_text="bb")
        json.dumps(outcome.to_dict())


# ──────────────────────────────────────────────────────────────────────────
# CLI argument handling
# ──────────────────────────────────────────────────────────────────────────


class TestCli:
    def test_dry_run_touches_nothing(self, repo, builder):
        before = builder.read_text(encoding="utf-8")
        result = CliRunner().invoke(
            ep.main,
            ["--section", "MEMORY_GUIDANCE", "--hermes-repo", str(repo), "--dry-run"],
        )
        assert result.exit_code == 0, result.output
        assert "Dry run" in flat(result.output)
        assert builder.read_text(encoding="utf-8") == before

    def test_dry_run_reports_the_zero_tolerance_setting(self, repo):
        result = CliRunner().invoke(
            ep.main,
            ["--all-sections", "--hermes-repo", str(repo), "--dry-run"],
        )
        assert "0% regression tolerance" in flat(result.output)

    def test_dry_run_states_the_acceptance_rule_and_the_power(self, repo):
        result = CliRunner().invoke(
            ep.main, ["--all-sections", "--hermes-repo", str(repo), "--dry-run"]
        )
        output = flat(result.output)
        assert "significant paired improvement of at least 10%" in output
        assert "no category regression past 5%" in output
        assert "Holdout power:" in output

    def test_dry_run_warns_when_the_holdout_is_too_small_to_test(self, repo, monkeypatch):
        monkeypatch.setattr(
            BehavioralSuite, "split", lambda self, **kwargs: ([], [], self.scenarios[:2])
        )
        result = CliRunner().invoke(
            ep.main, ["--section", "MEMORY_GUIDANCE", "--hermes-repo", str(repo), "--dry-run"]
        )
        assert "Underpowered" in flat(result.output)

    def test_dry_run_states_the_next_session_rule(self, repo):
        result = CliRunner().invoke(
            ep.main, ["--all-sections", "--hermes-repo", str(repo), "--dry-run"]
        )
        assert "NEXT session" in flat(result.output)

    def test_platform_hints_is_refused_with_an_explanation(self, repo):
        result = CliRunner().invoke(
            ep.main,
            ["--section", "PLATFORM_HINTS", "--hermes-repo", str(repo), "--dry-run"],
        )
        assert result.exit_code == 1
        assert "not an evolvable prompt section" in flat(result.output)
        assert "not a plain string" in flat(result.output)

    def test_unknown_section_is_refused(self, repo):
        result = CliRunner().invoke(
            ep.main,
            ["--section", "KANBAN_GUIDANCE", "--hermes-repo", str(repo), "--dry-run"],
        )
        assert result.exit_code == 1
        assert "KANBAN_GUIDANCE" in flat(result.output)

    def test_no_target_is_refused(self, repo):
        result = CliRunner().invoke(ep.main, ["--hermes-repo", str(repo), "--dry-run"])
        assert result.exit_code == 1
        assert "--all-sections" in flat(result.output)

    def test_missing_prompt_builder_is_refused(self, tmp_path):
        empty = tmp_path / "empty-repo"
        empty.mkdir()
        result = CliRunner().invoke(
            ep.main, ["--all-sections", "--hermes-repo", str(empty), "--dry-run"]
        )
        assert result.exit_code == 1
        assert "No evolvable prompt sections" in flat(result.output)

    def test_all_sections_lists_every_discovered_section(self, repo):
        result = CliRunner().invoke(
            ep.main, ["--all-sections", "--hermes-repo", str(repo), "--dry-run"]
        )
        for name in (
            "DEFAULT_AGENT_IDENTITY",
            "MEMORY_GUIDANCE",
            "SESSION_SEARCH_GUIDANCE",
            "SKILLS_GUIDANCE",
        ):
            assert name in flat(result.output)

    def test_platform_hints_is_reported_as_out_of_scope(self, repo):
        result = CliRunner().invoke(
            ep.main, ["--all-sections", "--hermes-repo", str(repo), "--dry-run"]
        )
        assert "PLATFORM_HINTS" in flat(result.output)
        assert "out of scope for this phase" in flat(result.output)

    def test_write_defaults_to_off(self):
        params = {p.name: p for p in ep.main.params}
        assert params["write"].default is False


# ──────────────────────────────────────────────────────────────────────────
# Full run with stubbed optimizer, harness, and gates
# ──────────────────────────────────────────────────────────────────────────


EVOLVED_MARKER = "Prefer facts that survive a week."


@pytest.fixture
def stubbed(monkeypatch):
    """Replace everything that would need a model or a benchmark."""

    def fake_optimize(section_name, baseline_text, trainset, valset, iterations, optimizer_model):
        return f"{baseline_text} {EVOLVED_MARKER}", "stub"

    def fake_evaluate(self, system_prompt, harness, judge=None, section_name="", run_name="", scenarios=None):
        targets = list(scenarios if scenarios is not None else self.scenarios)
        score = 0.9 if EVOLVED_MARKER in system_prompt else 0.4
        return BehavioralReport(
            outcomes=[
                BehavioralOutcome(
                    scenario_id=s.scenario_id,
                    category=s.category,
                    section=s.section_under_test,
                    score=score,
                    passed=score >= 0.6,
                )
                for s in targets
            ],
            harness="stub",
        )

    def fake_gates(**kwargs):
        return GateChain(strict=kwargs.get("strict", False)).run(
            gate("pytest", GateStatus.PASSED, "ok")
        )

    monkeypatch.setattr(ep, "_optimize_section", fake_optimize)
    monkeypatch.setattr(BehavioralSuite, "evaluate", fake_evaluate)
    monkeypatch.setattr(ep, "run_gate_ladder", fake_gates)
    monkeypatch.setattr(
        ep, "detect_active_session", lambda *a, **k: ActiveSessionReport(active=False)
    )
    return monkeypatch


class TestFullRun:
    def test_no_write_by_default(self, repo, builder, stubbed, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        before = builder.read_text(encoding="utf-8")
        code = ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo))
        assert code == 0
        assert builder.read_text(encoding="utf-8") == before

    def test_write_applies_the_evolved_section(self, repo, builder, stubbed, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo), write=True
        )
        assert code == 0
        assert EVOLVED_MARKER in load_sections(repo).get("MEMORY_GUIDANCE").baseline_text

    def test_write_leaves_neighbouring_constants_untouched(
        self, repo, builder, stubbed, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        before = constant_strings(builder.read_text(encoding="utf-8"))
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo), write=True)
        after = constant_strings(builder.read_text(encoding="utf-8"))
        changed = [k for k in before if after[k] != before[k]]
        assert changed == ["MEMORY_GUIDANCE"]
        assert after["KANBAN_GUIDANCE"] == before["KANBAN_GUIDANCE"]
        assert after["DEFAULT_AGENT_IDENTITY"] == before["DEFAULT_AGENT_IDENTITY"]

    def test_an_active_session_refuses_the_write(
        self, repo, builder, stubbed, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            ep,
            "detect_active_session",
            lambda *a, **k: ActiveSessionReport(
                active=True, evidence=("HERMES_SESSION_ID=abc",)
            ),
        )
        before = builder.read_text(encoding="utf-8")
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo), write=True
        )
        assert code == 2
        assert builder.read_text(encoding="utf-8") == before

    def test_a_blocked_gate_prevents_the_write(
        self, repo, builder, stubbed, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            ep,
            "run_gate_ladder",
            lambda **kwargs: GateChain().run(
                gate("tblite", GateStatus.FAILED, "regressed -3%")
            ),
        )
        before = builder.read_text(encoding="utf-8")
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo), write=True
        )
        assert code == 0
        assert builder.read_text(encoding="utf-8") == before

    def test_a_constraint_violation_prevents_the_write(
        self, repo, builder, stubbed, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            ep,
            "_optimize_section",
            lambda section_name, baseline_text, trainset, valset, iterations, optimizer_model: (
                baseline_text * 4,
                "stub",
            ),
        )
        monkeypatch.chdir(tmp_path)
        before = builder.read_text(encoding="utf-8")
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo), write=True
        )
        assert code == 0
        assert builder.read_text(encoding="utf-8") == before

    def test_identity_trait_loss_prevents_the_write(
        self, repo, builder, stubbed, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            ep,
            "_optimize_section",
            lambda section_name, baseline_text, trainset, valset, iterations, optimizer_model: (
                "You are a bot. Do tasks.",
                "stub",
            ),
        )
        monkeypatch.chdir(tmp_path)
        before = builder.read_text(encoding="utf-8")
        code = ep.evolve(
            section_names=["DEFAULT_AGENT_IDENTITY"], hermes_repo=str(repo), write=True
        )
        assert code == 0
        assert builder.read_text(encoding="utf-8") == before

    def test_metrics_and_artifacts_are_saved(self, repo, stubbed, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo))
        runs = sorted((tmp_path / "output" / "prompts").iterdir())
        assert runs
        metrics = json.loads((runs[-1] / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["regression_threshold"] == 0.0
        assert metrics["sections"][0]["name"] == "MEMORY_GUIDANCE"
        assert (runs[-1] / "evolved_MEMORY_GUIDANCE.txt").exists()
        assert (runs[-1] / "baseline_MEMORY_GUIDANCE.txt").exists()
        assert (runs[-1] / "scenarios.jsonl").exists()

    def _block_crossing_inventory(self, repo):
        """An inventory whose baseline sits one edit away from a new cache block."""
        baseline = "m" * (CACHE_BLOCK_TOKENS * 4 - 100)
        return SectionInventory(
            prompt_builder=repo / "agent" / "prompt_builder.py",
            sections=[
                EvolvableSection(
                    name="MEMORY_GUIDANCE",
                    path=repo / "agent" / "prompt_builder.py",
                    baseline_text=baseline,
                    span=None,
                )
            ],
        )

    def test_strict_mode_rejects_a_cache_block_crossing(
        self, repo, builder, stubbed, tmp_path, monkeypatch
    ):
        inventory = self._block_crossing_inventory(repo)
        monkeypatch.setattr(ep, "load_sections", lambda *a, **k: inventory)
        monkeypatch.setattr(
            ep,
            "_optimize_section",
            lambda section_name, baseline_text, trainset, valset, iterations, optimizer_model: (
                "m" * int(len(baseline_text) * 1.2),
                "stub",
            ),
        )
        monkeypatch.chdir(tmp_path)
        before = builder.read_text(encoding="utf-8")
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"],
            hermes_repo=str(repo),
            write=True,
            strict_gates=True,
        )
        assert code == 0
        assert builder.read_text(encoding="utf-8") == before

    def test_permissive_mode_accepts_the_same_crossing(
        self, repo, builder, stubbed, tmp_path, monkeypatch
    ):
        inventory = self._block_crossing_inventory(repo)
        grown = "m" * int(len(inventory.get("MEMORY_GUIDANCE").baseline_text) * 1.2)
        monkeypatch.setattr(ep, "load_sections", lambda *a, **k: inventory)
        monkeypatch.setattr(
            ep,
            "_optimize_section",
            lambda section_name, baseline_text, trainset, valset, iterations, optimizer_model: (
                grown,
                "stub",
            ),
        )

        def score_by_length(self, system_prompt, harness, judge=None, section_name="", run_name="", scenarios=None):
            targets = list(scenarios if scenarios is not None else self.scenarios)
            score = 0.9 if grown in system_prompt else 0.4
            return BehavioralReport(
                outcomes=[
                    BehavioralOutcome(
                        s.scenario_id, s.category, s.section_under_test, score, score >= 0.6
                    )
                    for s in targets
                ]
            )

        monkeypatch.setattr(BehavioralSuite, "evaluate", score_by_length)
        monkeypatch.chdir(tmp_path)
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo), write=True
        )
        assert code == 0
        assert load_sections(repo).get("MEMORY_GUIDANCE").baseline_text == grown

    def test_strict_mode_is_carried_into_the_ladder(self, repo, stubbed, tmp_path, monkeypatch):
        seen = {}

        def spy(**kwargs):
            seen.update(kwargs)
            return GateChain(strict=kwargs["strict"]).run(
                gate("pytest", GateStatus.PASSED, "ok")
            )

        monkeypatch.setattr(ep, "run_gate_ladder", spy)
        monkeypatch.chdir(tmp_path)
        ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo), strict_gates=True
        )
        assert seen["strict"] is True
        assert set(seen["updates"]) == {"MEMORY_GUIDANCE"}


# ──────────────────────────────────────────────────────────────────────────
# Pairing
# ──────────────────────────────────────────────────────────────────────────


class TestHoldoutPairing:
    def test_aligned_runs_return_index_matched_scores(self):
        base, cand = holdout_reports(
            ["memory_guidance", "identity_tone"], [0.4, 0.5], [0.9, 0.3]
        )
        ids, categories, baseline, candidate = ep.align_holdout_scores(base, cand)
        assert ids == ("s00", "s01")
        assert categories == ("memory_guidance", "identity_tone")
        assert baseline == (0.4, 0.5)
        assert candidate == (0.9, 0.3)

    def test_a_different_length_is_a_bug_not_a_fallback(self):
        base, _ = holdout_reports(["memory_guidance"] * 3, [0.4] * 3, [0.9] * 3)
        _, short = holdout_reports(["memory_guidance"] * 2, [0.4] * 2, [0.9] * 2)
        with pytest.raises(ep.UnpairedHoldout) as excinfo:
            ep.align_holdout_scores(base, short)
        assert "3 and 2" in str(excinfo.value)

    def test_a_reordered_run_is_refused(self):
        base, cand = holdout_reports(
            ["memory_guidance"] * 3, [0.4, 0.5, 0.6], [0.9, 0.8, 0.7]
        )
        cand.outcomes.reverse()
        with pytest.raises(ep.UnpairedHoldout) as excinfo:
            ep.align_holdout_scores(base, cand)
        assert "same scenario order" in str(excinfo.value)

    def test_compare_holdout_propagates_the_pairing_error(self):
        base, _ = holdout_reports(["memory_guidance"] * 3, [0.4] * 3, [0.9] * 3)
        _, short = holdout_reports(["memory_guidance"] * 2, [0.4] * 2, [0.9] * 2)
        with pytest.raises(ep.UnpairedHoldout):
            ep.compare_holdout(base, short)


# ──────────────────────────────────────────────────────────────────────────
# Power
# ──────────────────────────────────────────────────────────────────────────


class TestPower:
    def test_six_scenarios_is_the_floor_at_alpha_five_percent(self):
        """The floor assumes the friendliest possible data: every scenario
        moving the same way by the same amount.

        Six, not four. With every pair moving the same way the exact
        signed-rank p is 2 * 0.5**n, so four pairs give 0.125 and six give
        0.031. The normal approximation reported 0.046 at n = 4 and would have
        deployed a prompt on it.
        """
        assert ep.min_scenarios_for_significance(0.05) == 6
        assert 2 * 0.5 ** 6 < 0.05 <= 2 * 0.5 ** 5

    def test_a_stricter_alpha_needs_a_bigger_suite(self):
        assert ep.min_scenarios_for_significance(0.01) > ep.min_scenarios_for_significance(0.05)

    def test_a_perfect_lift_on_three_scenarios_cannot_be_significant(self):
        """The floor is a fact about the test, not about the candidate."""
        base, cand = uniform_holdout({"memory_guidance": (0.60, 3)})
        comparison = ep.compare_holdout(base, cand)
        assert comparison.delta == pytest.approx(0.60)
        assert not comparison.overall.significant_improvement
        assert comparison.underpowered
        assert not comparison.accepted

    def test_the_floor_only_holds_when_every_scenario_agrees(self):
        """Reaching the floor needs unanimity, not just size.

        Under the exact signed-rank null the magnitudes are irrelevant once
        every pair moves the same way: p is 2 * 0.5**n whatever the sizes. So
        six unanimous scenarios clear alpha, and six scenarios where one moves
        against the rest do not, at 0.44.
        """
        floor = ep.min_scenarios_for_significance(0.05)

        unanimous = ep.compare_holdout(
            *holdout_reports(
                ["memory_guidance"] * floor,
                [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
                [0.20, 0.50, 0.80, 1.00, 0.55, 0.62],
            )
        )
        assert unanimous.n == floor
        assert unanimous.overall.significant_improvement

        one_dissenter = ep.compare_holdout(
            *holdout_reports(
                ["memory_guidance"] * floor,
                [0.10, 0.20, 0.30, 0.40, 0.50, 0.90],
                [0.20, 0.30, 0.40, 0.50, 0.60, 0.20],
            )
        )
        assert one_dissenter.n == floor
        assert not one_dissenter.overall.significant_improvement

    def test_an_underpowered_run_says_how_many_more_scenarios_it_needs(self):
        base, cand = uniform_holdout({"memory_guidance": (0.60, 3)})
        comparison = ep.compare_holdout(base, cand)
        assert comparison.scenarios_needed == 3
        assert "3 more holdout scenario" in comparison.reason
        assert "sample size talking" in comparison.power_note

    def test_a_powered_run_is_not_flagged_underpowered(self):
        base, cand = uniform_holdout({"memory_guidance": (0.30, 8)})
        comparison = ep.compare_holdout(base, cand)
        assert not comparison.underpowered
        assert comparison.scenarios_needed == 0

    def test_the_smallest_detectable_shift_shrinks_as_the_suite_grows(self):
        small, _ = uniform_holdout({"memory_guidance": (0.0, 6)})
        big, _ = uniform_holdout({"memory_guidance": (0.0, 30)})
        smaller = ep.compare_holdout(small, small).min_detectable_shift
        bigger = ep.compare_holdout(big, big).min_detectable_shift
        assert smaller == pytest.approx(5 / 6)
        assert bigger == pytest.approx(5 / 30)

    def test_fifteen_scenarios_can_only_detect_a_third_of_the_suite_moving(self):
        base, cand = uniform_holdout({"memory_guidance": (0.20, 15)})
        comparison = ep.compare_holdout(base, cand)
        assert comparison.n == 15
        assert comparison.min_detectable_shift == pytest.approx(1 / 3)
        assert "33%" in comparison.power_note


# ──────────────────────────────────────────────────────────────────────────
# Aggregate verdict
# ──────────────────────────────────────────────────────────────────────────


class TestAggregateVerdict:
    def test_a_consistent_lift_clears_both_bars(self):
        base, cand = uniform_holdout({"memory_guidance": (0.30, 10)})
        comparison = ep.compare_holdout(base, cand, targeted_category="memory_guidance")
        assert comparison.overall.significant_improvement
        assert comparison.delta == pytest.approx(0.30)
        assert comparison.accepted
        assert comparison.headline == "accepted"

    def test_a_twelve_percent_lift_that_is_noise_is_not_written_back(self):
        """The case the audit was about: a big-looking delta with no evidence."""
        categories = ["memory_guidance"] * 15
        baseline = [0.40] * 15
        candidate = [0.90] * 8 + [0.40 - 2.2 / 7] * 7
        base, cand = holdout_reports(categories, baseline, candidate)
        comparison = ep.compare_holdout(base, cand, targeted_category="memory_guidance")

        assert comparison.delta == pytest.approx(0.12, abs=1e-9)
        assert comparison.overall.wilcoxon_p > 0.05
        assert not comparison.overall.significant_improvement
        assert not comparison.accepted
        assert "inconclusive" in comparison.reason
        assert "+12.0%" in comparison.reason

    def test_an_inconclusive_result_with_enough_scenarios_blames_the_direction(self):
        categories = ["memory_guidance"] * 15
        candidate = [0.90] * 8 + [0.40 - 2.2 / 7] * 7
        base, cand = holdout_reports(categories, [0.40] * 15, candidate)
        comparison = ep.compare_holdout(base, cand)
        assert comparison.scenarios_needed == 0
        assert "8 improved / 7 regressed" in comparison.reason

    def test_a_significant_but_tiny_lift_is_refused(self):
        base, cand = uniform_holdout({"memory_guidance": (0.02, 15)})
        comparison = ep.compare_holdout(base, cand, targeted_category="memory_guidance")
        assert comparison.overall.significant_improvement
        assert not comparison.accepted
        assert "under the 10% practical threshold" in comparison.reason
        assert comparison.headline == "under the 10% bar"

    def test_a_significant_regression_is_never_accepted(self):
        base, cand = uniform_holdout({"memory_guidance": (-0.30, 10)})
        comparison = ep.compare_holdout(base, cand, targeted_category="memory_guidance")
        assert comparison.overall.significant_regression
        assert not comparison.accepted

    def test_a_flat_targeted_category_blocks_an_aggregate_win(self):
        """PLAN.md asks for >=10% on the section being evolved, not somewhere else."""
        base, cand = uniform_holdout(
            {"memory_guidance": (0.02, 5), "platform_formatting": (0.50, 10)}
        )
        comparison = ep.compare_holdout(base, cand, targeted_category="memory_guidance")
        assert comparison.overall.significant_improvement
        assert comparison.delta > ep.PRACTICAL_IMPROVEMENT
        assert not comparison.accepted
        assert "targeted category memory_guidance" in comparison.reason

    def test_an_empty_holdout_is_not_evidence(self):
        base, cand = holdout_reports([], [], [])
        comparison = ep.compare_holdout(base, cand)
        assert comparison.n == 0
        assert not comparison.accepted
        assert comparison.reason == "no holdout evidence"


# ──────────────────────────────────────────────────────────────────────────
# Per-category guard
# ──────────────────────────────────────────────────────────────────────────


class TestCategoryGuard:
    def test_one_wrecked_category_sinks_an_improving_aggregate(self):
        """Six wrecked scenarios: the rejection is carried by the test itself.

        Six is the floor for the exact signed-rank null, so this is the
        smallest category that can be refused on evidence rather than on the
        tolerance alone.
        """
        base, cand = uniform_holdout(
            {"memory_guidance": (0.50, 10), "platform_formatting": (-0.20, 6)}
        )
        comparison = ep.compare_holdout(base, cand, targeted_category="memory_guidance")

        assert comparison.overall.significant_improvement
        assert comparison.delta > ep.PRACTICAL_IMPROVEMENT
        assert comparison.by_category["platform_formatting"].significant_regression
        assert comparison.regressed_categories == ("platform_formatting",)
        assert not comparison.accepted
        assert "category regression" in comparison.reason
        assert comparison.headline == "regressed: platform_formatting"

    def test_five_wrecked_scenarios_are_refused_on_the_tolerance_alone(self):
        """One below the floor the exact test cannot convict, so the point
        estimate has to. The candidate is refused either way, which is the
        whole reason the rule is a disjunction."""
        base, cand = uniform_holdout(
            {"memory_guidance": (0.50, 10), "platform_formatting": (-0.20, 5)}
        )
        comparison = ep.compare_holdout(base, cand, targeted_category="memory_guidance")
        platform = comparison.by_category["platform_formatting"]

        assert platform.wilcoxon_p == pytest.approx(2 * 0.5 ** 5)
        assert not platform.significant_regression
        assert comparison.regressed_categories == ("platform_formatting",)
        assert not comparison.accepted

    def test_a_point_estimate_breach_is_refused_without_significance(self):
        """Conservative gate: either kind of evidence is enough to reject."""
        base, cand = uniform_holdout(
            {"memory_guidance": (0.50, 12), "platform_formatting": (-0.30, 3)}
        )
        comparison = ep.compare_holdout(base, cand, targeted_category="memory_guidance")
        platform = comparison.by_category["platform_formatting"]

        assert not platform.significant_regression  # 3 scenarios cannot prove it
        assert platform.delta <= -ep.CATEGORY_REGRESSION_TOLERANCE
        assert comparison.regressed_categories == ("platform_formatting",)
        assert not comparison.accepted

    def test_a_small_significant_drop_is_refused_too(self):
        base, cand = uniform_holdout(
            {"memory_guidance": (0.50, 12), "identity_tone": (-0.06, 8)}
        )
        comparison = ep.compare_holdout(base, cand, targeted_category="memory_guidance")
        identity = comparison.by_category["identity_tone"]

        assert identity.significant_regression
        assert comparison.regressed_categories == ("identity_tone",)

    def test_drift_inside_the_tolerance_is_allowed(self):
        base, cand = uniform_holdout(
            {"memory_guidance": (0.50, 12), "platform_formatting": (-0.02, 3)}
        )
        comparison = ep.compare_holdout(base, cand, targeted_category="memory_guidance")
        assert comparison.regressed_categories == ()
        assert comparison.accepted

    def test_a_category_is_measured_only_against_its_own_scenarios(self):
        base, cand = uniform_holdout(
            {"memory_guidance": (0.50, 6), "skills_guidance": (0.10, 6)}
        )
        comparison = ep.compare_holdout(base, cand)
        assert comparison.by_category["memory_guidance"].n == 6
        assert comparison.by_category["memory_guidance"].delta == pytest.approx(0.50)
        assert comparison.by_category["skills_guidance"].delta == pytest.approx(0.10)

    def test_no_multiplicity_correction_as_the_category_count_grows(self):
        """Intersection-union: the per-category bar does not move with tool count.

        A Bonferroni-style correction would divide alpha by the number of
        categories, so the same wrecked category would look less and less
        significant the more categories a run covered. That is backwards for a
        safety gate, and this pins the behaviour.
        """
        offender = {"platform_formatting": (-0.20, 6)}
        two = ep.compare_holdout(
            *uniform_holdout({"memory_guidance": (0.50, 6), **offender}),
            targeted_category="memory_guidance",
        )
        five = ep.compare_holdout(
            *uniform_holdout(
                {
                    "memory_guidance": (0.50, 6),
                    "skills_guidance": (0.50, 6),
                    "session_search": (0.50, 6),
                    "identity_tone": (0.50, 6),
                    **offender,
                }
            ),
            targeted_category="memory_guidance",
        )

        assert two.by_category["platform_formatting"].wilcoxon_p == pytest.approx(
            five.by_category["platform_formatting"].wilcoxon_p
        )
        assert two.regressed_categories == five.regressed_categories
        assert not two.accepted and not five.accepted

    def test_underpowered_categories_are_named(self):
        base, cand = uniform_holdout(
            {"memory_guidance": (0.50, 10), "platform_formatting": (0.01, 3)}
        )
        comparison = ep.compare_holdout(base, cand, targeted_category="memory_guidance")
        assert comparison.underpowered_categories == ("platform_formatting",)
        assert not comparison.underpowered  # the aggregate is big enough


# ──────────────────────────────────────────────────────────────────────────
# Serialisation
# ──────────────────────────────────────────────────────────────────────────


class TestHoldoutSerialisation:
    def test_to_dict_carries_the_whole_comparison(self):
        base, cand = uniform_holdout(
            {"memory_guidance": (0.50, 10), "platform_formatting": (-0.20, 5)}
        )
        data = ep.compare_holdout(
            base, cand, targeted_category="memory_guidance"
        ).to_dict()

        json.dumps(data)
        assert data["n"] == 15
        assert data["overall"]["wilcoxon_p"] >= 0
        assert data["overall"]["delta_ci"]["confidence"] == 0.95
        assert set(data["by_category"]) == {"memory_guidance", "platform_formatting"}
        assert data["targeted"]["delta"] == pytest.approx(0.50)
        assert data["regressed_categories"] == ["platform_formatting"]
        assert data["min_scenarios_for_significance"] == 6
        assert data["movement"] == {"improved": 10, "regressed": 5, "unchanged": 0}
        assert data["accepted"] is False

    def test_the_bootstrap_is_reproducible(self):
        base, cand = uniform_holdout(
            {"memory_guidance": (0.50, 6), "platform_formatting": (-0.10, 6)}
        )
        first = ep.compare_holdout(base, cand).to_dict()
        second = ep.compare_holdout(base, cand).to_dict()
        assert first == second

    def test_section_outcome_carries_the_comparison(self):
        base, cand = uniform_holdout({"memory_guidance": (0.30, 10)})
        outcome = ep.SectionOutcome(
            name="MEMORY_GUIDANCE",
            baseline_text="a",
            evolved_text="b",
            holdout=ep.compare_holdout(base, cand, targeted_category="memory_guidance"),
        )
        data = outcome.to_dict()
        json.dumps(data)
        assert data["holdout"]["accepted"] is True


# ──────────────────────────────────────────────────────────────────────────
# The holdout inside a full run
# ──────────────────────────────────────────────────────────────────────────


def scoring_stub(deltas_by_category, base=0.40):
    """A BehavioralSuite.evaluate replacement with per-category deltas."""

    def _evaluate(
        self, system_prompt, harness, judge=None, section_name="", run_name="", scenarios=None
    ):
        targets = list(scenarios if scenarios is not None else self.scenarios)
        evolved = EVOLVED_MARKER in system_prompt
        outcomes = []
        for scenario in targets:
            delta = deltas_by_category.get(scenario.category, 0.0) if evolved else 0.0
            score = base + delta
            outcomes.append(
                BehavioralOutcome(
                    scenario_id=scenario.scenario_id,
                    category=scenario.category,
                    section=scenario.section_under_test,
                    score=score,
                    passed=score >= 0.6,
                )
            )
        return BehavioralReport(outcomes=outcomes, harness="stub")

    return _evaluate


def alternating_stub(deltas, base=0.40):
    """Per-position deltas, cycling, so a run can be made deliberately noisy."""

    def _evaluate(
        self, system_prompt, harness, judge=None, section_name="", run_name="", scenarios=None
    ):
        targets = list(scenarios if scenarios is not None else self.scenarios)
        evolved = EVOLVED_MARKER in system_prompt
        outcomes = []
        for index, scenario in enumerate(targets):
            score = base + (deltas[index % len(deltas)] if evolved else 0.0)
            outcomes.append(
                BehavioralOutcome(
                    scenario_id=scenario.scenario_id,
                    category=scenario.category,
                    section=scenario.section_under_test,
                    score=score,
                    passed=score >= 0.6,
                )
            )
        return BehavioralReport(outcomes=outcomes, harness="stub")

    return _evaluate


def latest_metrics(tmp_path):
    runs = sorted((tmp_path / "output" / "prompts").iterdir())
    return json.loads((runs[-1] / "metrics.json").read_text(encoding="utf-8"))


@pytest.fixture
def wide_console(monkeypatch):
    """Stop rich wrapping table headers mid-word at the default 80 columns.

    Only the assertions need this. The run itself does not care how wide the
    terminal is.
    """
    monkeypatch.setattr(ep, "console", Console(width=200))


def same_length_optimize(
    section_name, baseline_text, trainset, valset, iterations, optimizer_model
):
    """An optimizer stub that marks a section without growing it.

    The default stub appends to the baseline, which pushes a short section past
    the +20% growth ceiling and quietly drops it before the holdout ever runs.
    """
    keep = max(0, len(baseline_text) - len(EVOLVED_MARKER))
    return baseline_text[:keep] + EVOLVED_MARKER, "stub"


class TestFullRunHoldout:
    def test_metrics_record_the_full_comparison(self, repo, stubbed, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo))
        holdout = latest_metrics(tmp_path)["sections"][0]["holdout"]

        assert holdout["n"] > 0
        assert "wilcoxon_p" in holdout["overall"]
        assert "delta_ci" in holdout["overall"]
        assert holdout["by_category"]
        assert holdout["targeted_category"] == "memory_guidance"
        assert holdout["accepted"] is True

    def test_metrics_record_the_test_configuration(self, repo, stubbed, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo))
        config = latest_metrics(tmp_path)["holdout_test"]

        assert config["alpha"] == 0.05
        assert config["practical_threshold"] == 0.10
        assert config["bootstrap_seed"] == ep.HOLDOUT_BOOTSTRAP_SEED
        assert "intersection-union" in config["multiplicity_correction"]

    def test_an_inconclusive_holdout_prevents_the_write(
        self, repo, builder, stubbed, tmp_path, monkeypatch
    ):
        # +12% on the mean, with the direction flipping scenario to scenario.
        monkeypatch.setattr(
            BehavioralSuite, "evaluate", alternating_stub([0.60, -0.36])
        )
        monkeypatch.chdir(tmp_path)
        before = builder.read_text(encoding="utf-8")
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo), write=True
        )

        assert code == 0
        assert builder.read_text(encoding="utf-8") == before
        holdout = latest_metrics(tmp_path)["sections"][0]["holdout"]
        assert holdout["overall"]["delta"] == pytest.approx(0.12)
        assert holdout["accepted"] is False
        assert "inconclusive" in holdout["reason"]

    def test_the_inconclusive_verdict_reaches_the_console(
        self, repo, stubbed, wide_console, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            BehavioralSuite, "evaluate", alternating_stub([0.60, -0.36])
        )
        monkeypatch.chdir(tmp_path)
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo))
        output = flat(capsys.readouterr().out)

        assert "inconclusive" in output
        assert "not distinguishable from noise" in output

    def test_a_wrecked_category_prevents_the_write(
        self, repo, builder, stubbed, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            BehavioralSuite,
            "evaluate",
            scoring_stub({"memory_guidance": 0.50, "platform_formatting": -0.30}),
        )
        monkeypatch.chdir(tmp_path)
        before = builder.read_text(encoding="utf-8")
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo), write=True
        )

        assert code == 0
        assert builder.read_text(encoding="utf-8") == before
        holdout = latest_metrics(tmp_path)["sections"][0]["holdout"]
        assert holdout["regressed_categories"] == ["platform_formatting"]
        assert holdout["by_category"]["memory_guidance"]["delta"] == pytest.approx(0.50)

    def test_the_results_table_reports_the_interval_and_the_p_value(
        self, repo, stubbed, wide_console, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo))
        output = flat(capsys.readouterr().out)

        assert "Δ (95% CI)" in output
        assert "Holdout by category" in output
        assert "targeted" in output

    def test_power_is_stated_even_on_a_clean_run(
        self, repo, stubbed, wide_console, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo))
        output = flat(capsys.readouterr().out)

        assert "Power:" in output
        assert "smallest shift an exact paired test could call significant" in output

    def test_the_holdout_covers_categories_the_section_was_not_aimed_at(
        self, repo, stubbed, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo))
        holdout = latest_metrics(tmp_path)["sections"][0]["holdout"]
        assert set(holdout["by_category"]) == {"memory_guidance", "platform_formatting"}

    def test_one_baseline_run_is_shared_by_every_candidate(
        self, repo, stubbed, tmp_path, monkeypatch
    ):
        """Two candidates, one baseline. Re-measuring it per section would give
        each candidate a differently sampled baseline and break the pairing."""
        runs: list[str] = []
        original = BehavioralSuite.evaluate

        def spy(self, system_prompt, harness, judge=None, section_name="", run_name="", scenarios=None):
            runs.append(run_name)
            return original(
                self,
                system_prompt,
                harness,
                judge=judge,
                section_name=section_name,
                run_name=run_name,
                scenarios=scenarios,
            )

        monkeypatch.setattr(BehavioralSuite, "evaluate", spy)
        monkeypatch.setattr(ep, "_optimize_section", same_length_optimize)
        monkeypatch.chdir(tmp_path)
        ep.evolve(
            section_names=["MEMORY_GUIDANCE", "SKILLS_GUIDANCE"], hermes_repo=str(repo)
        )

        assert len([r for r in runs if r.startswith("hase-holdout-base")]) == 1
        assert len([r for r in runs if r.startswith("hase-holdout-evolved")]) == 2

    def test_both_sides_of_the_pair_get_the_same_section_label(
        self, repo, stubbed, tmp_path, monkeypatch
    ):
        """The label reaches the direct harness's scaffold, so it must not vary."""
        labels: list[str] = []
        original = BehavioralSuite.evaluate

        def spy(self, system_prompt, harness, judge=None, section_name="", run_name="", scenarios=None):
            if run_name.startswith("hase-holdout"):
                labels.append(section_name)
            return original(
                self,
                system_prompt,
                harness,
                judge=judge,
                section_name=section_name,
                run_name=run_name,
                scenarios=scenarios,
            )

        monkeypatch.setattr(BehavioralSuite, "evaluate", spy)
        monkeypatch.chdir(tmp_path)
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo))

        assert labels == [ep.HOLDOUT_SECTION_LABEL, ep.HOLDOUT_SECTION_LABEL]

    def test_a_flat_holdout_is_not_deployable(
        self, repo, builder, stubbed, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(BehavioralSuite, "evaluate", scoring_stub({}))
        monkeypatch.chdir(tmp_path)
        before = builder.read_text(encoding="utf-8")
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo), write=True
        )
        assert code == 0
        assert builder.read_text(encoding="utf-8") == before


class TestUnmeasuredScenariosAreNotFailures:
    """A transcript the harness never produced is missing data, not a zero.

    Audit finding: scoring an absent run 0.0 on one side of a paired comparison
    and a real score on the other manufactures a difference out of a timeout.
    Six baseline scenarios timing out against a clean candidate run read as a
    significant +32% improvement and deployed.
    """

    def _outcome(self, i, score, measured=True, category="memory_guidance"):
        return BehavioralOutcome(
            scenario_id=f"s{i:02d}",
            category=category,
            section="MEMORY_GUIDANCE",
            score=score,
            passed=score >= 0.5,
            measured=measured,
        )

    def _report(self, outcomes):
        return BehavioralReport(outcomes=outcomes)

    def test_a_harness_flake_cannot_manufacture_an_improvement(self):
        baseline = self._report(
            [self._outcome(i, 0.0, measured=False) for i in range(6)]
            + [self._outcome(i, 0.8) for i in range(6, 15)]
        )
        candidate = self._report([self._outcome(i, 0.8) for i in range(15)])
        with pytest.raises(ep.UnpairedHoldout, match="no transcript"):
            ep.compare_holdout(baseline, candidate)

    def test_a_few_dropped_scenarios_shrink_the_sample_not_the_scores(self):
        baseline = self._report(
            [self._outcome(0, 0.0, measured=False)]
            + [self._outcome(i, 0.5) for i in range(1, 15)]
        )
        candidate = self._report([self._outcome(i, 0.5) for i in range(15)])
        comparison = ep.compare_holdout(baseline, candidate)
        assert comparison.n == 14
        assert comparison.delta == pytest.approx(0.0)
        assert not comparison.accepted

    def test_an_unmeasured_candidate_scenario_drops_the_pair_too(self):
        baseline = self._report([self._outcome(i, 0.5) for i in range(15)])
        candidate = self._report(
            [self._outcome(0, 0.0, measured=False)]
            + [self._outcome(i, 0.9) for i in range(1, 15)]
        )
        comparison = ep.compare_holdout(baseline, candidate)
        assert comparison.n == 14
        assert comparison.delta == pytest.approx(0.4)

    def test_everything_measured_keeps_the_whole_suite(self):
        baseline = self._report([self._outcome(i, 0.5) for i in range(15)])
        candidate = self._report([self._outcome(i, 0.7) for i in range(15)])
        assert ep.compare_holdout(baseline, candidate).n == 15
