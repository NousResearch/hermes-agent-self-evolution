"""Tests for corpus-derived size budgets and A/B reporting.

The 15 KB cap these budgets replace rejected 27 of the 201 installed skills at
their own baseline — the largest is 72 KB — so those skills could never be
evolved at all.

The report tests pin the standard the readtool eval already holds Hermes to:
deltas stated against a noise band, an explicit verdict, and caveats that
appear on their own rather than being remembered.
"""

from __future__ import annotations

import json

import pytest

from evolution.core.corpus import (
    BASELINE_HEADROOM,
    MIN_BUDGET_CHARS,
    CorpusStats,
    derive_size_budget,
    iter_skill_files,
    measure_corpus,
)
from evolution.core.report import ABReport, ArmStats, arm_from_eval_run, arm_from_scores


# ── corpus ──────────────────────────────────────────────────────────────


@pytest.fixture
def skills_tree(tmp_path):
    root = tmp_path / "skills"
    for name, size in [("tiny", 400), ("small", 2_000), ("mid", 6_000), ("big", 30_000)]:
        d = root / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("x" * size)
    return root


class TestMeasureCorpus:
    def test_counts_every_skill(self, skills_tree):
        assert measure_corpus(skills_tree).count == 4

    def test_reports_the_distribution(self, skills_tree):
        stats = measure_corpus(skills_tree)
        assert stats.smallest == 400
        assert stats.largest == 30_000

    def test_empty_tree_measures_to_none(self, tmp_path):
        assert measure_corpus(tmp_path / "nothing") is None

    def test_duplicate_files_across_roots_are_counted_once(self, skills_tree):
        """Repo and user trees overlap; double counting skews the percentile."""
        both = measure_corpus(skills_tree, skills_tree)
        assert both.count == 4

    def test_iteration_is_deduplicated(self, skills_tree):
        assert len(list(iter_skill_files(skills_tree, skills_tree))) == 4


class TestDeriveSizeBudget:
    def test_budget_comes_from_the_corpus(self):
        stats = CorpusStats(count=201, smallest=400, median=5_200, p75=9_800, p90=17_500, largest=72_170)
        budget, reason = derive_size_budget(5_000, stats, percentile=90)
        assert budget == 17_500
        assert "p90" in reason

    def test_an_oversized_skill_is_never_disqualified_by_its_own_size(self):
        """research-paper-writing is 72 KB; a 15 KB cap made it unevolvable."""
        stats = CorpusStats(count=201, smallest=400, median=5_200, p75=9_800, p90=17_500, largest=72_170)
        budget, reason = derive_size_budget(72_170, stats)
        assert budget > 72_170
        assert budget == int(72_170 * (1 + BASELINE_HEADROOM))
        assert "baseline" in reason

    def test_no_corpus_falls_back_and_says_so(self):
        budget, reason = derive_size_budget(5_000, None, fallback=15_000)
        assert budget == 15_000
        assert "fallback" in reason

    def test_a_tiny_corpus_cannot_produce_an_unusable_budget(self):
        stats = CorpusStats(count=2, smallest=100, median=150, p75=200, p90=200, largest=200)
        budget, _ = derive_size_budget(120, stats)
        assert budget >= MIN_BUDGET_CHARS

    def test_percentile_selection(self):
        stats = CorpusStats(count=10, smallest=1, median=100, p75=200, p90=300, largest=900)
        assert derive_size_budget(0, stats, percentile=50)[0] == max(100, MIN_BUDGET_CHARS)
        assert derive_size_budget(0, stats, percentile=99)[0] == max(900, MIN_BUDGET_CHARS)


# ── reporting ───────────────────────────────────────────────────────────


def report(baseline_scores, evolved_scores, **kw):
    return ABReport(
        subject="demo",
        baseline=arm_from_scores("baseline", baseline_scores, kw.pop("baseline_size", 1_000)),
        evolved=arm_from_scores("evolved", evolved_scores, kw.pop("evolved_size", 1_000)),
        **kw,
    )


class TestVerdict:
    def test_a_clear_win_ships(self):
        r = report([0.50, 0.51, 0.49], [0.80, 0.81, 0.79])
        assert r.verdict()[0] == "SHIP"

    def test_a_delta_inside_the_noise_band_holds(self):
        """A positive but indistinguishable delta is noise, not an improvement."""
        r = report([0.50, 0.60, 0.40], [0.52, 0.62, 0.42])
        assert r.within_noise
        assert r.verdict()[0] == "HOLD"
        assert "noise band" in r.verdict()[1]

    def test_a_regression_holds(self):
        r = report([0.80, 0.81, 0.79], [0.50, 0.51, 0.49])
        verdict, reason = r.verdict()
        assert verdict == "HOLD"
        assert "worse" in reason

    def test_constraint_failure_overrides_a_win(self):
        r = report(
            [0.1, 0.1, 0.1], [0.9, 0.9, 0.9],
            constraints_passed=False,
            constraint_failures=["size_limit: 20,000/17,500 chars"],
        )
        verdict, reason = r.verdict()
        assert verdict == "HOLD"
        assert "size_limit" in reason

    def test_an_empty_arm_holds(self):
        assert report([], [0.9]).verdict()[0] == "HOLD"


class TestNoiseBand:
    def test_band_comes_from_measured_spread(self):
        r = report([0.5, 0.7], [0.5, 0.7])
        assert r.noise_band > 0

    def test_single_rep_uses_an_assumed_band(self):
        r = report([0.5], [0.6])
        assert r.noise_band > 0
        assert "assumed" in r.verdict()[1]


class TestCaveats:
    def test_single_rep_is_flagged_without_being_asked(self):
        caveats = " ".join(report([0.5], [0.9]).auto_caveats())
        assert "Single repetition" in caveats

    def test_unbalanced_arms_are_flagged(self):
        caveats = " ".join(report([0.5, 0.5, 0.5], [0.9]).auto_caveats())
        assert "unbalanced" in caveats

    def test_a_larger_artifact_is_flagged_on_a_win(self):
        r = report([0.1, 0.1, 0.1], [0.9, 0.9, 0.9], baseline_size=1_000, evolved_size=1_800)
        assert any("larger" in c for c in r.auto_caveats())

    def test_explicit_caveats_are_preserved(self):
        r = report([0.5], [0.6])
        r.caveats.append("Arms used different prompts.")
        assert "Arms used different prompts." in r.auto_caveats()

    def test_errored_runs_are_disclosed(self):
        r = report([0.5, 0.5], [0.9, 0.9])
        r.evolved.errors = 3
        assert any("Errored runs excluded" in c for c in r.auto_caveats())


class TestRendering:
    def test_markdown_carries_the_verdict_and_the_table(self):
        md = report([0.5, 0.5, 0.5], [0.9, 0.9, 0.9]).to_markdown()
        assert "**Verdict:" in md
        assert "| score |" in md
        assert "Noise band" in md

    def test_caveats_section_appears_only_when_there_are_caveats(self):
        clean = report([0.5, 0.5, 0.5], [0.9, 0.9, 0.9]).to_markdown()
        assert "Caveats recorded" not in clean

        single_rep = report([0.5], [0.9]).to_markdown()
        assert "Caveats recorded" in single_rep
        assert "Single repetition" in single_rep

    def test_json_round_trips(self):
        payload = json.loads(json.dumps(report([0.5], [0.9]).to_dict()))
        assert payload["verdict"] in {"SHIP", "HOLD"}
        assert "noise_band" in payload

    def test_write_produces_both_files(self, tmp_path):
        md_path, json_path = report([0.5], [0.9]).write(tmp_path / "out")
        assert md_path.is_file() and json_path.is_file()
        assert "Verdict" in md_path.read_text()


class TestArmBuilders:
    def test_from_eval_run_excludes_errors_from_the_means(self):
        from evolution.core.agent_runner import EvalRun, TaskResult

        run = EvalRun(
            label="x",
            results=[
                TaskResult("a", 1.0, 2, 3, 100, 1.0),
                TaskResult("b", 0.0, 0, 0, 0, 0.0, error="boom"),
            ],
        )
        arm = arm_from_eval_run("evolved", run, size_chars=10)
        assert arm.scores == [1.0]
        assert arm.errors == 1

    def test_arm_stats_are_json_safe(self):
        payload = ArmStats(label="x", scores=[0.5, 0.7]).as_dict()
        assert payload["n"] == 2
        assert payload["mean_score"] == pytest.approx(0.6)
