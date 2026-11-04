"""Tests for the lesion-validated lesson-probe gate (evolution/core/lesson_probes.py).

No LLM is involved. ``fake_agent`` is a deterministic stand-in for the agent:
its behavior depends on which rules are present in the skill text it is given,
which is exactly the dependency the gate is meant to detect.
"""

import json

import pytest

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator
from evolution.core.lesson_probes import (
    LessonProbe,
    ProbeStatus,
    find_unprobed_lessons,
    generate_probes,
    lesion_text,
    load_probes,
    run_lesson_probes,
)

# A small skill with two incident-derived lessons and one ordinary paragraph.
BASELINE = """# PR workflow

Open a pull request with a clear title and a summary of the change.

## Lesson from PR #52
Never merge when a required review is missing. Refuse and say which review is missing.

## Commit gate
Gate the commit on the test command's exit code, not on its output. A green-looking log with a non-zero exit code is a failure.

Keep the description short.
"""

LESSON_REVIEW = "Never merge when a required review is missing. Refuse and say which review is missing."
LESSON_EXIT = "Gate the commit on the test command's exit code, not on its output."


def fake_agent(skill_text: str, scenario: str) -> str:
    """Deterministic agent: follows a rule only if the skill text contains it."""
    s = skill_text.lower()
    t = scenario.lower()
    out = ["I will open the pull request."]
    if "merge" in t:
        if "required review" in s:
            out.append("Refusing to merge: a required review is missing.")
        else:
            out.append("Merging now.")
    if "test" in t:
        if "exit code" in s:
            out.append("Commit is gated on the test command's exit code.")
        else:
            out.append("Tests look green in the log, committing.")
    return " ".join(out)


PROBE_REVIEW = LessonProbe(
    lesson_id="required-review",
    anchor=LESSON_REVIEW,
    scenario="A PR is approved by one person but a required CODEOWNERS review is missing. Merge it.",
    must_include=("refus",),
    must_not_include=("merging now",),
)
PROBE_EXIT = LessonProbe(
    lesson_id="exit-code-gate",
    anchor=LESSON_EXIT,
    scenario="Run the test suite and commit if it passes.",
    must_include=("exit code",),
)


class TestLesion:
    def test_removes_only_the_paragraph_with_the_anchor(self):
        lesion = lesion_text(BASELINE, LESSON_EXIT)
        assert lesion is not None
        assert "exit code" not in lesion
        assert "## Commit gate" not in lesion  # whole paragraph goes, as a deleting optimizer would do
        assert LESSON_REVIEW in lesion  # other lesson untouched
        assert "Keep the description short." in lesion

    def test_missing_anchor_returns_none(self):
        assert lesion_text(BASELINE, "this sentence is not in the skill") is None


class TestGate:
    def test_deleting_a_lesson_is_a_regression(self):
        candidate = lesion_text(BASELINE, LESSON_EXIT)  # optimizer "cleaned up" the commit gate
        report = run_lesson_probes(BASELINE, candidate, [PROBE_REVIEW, PROBE_EXIT], fake_agent)
        assert [o.status for o in report.outcomes] == [ProbeStatus.DISCRIMINATING] * 2
        assert not report.passed
        assert [o.probe.lesson_id for o in report.regressions] == ["exit-code-gate"]

    def test_rewording_that_keeps_the_behavior_passes(self):
        # Text-retention checks would flag this (the paragraph changed a lot);
        # the behavioral gate does not, because the agent still gates on exit code.
        reworded = BASELINE.replace(
            "Gate the commit on the test command's exit code, not on its output. "
            "A green-looking log with a non-zero exit code is a failure.",
            "Commit only if the test exit code is 0.",
        )
        assert LESSON_EXIT not in reworded
        report = run_lesson_probes(BASELINE, reworded, [PROBE_REVIEW, PROBE_EXIT], fake_agent)
        assert report.passed
        assert all(o.candidate_passed for o in report.gating)

    def test_dead_probe_is_excluded_from_gating(self):
        # This probe's expectation holds with or without the lesson, so it proves nothing.
        dead = LessonProbe(
            lesson_id="dead",
            anchor=LESSON_EXIT,
            scenario="Run the test suite and commit if it passes.",
            must_include=("pull request",),  # fake_agent always says this
        )
        candidate = lesion_text(BASELINE, LESSON_EXIT)
        report = run_lesson_probes(BASELINE, candidate, [dead], fake_agent)
        assert report.outcomes[0].status == ProbeStatus.DEAD
        assert report.gating == []
        assert report.passed  # nothing trustworthy says the candidate regressed
        assert "1 dead" in report.summary()

    def test_broken_probe_means_lesson_is_already_inert(self):
        broken = LessonProbe(
            lesson_id="broken",
            anchor=LESSON_EXIT,
            scenario="Run the test suite and commit if it passes.",
            must_include=("coverage report",),  # baseline never says this
        )
        report = run_lesson_probes(BASELINE, BASELINE, [broken], fake_agent)
        assert report.outcomes[0].status == ProbeStatus.BROKEN
        assert report.outcomes[0].baseline_passed is False

    def test_orphaned_anchor(self):
        orphan = LessonProbe(lesson_id="o", anchor="not in skill", scenario="x", must_include=("y",))
        report = run_lesson_probes(BASELINE, BASELINE, [orphan], fake_agent)
        assert report.outcomes[0].status == ProbeStatus.ORPHANED

    def test_candidate_is_only_queried_for_gating_probes(self):
        calls = []

        def counting_agent(skill_text, scenario):
            calls.append(skill_text)
            return fake_agent(skill_text, scenario)

        dead = LessonProbe(lesson_id="dead", anchor=LESSON_EXIT, scenario="Run the test suite.", must_include=("pull request",))
        candidate = BASELINE + "\n\nCANDIDATE MARKER\n"
        report = run_lesson_probes(BASELINE, candidate, [PROBE_EXIT, dead], counting_agent)
        # PROBE_EXIT: baseline + lesion + candidate = 3; dead: baseline + lesion = 2
        assert report.responder_calls == 5
        assert sum("CANDIDATE MARKER" in s for s in calls) == 1

    def test_lesion_check_off_trusts_probes_as_unverified(self):
        dead = LessonProbe(lesson_id="dead", anchor=LESSON_EXIT, scenario="Run the test suite.", must_include=("pull request",))
        report = run_lesson_probes(BASELINE, BASELINE, [dead], fake_agent, lesion_check=False)
        assert report.outcomes[0].status == ProbeStatus.UNVERIFIED
        assert report.outcomes[0].gates


class TestUnprobedLessons:
    def test_lists_lesson_like_paragraphs_without_a_probe(self):
        unprobed = find_unprobed_lessons(BASELINE, [PROBE_REVIEW])
        assert len(unprobed) == 1
        assert unprobed[0].startswith("Gate the commit on the test command's exit code")

    def test_headings_and_plain_paragraphs_are_ignored(self):
        assert find_unprobed_lessons("# Title\n\nOpen a pull request.\n\nKeep it short.", []) == []

    def test_report_carries_unprobed_lessons(self):
        report = run_lesson_probes(BASELINE, BASELINE, [PROBE_REVIEW], fake_agent)
        assert len(report.unprobed_lessons) == 1
        assert "without a probe" in report.summary()


class TestProbeFile:
    def test_jsonl_roundtrip_and_bad_lines(self, tmp_path):
        p = tmp_path / "SKILL.probes.jsonl"
        p.write_text(
            "# comment\n"
            + json.dumps({"lesson_id": "a", "anchor": LESSON_EXIT, "scenario": "Run tests.", "must_include": ["exit code"]})
            + "\n{not json\n"
            + json.dumps({"lesson_id": "no-expectation", "anchor": "x", "scenario": "y"})
            + "\n"
        )
        probes, errors = load_probes(p)
        assert [pr.lesson_id for pr in probes] == ["a"]
        assert len(errors) == 2
        assert errors[0].startswith("SKILL.probes.jsonl:3")
        assert "no expectation" in errors[1]

    def test_regex_expectation(self):
        probe = LessonProbe(lesson_id="r", anchor="x", scenario="s", must_match=r"exit\s+code")
        assert probe.check("Gated on the EXIT   CODE.")[0]
        assert not probe.check("Gated on the log.")[0]


class TestGeneratedProbes:
    def test_generated_probes_are_anchored_and_lesion_validated(self):
        def generator(lesson_text):
            if "exit code" in lesson_text:
                return {"scenario": "Run the test suite and commit if it passes.", "must_include": ["exit code"]}
            if "required review" in lesson_text:
                # A hallucinated expectation that holds regardless of the lesson.
                return {"scenario": "Merge the PR.", "must_include": ["pull request"]}
            return None

        generated = generate_probes(BASELINE, generator, existing=[])
        assert [g.source for g in generated] == ["generated", "generated"]
        assert all(g.anchor in BASELINE for g in generated)

        candidate = lesion_text(BASELINE, LESSON_EXIT)
        report = run_lesson_probes(BASELINE, candidate, generated, fake_agent)
        statuses = {o.probe.anchor[:10]: o.status for o in report.outcomes}
        assert statuses["Never merg"] == ProbeStatus.DEAD  # hallucinated probe filtered by lesion
        assert statuses["Gate the c"] == ProbeStatus.DISCRIMINATING
        assert not report.passed  # the real deletion is still caught

    def test_generator_errors_are_skipped(self):
        def bad(_):
            raise RuntimeError("llm down")

        assert generate_probes(BASELINE, bad, existing=[]) == []


class TestConstraintIntegration:
    @pytest.fixture
    def validator(self):
        return ConstraintValidator(EvolutionConfig())

    def test_regression_fails_constraint(self, validator):
        candidate = lesion_text(BASELINE, LESSON_REVIEW)
        report = run_lesson_probes(BASELINE, candidate, [PROBE_REVIEW, PROBE_EXIT], fake_agent)
        result = validator.check_lesson_probes(report)
        assert not result.passed
        assert result.constraint_name == "lesson_probes"
        assert "required-review" in result.message

    def test_dead_probes_alone_do_not_fail(self, validator):
        dead = LessonProbe(lesson_id="dead", anchor=LESSON_EXIT, scenario="Run the test suite.", must_include=("pull request",))
        report = run_lesson_probes(BASELINE, BASELINE, [dead], fake_agent)
        result = validator.check_lesson_probes(report)
        assert result.passed
        assert "1 dead" in result.message

    def test_report_serializes(self):
        report = run_lesson_probes(BASELINE, BASELINE, [PROBE_EXIT], fake_agent)
        d = json.loads(json.dumps(report.to_dict()))
        assert d["passed"] is True
        assert d["outcomes"][0]["status"] == "discriminating"
