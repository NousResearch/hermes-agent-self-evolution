"""Tests for relevance scoring after the rubric tightening.

The original defect: the relevance judge matched on shared vocabulary, so an
example about reverse-engineering an app (with "gates" and "evidence" in it)
was classified relevant to the gate-evidence-ledger skill. The whole
optimization then trained a gate-evidence skill against app-reverse-engineering
examples and produced mutations about "god mode prompts for iOS clones".

These tests pin the two parts of the fix:
  1. _skill_summary feeds the judge a purpose statement, not raw frontmatter.
  2. The ScoreRelevance rubric forbids vocabulary matching.
"""

import inspect

import pytest

from evolution.core import external_importers as ei


SKILL = """---
name: gate-evidence-ledger
description: Use for gated, phased, or spec-driven work needing evidence.
version: 1.0.0
metadata:
  hermes:
    tags: [gates, evidence, audit]
    related_skills: [spec-conformance-audit]
---

# Gate Evidence Ledger

Track requirement -> evidence -> verdict across numbered gates.
"""


class TestSkillSummary:
    def test_prefers_frontmatter_description(self):
        summary = ei._skill_summary("gate-evidence-ledger", SKILL)
        assert "Use for gated, phased, or spec-driven work needing evidence" in summary
        assert summary.startswith("gate-evidence-ledger:")

    def test_does_not_include_yaml_tags(self):
        """The summary must not be frontmatter cruft like `tags: [gates]`."""
        summary = ei._skill_summary("gate-evidence-ledger", SKILL)
        assert "metadata:" not in summary
        assert "related_skills:" not in summary
        assert "version:" not in summary

    def test_falls_back_to_body_when_no_description(self):
        body_only = "---\nname: foo\n---\n\n# Foo\n\nDoes the thing."
        summary = ei._skill_summary("foo", body_only)
        assert "foo" in summary

    def test_bounded_length(self):
        long_skill = "---\nname: x\ndescription: " + ("word " * 500) + "\n---\nbody"
        summary = ei._skill_summary("x", long_skill)
        assert len(summary) < 400


class TestRubric:
    def test_rubric_forbids_vocabulary_matching(self):
        """The rubric must tell the judge that shared words are not enough."""
        rubric = " ".join(inspect.getsource(ei.RelevanceFilter.ScoreRelevance).split())
        assert "NOT a vocabulary match" in rubric
        assert "would following this skill's procedure" in rubric

    def test_rubric_requires_central_goal(self):
        rubric = " ".join(inspect.getsource(ei.RelevanceFilter.ScoreRelevance).split())
        assert "central goal" in rubric

    def test_relevant_means_procedure_exercise(self):
        """The output contract must define relevance as procedure-exercise."""
        rubric = " ".join(inspect.getsource(ei.RelevanceFilter.ScoreRelevance).split())
        assert "this skill's procedure exists to do" in rubric
