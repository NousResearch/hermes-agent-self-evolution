"""Tests for skills-as-bundles.

96 of the 201 installed skills ship supporting files under progressive
disclosure. An optimizer that sees only SKILL.md can drop a link to one of
them — which reads as a size *win* to every other check — or invent a path
that was never written.
"""

from __future__ import annotations

import pytest

from evolution.core.skill_bundle import (
    broken_references,
    extract_references,
    invented_references,
    load_bundle,
)


ENTRY = """---
name: demo
description: demo skill
---

# Demo

Read [the API guide](references/api.md) first.
Then run scripts/setup.sh.
See also https://example.com/not-a-file.md and #anchor-only.
"""


@pytest.fixture
def bundle_dir(tmp_path):
    root = tmp_path / "demo"
    (root / "references").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)
    (root / "SKILL.md").write_text(ENTRY)
    (root / "references" / "api.md").write_text("API details.")
    (root / "scripts" / "setup.sh").write_text("#!/bin/sh\necho hi\n")
    (root / "references" / "unused.md").write_text("Nobody links here.")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    (root / ".DS_Store").write_text("")
    return root


class TestExtractReferences:
    def test_markdown_links(self):
        assert "references/api.md" in extract_references("[x](references/api.md)")

    def test_bare_relative_paths(self):
        assert "scripts/setup.sh" in extract_references("Then run scripts/setup.sh.")

    def test_external_urls_are_not_references(self):
        assert extract_references("[x](https://example.com/a.md)") == set()

    def test_anchors_are_stripped(self):
        assert "guide.md" in extract_references("[x](guide.md#section)")

    def test_leading_dot_slash_is_normalized(self):
        assert "guide.md" in extract_references("[x](./guide.md)")


class TestLoadBundle:
    def test_supporting_files_are_found(self, bundle_dir):
        bundle = load_bundle(bundle_dir / "SKILL.md")
        names = {f.relpath for f in bundle.supporting}
        assert "references/api.md" in names
        assert "scripts/setup.sh" in names

    def test_noise_files_are_ignored(self, bundle_dir):
        bundle = load_bundle(bundle_dir / "SKILL.md")
        names = {f.relpath for f in bundle.supporting}
        assert not any("__pycache__" in n for n in names)
        assert ".DS_Store" not in names

    def test_referenced_files_are_flagged(self, bundle_dir):
        bundle = load_bundle(bundle_dir / "SKILL.md")
        referenced = {f.relpath for f in bundle.referenced_files}
        assert referenced == {"references/api.md", "scripts/setup.sh"}

    def test_unreferenced_files_are_not_flagged(self, bundle_dir):
        bundle = load_bundle(bundle_dir / "SKILL.md")
        unused = next(f for f in bundle.supporting if f.relpath == "references/unused.md")
        assert unused.referenced is False

    def test_single_file_skill_is_not_a_bundle(self, tmp_path):
        root = tmp_path / "solo"
        root.mkdir()
        (root / "SKILL.md").write_text(ENTRY)
        bundle = load_bundle(root / "SKILL.md")
        assert bundle.is_bundle is False
        assert bundle.describe() == "single file"

    def test_total_size_counts_the_whole_bundle(self, bundle_dir):
        bundle = load_bundle(bundle_dir / "SKILL.md")
        assert bundle.total_size > len(bundle.entry_text)


class TestOptimizerContext:
    def test_only_referenced_files_are_shown(self, bundle_dir):
        context = load_bundle(bundle_dir / "SKILL.md").context_for_optimizer()
        assert "references/api.md" in context
        assert "references/unused.md" not in context

    def test_context_tells_the_optimizer_not_to_inline(self, bundle_dir):
        context = load_bundle(bundle_dir / "SKILL.md").context_for_optimizer()
        assert "do not" in context.lower() and "inline" in context.lower()

    def test_no_referenced_files_means_no_context(self, tmp_path):
        root = tmp_path / "x"
        root.mkdir()
        (root / "SKILL.md").write_text("# No links here\n")
        (root / "extra.md").write_text("orphan")
        assert load_bundle(root / "SKILL.md").context_for_optimizer() == ""


class TestReferenceIntegrity:
    def test_dropping_a_link_is_detected(self, bundle_dir):
        bundle = load_bundle(bundle_dir / "SKILL.md")
        stripped = ENTRY.replace("[the API guide](references/api.md)", "the API guide")
        assert broken_references(bundle, stripped) == ["references/api.md"]

    def test_keeping_every_link_is_clean(self, bundle_dir):
        bundle = load_bundle(bundle_dir / "SKILL.md")
        assert broken_references(bundle, ENTRY) == []

    def test_a_link_by_basename_still_counts(self, bundle_dir):
        bundle = load_bundle(bundle_dir / "SKILL.md")
        rewritten = ENTRY.replace("references/api.md", "api.md")
        assert broken_references(bundle, rewritten) == []

    def test_invented_paths_are_detected(self, bundle_dir):
        bundle = load_bundle(bundle_dir / "SKILL.md")
        embellished = ENTRY + "\nAlso read [the deep dive](references/deep.md).\n"
        assert invented_references(bundle, embellished) == ["references/deep.md"]

    def test_preexisting_broken_links_are_not_blamed_on_this_run(self, tmp_path):
        root = tmp_path / "y"
        (root / "references").mkdir(parents=True)
        (root / "SKILL.md").write_text("# Y\nSee [gone](references/gone.md).\n")
        (root / "references" / "here.md").write_text("x")
        bundle = load_bundle(root / "SKILL.md")

        # The baseline already pointed at a missing file; an unchanged rewrite
        # must not be reported as having invented it.
        assert invented_references(bundle, bundle.entry_text) == []
