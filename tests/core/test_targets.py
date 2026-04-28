"""Tests for repository target discovery."""

from evolution.repos.targets import scan_skill_targets


def test_scan_skill_targets_reads_frontmatter_name_and_description(tmp_path):
    repo_path = tmp_path / "repo"
    skill_dir = repo_path / "skills" / "github" / "github-code-review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: github-code-review\ndescription: Review PRs\n---\n\n# Review\n"
    )

    targets = scan_skill_targets(repo_path)

    assert len(targets) == 1
    assert targets[0].target_type == "skill"
    assert targets[0].name == "github-code-review"
    assert targets[0].file_path == "skills/github/github-code-review/SKILL.md"
    assert targets[0].metadata["description"] == "Review PRs"
