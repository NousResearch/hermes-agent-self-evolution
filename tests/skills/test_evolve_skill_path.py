"""--skill-path lets evolve_skill target a skill anywhere on disk."""

from click.testing import CliRunner

from evolution.skills.evolve_skill import main

SKILL = "---\nname: outside-skill\ndescription: Lives outside the hermes repo\n---\n\n# Body\nDo things.\n"


def _run(tmp_path, *args):
    return CliRunner().invoke(main, [*args, "--dry-run", "--hermes-repo", str(tmp_path)])


def test_skill_path_to_directory(tmp_path):
    d = tmp_path / "ext" / "outside-skill"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(SKILL)
    r = _run(tmp_path, "--skill-path", str(d))
    assert r.exit_code == 0, r.output
    assert "Name: outside-skill" in r.output


def test_skill_path_to_file(tmp_path):
    f = tmp_path / "SKILL.md"
    f.write_text(SKILL)
    r = _run(tmp_path, "--skill-path", str(f))
    assert r.exit_code == 0, r.output


def test_missing_skill_path_fails(tmp_path):
    r = _run(tmp_path, "--skill-path", str(tmp_path / "nope"))
    assert r.exit_code == 1
    assert "not found" in r.output


def test_requires_skill_or_skill_path(tmp_path):
    r = _run(tmp_path)
    assert r.exit_code != 0
    assert "--skill" in r.output
