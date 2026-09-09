"""The persisted artifact name must never outrank the verdict.

Regression cover for a live failure: a run whose evolved skill was
byte-identical to its baseline still wrote `evolved_skill.md`, and the cron
wrapper reading that directory reported "variant is deployable" and printed a
`cp` command over the live skill.
"""

from evolution.skills.evolve_skill import (
    DEPLOYABLE_ARTIFACT,
    HOLD_ARTIFACT,
    artifact_name,
    is_deployable,
)


class TestIsDeployable:
    def test_ship_with_no_failures_is_deployable(self):
        assert is_deployable("SHIP", [])

    def test_ship_is_vetoed_by_a_constraint_failure(self):
        # A measured win does not override a constraint breach.
        assert not is_deployable("SHIP", ["size_limit: 16740/15000 chars"])

    def test_hold_is_never_deployable(self):
        assert not is_deployable("HOLD", [])

    def test_none_failures_is_treated_as_clean(self):
        assert is_deployable("SHIP", None)

    def test_unknown_verdict_is_not_deployable(self):
        # Fail closed on anything the report grows later.
        assert not is_deployable("DRY_RUN", [])
        assert not is_deployable("", [])


class TestArtifactName:
    def test_deployable_gets_the_deploy_name(self):
        assert artifact_name(True) == DEPLOYABLE_ARTIFACT

    def test_non_deployable_never_gets_the_deploy_name(self):
        assert artifact_name(False) == HOLD_ARTIFACT
        assert artifact_name(False) != DEPLOYABLE_ARTIFACT

    def test_the_two_names_are_distinct(self):
        assert DEPLOYABLE_ARTIFACT != HOLD_ARTIFACT

    def test_a_held_no_op_does_not_produce_a_deployable_filename(self):
        """The exact live case: constraints pass, delta inside the noise band."""
        deployable = is_deployable("HOLD", [])
        assert artifact_name(deployable) == HOLD_ARTIFACT
