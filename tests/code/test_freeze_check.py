"""Tests for the zero-LM freeze + diff-shape guards."""

from evolution.code.freeze_check import (
    check_diff_shape,
    extract_public_surface,
    freeze_violations,
    surface_drift,
)

_BASE = '''\
"""A tool."""
import re

TOOL_SCHEMA = {"name": "do_thing"}


def do_thing(content: str, count: int = 1) -> str:
    """Public entry point."""
    return _helper(content) * count


def _helper(s: str) -> str:
    return s.strip()


class Result:
    pass
'''


class TestExtractPublicSurface:
    def test_collects_public_functions_classes_schemas(self):
        s = extract_public_surface(_BASE)
        assert "do_thing" in s.functions
        assert "Result" in s.classes
        assert "TOOL_SCHEMA" in s.schemas

    def test_excludes_private_helpers(self):
        s = extract_public_surface(_BASE)
        assert "_helper" not in s.functions


class TestSurfaceDrift:
    def test_identical_surface_no_drift(self):
        a = extract_public_surface(_BASE)
        assert surface_drift(a, a) == []

    def test_renamed_function_is_drift(self):
        renamed = _BASE.replace("def do_thing(", "def do_thing_renamed(")
        reasons = surface_drift(
            extract_public_surface(_BASE), extract_public_surface(renamed)
        )
        assert any("do_thing" in r and "removed or renamed" in r for r in reasons)

    def test_added_parameter_is_signature_drift(self):
        added = _BASE.replace(
            "def do_thing(content: str, count: int = 1)",
            "def do_thing(content: str, count: int = 1, extra=None)",
        )
        reasons = surface_drift(
            extract_public_surface(_BASE), extract_public_surface(added)
        )
        assert any("signature of 'do_thing' changed" in r for r in reasons)

    def test_removed_schema_is_drift(self):
        no_schema = _BASE.replace('TOOL_SCHEMA = {"name": "do_thing"}\n', "")
        reasons = surface_drift(
            extract_public_surface(_BASE), extract_public_surface(no_schema)
        )
        assert any("TOOL_SCHEMA" in r for r in reasons)

    def test_adding_new_public_function_is_allowed(self):
        added = _BASE + "\n\ndef brand_new(x):\n    return x\n"
        reasons = surface_drift(
            extract_public_surface(_BASE), extract_public_surface(added)
        )
        assert reasons == []

    def test_annotation_only_change_is_not_drift(self):
        # Refining a type hint is allowed; only param names/kinds/count matter.
        retyped = _BASE.replace("count: int = 1", "count: float = 1")
        reasons = surface_drift(
            extract_public_surface(_BASE), extract_public_surface(retyped)
        )
        assert reasons == []


class TestDiffShape:
    def test_within_bounds_ok(self):
        slightly_edited = _BASE.replace("s.strip()", "s.rstrip()")
        assert check_diff_shape(_BASE, slightly_edited) == []

    def test_shrink_below_floor_flagged(self):
        gutted = '"""A tool."""\nTOOL_SCHEMA = {}\n'
        reasons = check_diff_shape(_BASE, gutted)
        assert any("shrank" in r for r in reasons)

    def test_unparseable_flagged(self):
        reasons = check_diff_shape(_BASE, "def broken(:\n")
        assert any("does not parse" in r for r in reasons)


class TestFreezeViolations:
    def test_clean_fix_no_violations(self):
        fixed = _BASE.replace("s.strip()", "s.strip() or ''")
        assert freeze_violations(_BASE, fixed) == []

    def test_unparseable_short_circuits_without_surface_error(self):
        # Must not raise from surface extraction on unparseable output.
        reasons = freeze_violations(_BASE, "def x(:")
        assert reasons and all("does not parse" in r for r in reasons)

    def test_combines_shape_and_surface(self):
        renamed_and_shrunk = 'def other():\n    return 1\n'
        reasons = freeze_violations(_BASE, renamed_and_shrunk)
        assert any("shrank" in r for r in reasons)
        assert any("removed or renamed" in r for r in reasons)
