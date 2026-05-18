"""Tests for signature_freeze module."""

from textwrap import dedent

from evolution.code.signature_freeze import (
    compare_sources,
    count_try_except,
    extract_public_signatures,
    extract_registry_calls,
)

SAMPLE_TOOL = dedent("""\
    from tools.registry import register

    def read_file(path: str, encoding: str = "utf-8") -> str:
        try:
            with open(path, encoding=encoding) as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def _helper():
        pass

    register("read_file", read_file)
""")


class TestExtractPublicSignatures:
    def test_extracts_public_only(self):
        sigs = extract_public_signatures(SAMPLE_TOOL)
        assert "read_file" in sigs
        assert "_helper" not in sigs

    def test_args_match(self):
        sigs = extract_public_signatures(SAMPLE_TOOL)
        assert "path:" in sigs["read_file"]
        assert "encoding:" in sigs["read_file"]


class TestExtractRegistryCalls:
    def test_finds_register_call(self):
        calls = extract_registry_calls(SAMPLE_TOOL)
        assert "register(read_file)" in calls


class TestCountTryExcept:
    def test_counts_correctly(self):
        assert count_try_except(SAMPLE_TOOL) == 1


class TestCompareSources:
    def test_identical_sources_pass(self):
        report = compare_sources(SAMPLE_TOOL, SAMPLE_TOOL)
        assert report.passed

    def test_signature_change_flagged(self):
        baseline = "def read_file(path): pass\n"
        candidate = "def read_file(path, mode='r'): pass\n"
        report = compare_sources(baseline, candidate)
        assert not report.passed
        assert report.signature_violations

    def test_removed_register_flagged(self):
        mutated = SAMPLE_TOOL.replace('register("read_file", read_file)', "")
        report = compare_sources(SAMPLE_TOOL, mutated)
        assert not report.passed
        assert any("removed" in v for v in report.registry_violations)

    def test_error_handling_decrease_flagged(self):
        mutated = SAMPLE_TOOL.replace("try:", "if True:").replace("except FileNotFoundError:", "if False:")
        report = compare_sources(SAMPLE_TOOL, mutated)
        assert report.error_handling_decreased
        assert not report.passed

    def test_syntax_error_in_candidate(self):
        report = compare_sources(SAMPLE_TOOL, "this is not python")
        assert not report.passed
        assert report.signature_violations
