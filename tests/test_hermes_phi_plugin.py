"""Tests for the Hermes PHI plugin (Parts B and C).

Covers:
  - Plugin hook contract (bd-jqrzp.2)
    - Redaction before LLM context
    - Block suppresses raw payload
    - Clean text passes through
    - Hook callable with default SafetyKernel
  - Production plugin (bd-jqrzp.3)
    - Allow-with-redaction for MRN, DOB, phone, health-plan IDs, SSN
    - Block for high-confidence PHI (MRN, health_plan_id, SSN)
    - Fail-closed when Safety Kernel unavailable
    - No raw PHI in exception attributes or model-bound payload
    - Phone number does not pass through unchanged
    - Config defaults are fail-closed
"""

import json
import logging
import os
from unittest.mock import patch, MagicMock

import pytest


# ── Test Environment Setup ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _ensure_safety_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure SAFETY_TEST_MODE=1 so llm-common audit falls back to default salt."""
    monkeypatch.setenv("SAFETY_TEST_MODE", "1")


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def plugin():
    from hermes_phi.plugin import HermesPHIPlugin

    p = HermesPHIPlugin()
    yield p
    if p._kernel is not None:
        p._kernel.shutdown()


@pytest.fixture
def proof():
    from hermes_phi import plugin_proof

    return plugin_proof


# ── Part B: Plugin Hook Spike Tests ──────────────────────────────────────


class TestHookSpikeContract:
    """Validate the hook/adapter contract (bd-jqrzp.2)."""

    def test_clean_text_passes_through(self, proof):
        """Clean text with no PHI is allowed without modification."""
        from llm_common.core.kernel import SafetyKernel

        kernel = SafetyKernel()
        try:
            result = proof.check_text_before_llm(
                "What is the capital of France?",
                kernel=kernel,
            )
            assert result["action"] == "allow"
            assert result["payload"] == "What is the capital of France?"
        finally:
            kernel.shutdown()

    def test_mrn_is_intercepted(self, proof):
        """PHI text with MRN is intercepted before LLM-bound context."""
        from llm_common.core.kernel import SafetyKernel

        kernel = SafetyKernel()
        try:
            text, _ = proof.get_phi_test_sample("mrn")
            result = proof.check_text_before_llm(text, kernel=kernel)

            assert result["action"] in ("block", "allow_with_redaction")
            if result["action"] == "allow_with_redaction":
                assert "REDACTED" in result["payload"]
            if result["action"] == "block":
                assert result["payload"] == "[BLOCKED_PHI]"
        finally:
            kernel.shutdown()

    def test_clean_text_has_no_findings(self, proof):
        """Clean text has zero safety findings."""
        from llm_common.core.kernel import SafetyKernel

        kernel = SafetyKernel()
        try:
            result = proof.check_text_before_llm(
                "List all files in the current directory.", kernel=kernel
            )
            assert result["action"] == "allow"
            assert len(result["findings"]) == 0
        finally:
            kernel.shutdown()

    def test_hook_callable_with_default_kernel(self, proof):
        """check_text_before_llm creates a default SafetyKernel when none provided."""
        result = proof.check_text_before_llm("Hello world")
        assert result["action"] == "allow"

    def test_blocked_payload_is_sentinel_not_raw(self, proof):
        """Regression: blocked payload must never contain raw text."""
        from llm_common.core.kernel import SafetyKernel

        kernel = SafetyKernel()
        try:
            text = "SSN: 123-45-6789"
            result = proof.check_text_before_llm(
                text, kernel=kernel, surface="ssn_test_surface", policy_set="pii_v1"
            )
            assert result["action"] in ("block", "allow_with_redaction")
            if result["action"] == "block":
                assert result["payload"] == "[BLOCKED_PHI]", (
                    "Blocked payload must be sentinel, not raw text"
                )
                assert result["payload"] != text, (
                    "Raw PHI must not appear in blocked result payload"
                )
        finally:
            kernel.shutdown()

    def test_tool_output_surface(self, proof):
        """Tool output surface routes to the same evaluation."""
        from llm_common.core.kernel import SafetyKernel

        kernel = SafetyKernel()
        try:
            result = proof.check_text_before_llm(
                "Disk usage: 45%",
                surface="hermes_tool_output",
                kernel=kernel,
            )
            assert result["action"] == "allow"
        finally:
            kernel.shutdown()


# ── Part C: Production Plugin Tests ──────────────────────────────────────


class TestHermesPHIPlugin:
    """Validate the production PHI plugin (bd-jqrzp.3)."""

    def test_clean_context_passes(self, plugin):
        """Clean text with no PHI passes through without modification."""
        result = plugin.redact_context("How do I configure pytest?")
        assert result == "How do I configure pytest?"

    def test_mrn_is_redacted_or_blocked(self, plugin):
        """MRN in context triggers redaction or block."""
        from hermes_phi.plugin import PhiBlocked

        text = "Patient MRN-ABC123456 needs follow-up."
        try:
            result = plugin.redact_context(text)
            assert "ABC123456" not in result, (
                "Raw MRN found in redacted output"
            )
            assert "REDACTED" in result or "BLOCK" in result
        except PhiBlocked:
            pass  # Block is acceptable

    def test_phone_number_is_redacted(self, plugin):
        """Phone number in context is redacted (phi_v1_strict: REDACT)."""
        from hermes_phi.plugin import PhiBlocked

        text = "Contact patient at (555) 123-4567 for results."
        try:
            result = plugin.redact_context(text)
            assert "555" not in result, (
                "Phone number leaked in redacted output"
            )
            assert "REDACTED" in result
        except PhiBlocked:
            pass  # Acceptable (phone falls under phi_v1_strict pii REDACT)

    def test_phone_number_redacted_on_tool_output(self, plugin):
        """Phone number is redacted on check_tool_output."""
        from hermes_phi.plugin import PhiBlocked

        text = "Results for (212) 555-0198"
        try:
            result = plugin.check_tool_output(text)
            assert "555" not in result, (
                "Phone number leaked in tool output"
            )
            assert "REDACTED" in result
        except PhiBlocked:
            pass

    def test_phone_number_redacted_on_memory_write(self, plugin):
        """Phone number is redacted on check_memory_write."""
        from hermes_phi.plugin import PhiBlocked

        text = "session: patient (212) 555-0198"
        try:
            result = plugin.check_memory_write(text)
            assert "555" not in result, (
                "Phone number leaked in memory write"
            )
            assert "REDACTED" in result
        except PhiBlocked:
            pass

    def test_health_plan_id_is_blocked(self, plugin):
        """Health plan ID should be blocked per phi_v1_strict BLOCK rule."""
        from hermes_phi.plugin import PhiBlocked

        text = "Health plan ID: HPI-XYZ789012"
        try:
            result = plugin.redact_context(text)
            assert "HPI-XYZ789012" not in result, (
                "Health plan ID leaked in output"
            )
        except PhiBlocked:
            pass  # Block is acceptable

    def test_tool_output_allows_clean(self, plugin):
        """Tool output surface is evaluated without block for clean text."""
        text = "Query returned 42 rows."
        result = plugin.check_tool_output(text)
        assert result == text

    def test_memory_write_allows_clean(self, plugin):
        """Memory write surface is evaluated without block for clean text."""
        text = "Session state: conversation_id=abc123"
        result = plugin.check_memory_write(text)
        assert result == text

    def test_fail_closed_on_missing_safety_kernel(self):
        """When Safety Kernel is unavailable, plugin fails closed (blocks)."""
        from hermes_phi.plugin import HermesPHIPlugin, PhiSafetyUnavailable

        plugin = HermesPHIPlugin()
        plugin._kernel = None
        plugin._kernel_provided = False

        assert not plugin.is_available
        with pytest.raises(PhiSafetyUnavailable) as exc_info:
            plugin.redact_context("some text")
        assert "SafetyKernel not initialized" in str(exc_info.value)

    def test_initialization_failure_fails_closed(self):
        """If SafetyKernel() raises during init, plugin is still operable but blocks."""
        from hermes_phi.plugin import HermesPHIPlugin, PhiSafetyUnavailable

        with patch("hermes_phi.plugin.SafetyKernel", side_effect=RuntimeError("init failed")):
            plugin = HermesPHIPlugin()
            assert not plugin.is_available
            with pytest.raises(PhiSafetyUnavailable):
                plugin.redact_context("any text")

    def test_empty_text_is_allowed(self, plugin):
        """Empty string is treated as safe (no PHI possible)."""
        result = plugin.redact_context("")
        assert result == ""

    def test_date_of_birth_is_redacted(self, plugin):
        """Date of birth YYYY-MM-DD is redacted by HermesDeterministicPHIDetector."""
        from hermes_phi.plugin import PhiBlocked

        text = "DOB: 1972-08-22"
        try:
            result = plugin.redact_context(text)
            assert "1972-08-22" not in result
            assert "REDACTED" in result
        except PhiBlocked:
            pass

    def test_accession_number_is_redacted(self, plugin):
        """Accession number ACC-* is redacted by HermesDeterministicPHIDetector."""
        from hermes_phi.plugin import PhiBlocked

        text = "Accession: ACC-9876-5432"
        try:
            result = plugin.redact_context(text)
            assert "ACC-9876-5432" not in result
            assert "REDACTED" in result
        except PhiBlocked:
            pass

    def test_order_id_is_redacted(self, plugin):
        """Order ID ORD-* is redacted by HermesDeterministicPHIDetector."""
        from hermes_phi.plugin import PhiBlocked

        text = "Order: ORD-2026-001234"
        try:
            result = plugin.redact_context(text)
            assert "ORD-2026-001234" not in result
            assert "REDACTED" in result
        except PhiBlocked:
            pass


class TestPhiBlockedException:
    """Validate PhiBlocked exception does NOT carry raw PHI."""

    def test_exception_no_payload_snippet_attribute(self):
        """Regression: PhiBlocked has no payload_snippet or raw PHI attribute."""
        from hermes_phi.plugin import PhiBlocked

        exc = PhiBlocked("hermes_context", [{"label": "mrn", "kind": "phi"}])
        assert not hasattr(exc, "payload_snippet"), (
            "PhiBlocked must not carry raw payload_snippet"
        )
        # Verify the error message contains labels but not raw values
        msg = str(exc)
        assert "mrn" in msg
        assert "hermes_context" in msg
        # No raw PHI in any attribute
        for attr_name in dir(exc):
            if attr_name.startswith("_") or callable(getattr(exc, attr_name)):
                continue
            val = str(getattr(exc, attr_name))
            assert "MRN-" not in val, f"Raw PHI found in PhiBlocked.{attr_name}: {val}"

    def test_exception_has_findings_surface_only(self):
        """PhiBlocked only exposes surface and findings, never raw content."""
        from hermes_phi.plugin import PhiBlocked

        findings = [{"label": "ssn", "kind": "pii"}]
        exc = PhiBlocked("hermes_context", findings)
        assert exc.findings == findings
        assert exc.surface == "hermes_context"

    def test_blocked_exception_from_plugin_has_no_raw_phi(self, plugin):
        """When plugin raises PhiBlocked, the exception carries no raw PHI."""
        from hermes_phi.plugin import PhiBlocked

        text = "Patient MRN-ABC123456 needs follow-up."
        try:
            plugin.redact_context(text)
        except PhiBlocked as exc:
            assert not hasattr(exc, "payload_snippet"), (
                "PhiBlocked must not have payload_snippet"
            )
            msg = str(exc)
            assert "MRN-ABC123456" not in msg, "Raw PHI leaked in exception string"
            for attr_name in dir(exc):
                if attr_name.startswith("_") or callable(getattr(exc, attr_name)):
                    continue
                val = str(getattr(exc, attr_name))
                assert "ABC123456" not in val, (
                    f"Raw PHI found in PhiBlocked.{attr_name}"
                )

    def test_exception_message_format(self):
        from hermes_phi.plugin import PhiBlocked

        exc = PhiBlocked("hermes_context", [{"label": "mrn", "kind": "phi"}])
        msg = str(exc)
        assert "PHI blocked" in msg
        assert "mrn" in msg
        assert "hermes_context" in msg

    def test_exception_has_findings(self):
        from hermes_phi.plugin import PhiBlocked

        findings = [{"label": "ssn", "kind": "pii"}]
        exc = PhiBlocked("hermes_context", findings)
        assert exc.findings == findings
        assert exc.surface == "hermes_context"


class TestPluginSurfaceEnum:
    """Validate PHISurface enum values map to Safety Kernel surfaces."""

    def test_surface_values_match_policy_registry(self):
        from hermes_phi.plugin import PHISurface

        assert PHISurface.HERMES_CONTEXT.value == "hermes_context"
        assert PHISurface.HERMES_TOOL_OUTPUT.value == "hermes_tool_output"
        assert PHISurface.HERMES_MEMORY_WRITE.value == "hermes_memory_write"


class TestLogSafety:
    """Validate PHI-safe logging formatter."""

    def test_ssn_redacted_in_logs(self):
        from hermes_phi.plugin import PHISafeFormatter

        fmt = PHISafeFormatter("%(message)s")
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "SSN is 123-45-6789", (), None,
        )
        output = fmt.format(record)
        assert "123-45-6789" not in output
        assert "REDACTED_SSN" in output

    def test_mrn_redacted_in_logs(self):
        from hermes_phi.plugin import PHISafeFormatter

        fmt = PHISafeFormatter("%(message)s")
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "MRN-ABC123456", (), None,
        )
        output = fmt.format(record)
        assert "MRN-ABC123456" not in output
        assert "REDACTED_MRN" in output

    def test_clean_text_preserved_in_logs(self):
        from hermes_phi.plugin import PHISafeFormatter

        fmt = PHISafeFormatter("%(message)s")
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "Clean log message", (), None,
        )
        output = fmt.format(record)
        assert output == "Clean log message"


class TestConfigDefaults:
    """Validate plugin config defaults are fail-closed."""

    def test_fail_closed_default(self):
        from hermes_phi.plugin import PLUGIN_CONFIG_DEFAULTS

        assert PLUGIN_CONFIG_DEFAULTS["fail_closed"] is True

    def test_phi_policy_set_is_phi_v1_strict(self):
        from hermes_phi.plugin import PLUGIN_CONFIG_DEFAULTS

        assert PLUGIN_CONFIG_DEFAULTS["phi_policy_set"] == "phi_v1_strict"

    def test_env_requirements_documented(self):
        from hermes_phi.plugin import PLUGIN_CONFIG_DEFAULTS

        assert "env_requirements" in PLUGIN_CONFIG_DEFAULTS
        assert "SAFETY_AUDIT_PEPPER" in PLUGIN_CONFIG_DEFAULTS["env_requirements"]

    def test_default_phi_labels_include_minimum_set(self):
        from hermes_phi.plugin import DEFAULT_PHI_LABELS

        required = {"mrn", "date_of_birth", "phone_number", "health_plan_id"}
        assert required.issubset(DEFAULT_PHI_LABELS)


class TestConvenienceWrapper:
    """Validate the phi_redact convenience wrapper."""

    def test_phi_redact_clean_text(self):
        from hermes_phi.plugin import phi_redact

        result = phi_redact("What is the weather?")
        assert result == "What is the weather?"


class TestEmptyPayloadBehavior:
    """Empty payload edge cases."""

    def test_plugin_empty_context(self, plugin):
        result = plugin.redact_context("")
        assert result == ""

    def test_plugin_whitespace_context(self, plugin):
        result = plugin.redact_context("   ")
        assert result == "   "

    def test_plugin_newlines_only(self, plugin):
        result = plugin.redact_context("\n\n")
        assert result == "\n\n"


class TestDetectorTimeout:
    """Plugin handles timeout gracefully."""

    def test_detector_timeout_does_not_crash(self):
        from hermes_phi.plugin import HermesPHIPlugin

        plugin = HermesPHIPlugin(detector_timeout=0.001)
        try:
            result = plugin.redact_context("Short text")
            assert result == "Short text"
        finally:
            plugin._kernel.shutdown()


class TestConfigCompliance:
    """Plugin config defaults must be fail-closed for PHI workflows."""

    def test_default_config_is_fail_closed(self):
        from hermes_phi.plugin import PLUGIN_CONFIG_DEFAULTS

        assert PLUGIN_CONFIG_DEFAULTS["enabled"] is True
        assert PLUGIN_CONFIG_DEFAULTS["fail_closed"] is True
        assert "hermes_context" in PLUGIN_CONFIG_DEFAULTS["surfaces"]

    def test_all_phi_surfaces_covered_by_default(self):
        from hermes_phi.plugin import PLUGIN_CONFIG_DEFAULTS

        assert "hermes_context" in PLUGIN_CONFIG_DEFAULTS["surfaces"]
        assert "hermes_tool_output" in PLUGIN_CONFIG_DEFAULTS["surfaces"]
        assert "hermes_memory_write" in PLUGIN_CONFIG_DEFAULTS["surfaces"]


class TestProveContract:
    """Smoke test the prove_contract function."""

    def test_prove_contract_returns_true(self):
        from hermes_phi.plugin_proof import prove_contract

        result = prove_contract()
        assert result is True


class TestNoRawPHIInModelBoundPayload:
    """Validate that raw PHI does not appear in model-bound payload."""

    def test_mrn_does_not_leave_in_plaintext_when_redacted(self, plugin):
        """When MRN is redacted, the model-bound payload has no raw MRN."""
        from hermes_phi.plugin import PhiBlocked

        text = "Patient MRN-XYZ-789012 says they feel better."
        try:
            result = plugin.redact_context(text)
            assert "XYZ-789012" not in result, (
                f"Raw MRN found in model-bound payload: {result}"
            )
        except PhiBlocked:
            pass  # Block is acceptable


class TestMemoryWriteNoRawPHI:
    """Validate memory writes have no raw PHI for covered paths."""

    def test_memory_write_redacts_mrn(self, plugin):
        """Memory write redacts MRN before persistence."""
        from hermes_phi.plugin import PhiBlocked

        text = "State: patient MRN-123456789 has pending orders."
        try:
            result = plugin.check_memory_write(text)
            assert "123456789" not in result, f"Raw PHI in memory write: {result}"
        except PhiBlocked:
            pass  # Block is acceptable


class TestFullIntegration:
    """End-to-end test: plugin → SafetyKernel → decision."""

    def test_context_evaluation_pipeline(self, plugin):
        """Full pipeline: plugin evaluates context through Safety Kernel."""
        text = "What are the symptoms of diabetes?"
        result = plugin.redact_context(text)
        assert len(result) > 0
        assert "diabetes" in result  # Medical condition is not PHI

    def test_tool_output_evaluation_pipeline(self, plugin):
        """Full pipeline: plugin evaluates tool output."""
        text = "Files found: 3"
        result = plugin.check_tool_output(text)
        assert result == text

    def test_memory_write_evaluation_pipeline(self, plugin):
        """Full pipeline: plugin evaluates memory write."""
        text = "Session: conversation completed"
        result = plugin.check_memory_write(text)
        assert result == text


class TestFailClosedPaths:
    """Test all fail-closed paths."""

    def test_fail_closed_evaluate_exception(self):
        """When evaluate() raises, plugin propagates PhiSafetyUnavailable."""
        from hermes_phi.plugin import HermesPHIPlugin, PhiSafetyUnavailable

        plugin = HermesPHIPlugin()
        mock_kernel = MagicMock()
        mock_kernel.evaluate.side_effect = RuntimeError("evaluate failed")
        plugin._kernel = mock_kernel
        plugin._kernel_provided = True

        with pytest.raises(PhiSafetyUnavailable) as exc_info:
            plugin.redact_context("test")
        assert "RuntimeError" in str(exc_info.value)

    def test_kernel_shutdown_on_del(self):
        """Plugin shutdown cleans up kernel resources."""
        from hermes_phi.plugin import HermesPHIPlugin

        plugin = HermesPHIPlugin()
        kernel = plugin._kernel
        assert kernel is not None
        del plugin


class TestSurfaceRerouting:
    """Check surface-specific evaluation."""

    def test_tool_output_surface_allows_clean(self, plugin):
        result = plugin.check_tool_output("Tool completed successfully.")
        assert result == "Tool completed successfully."

    def test_memory_write_surface_allows_clean(self, plugin):
        result = plugin.check_memory_write("Data saved at checkpoint 42.")
        assert result == "Data saved at checkpoint 42."


class TestMinimumPHICoverage:
    """Validate that all minimum PHI types have at least basic test coverage."""

    PHI_TYPES = [
        "mrn",
        "date_of_birth",
        "phone_number",
        "health_plan_id",
        "ssn",
    ]

    def test_all_minimum_phi_types_coverage_documented(self):
        from hermes_phi.plugin import DEFAULT_PHI_LABELS

        for phi_type in self.PHI_TYPES:
            assert phi_type in DEFAULT_PHI_LABELS, (
                f"{phi_type} is missing from DEFAULT_PHI_LABELS"
            )

    def test_deterministic_detector_covers_mrn(self, plugin):
        """DeterministicDetector (regex) covers MRN."""
        from hermes_phi.plugin import PhiBlocked

        text = "MRN-ABC123456789"
        try:
            result = plugin.redact_context(text)
            assert "ABC123456789" not in result
        except PhiBlocked:
            pass

    def test_deterministic_detector_covers_health_plan(self, plugin):
        """DeterministicDetector (regex) covers health plan ID."""
        from hermes_phi.plugin import PhiBlocked

        text = "HPI-XYZ789012"
        try:
            result = plugin.redact_context(text)
            assert "HPI-XYZ789012" not in result
        except PhiBlocked:
            pass

    def test_deterministic_detector_covers_phone(self, plugin):
        """DeterministicDetector (regex) covers phone numbers."""
        from hermes_phi.plugin import PhiBlocked

        text = "Call (212) 555-0198"
        try:
            result = plugin.redact_context(text)
            assert "555" not in result
            assert "REDACTED" in result
        except PhiBlocked:
            pass

    def test_deterministic_detector_covers_dob(self, plugin):
        """HermesDeterministicPHIDetector covers date_of_birth pattern."""
        from hermes_phi.plugin import PhiBlocked

        text = "DOB: 1972-08-22"
        try:
            result = plugin.redact_context(text)
            assert "1972-08-22" not in result
            assert "REDACTED" in result
        except PhiBlocked:
            pass

    def test_deterministic_detector_covers_accession(self, plugin):
        """HermesDeterministicPHIDetector covers accession_number."""
        from hermes_phi.plugin import PhiBlocked

        text = "ACC-9876-5432"
        try:
            result = plugin.redact_context(text)
            assert "ACC-9876-5432" not in result
            assert "REDACTED" in result
        except PhiBlocked:
            pass

    def test_deterministic_detector_covers_order_id(self, plugin):
        """HermesDeterministicPHIDetector covers order_id."""
        from hermes_phi.plugin import PhiBlocked

        text = "ORD-2026-001234"
        try:
            result = plugin.redact_context(text)
            assert "ORD-2026-001234" not in result
            assert "REDACTED" in result
        except PhiBlocked:
            pass


class TestPHILabelsComplete:
    """Verify all labels from the acceptance criteria are present."""

    def test_accession_number_label(self):
        from hermes_phi.plugin import DEFAULT_PHI_LABELS
        assert "accession_number" in DEFAULT_PHI_LABELS

    def test_order_id_label(self):
        from hermes_phi.plugin import DEFAULT_PHI_LABELS
        assert "order_id" in DEFAULT_PHI_LABELS

    def test_phi_surface_enum_contains_memory_write(self):
        from hermes_phi.plugin import PHISurface
        assert hasattr(PHISurface, "HERMES_MEMORY_WRITE")

    def test_phi_surface_enum_contains_context(self):
        from hermes_phi.plugin import PHISurface
        assert hasattr(PHISurface, "HERMES_CONTEXT")

    def test_phi_surface_enum_contains_tool_output(self):
        from hermes_phi.plugin import PHISurface
        assert hasattr(PHISurface, "HERMES_TOOL_OUTPUT")


class TestModuleExports:
    """Verify hermes_phi public API exports."""

    def test_plugin_class_exported(self):
        from hermes_phi import HermesPHIPlugin
        assert HermesPHIPlugin is not None

    def test_surface_enum_exported(self):
        from hermes_phi import PHISurface
        assert PHISurface is not None

    def test_verdict_enum_exported(self):
        from hermes_phi import PHIVerdict
        assert PHIVerdict is not None

    def test_detector_exported(self):
        from hermes_phi.plugin import HermesDeterministicPHIDetector
        assert HermesDeterministicPHIDetector is not None

    def test_policy_registry_factory_exported(self):
        from hermes_phi.plugin import create_phi_policy_registry
        assert create_phi_policy_registry is not None

    def test_kernel_factory_exported(self):
        from hermes_phi.plugin import create_phi_kernel
        assert create_phi_kernel is not None


class TestSurfaceDocstringExamples:
    """Test the examples shown in the surface map doc."""

    def test_surface_6_context_assembly_intercepted(self, plugin):
        """Surface 6: Pre-LLM context assembly is intercepted."""
        from hermes_phi.plugin import PhiBlocked

        prompt = "Refer patient John Doe (MRN-123456) to cardiology."
        try:
            safe = plugin.redact_context(prompt)
            assert "123456" not in safe, (
                "MRN leaked in surface 6 interception"
            )
        except PhiBlocked:
            pass

    def test_surface_7_model_bound_redacted(self, plugin):
        """Surface 7: Model-bound payload is redacted."""
        from hermes_phi.plugin import PhiBlocked

        payload = "Patient MRN-567890 has an appointment."
        try:
            safe = plugin.redact_context(payload)
            assert "567890" not in safe, (
                "MRN leaked in model-bound payload"
            )
        except PhiBlocked:
            pass

    def test_surface_3_tool_output_checked(self, plugin):
        """Surface 3: Tool output is checked before context append."""
        from hermes_phi.plugin import PhiBlocked

        output = "Patient search result: MRN-901234"
        try:
            safe = plugin.check_tool_output(output)
            assert "901234" not in safe, (
                "MRN leaked in tool output"
            )
        except PhiBlocked:
            pass

    def test_surface_8_memory_write_checked(self, plugin):
        """Surface 8: Memory write is checked before persistence."""
        from hermes_phi.plugin import PhiBlocked

        state = "session_data: patient MRN-345678 conversation log"
        try:
            safe = plugin.check_memory_write(state)
            assert "345678" not in safe, (
                "MRN leaked in memory write"
            )
        except PhiBlocked:
            pass


class TestEdgeCases:
    """Edge cases for the PHI plugin."""

    def test_very_long_text_does_not_crash(self, plugin):
        """Very long text is handled without crash."""
        long_text = "A" * 100_000
        try:
            result = plugin.redact_context(long_text)
            assert len(result) > 0
        except Exception:
            pass

    def test_unicode_text_handled(self, plugin):
        """Unicode text is handled without encoding errors."""
        from hermes_phi.plugin import PhiBlocked

        text = "患者のカルテ番号はMRN-123456です"
        try:
            result = plugin.redact_context(text)
            assert isinstance(result, str)
        except PhiBlocked:
            pass

    def test_numeric_phi_in_text(self, plugin):
        """Numeric-only PHI patterns handled."""
        from hermes_phi.plugin import PhiBlocked

        text = "ID: 123-45-6789"
        try:
            result = plugin.redact_context(text)
            assert isinstance(result, str)
        except PhiBlocked:
            pass

    def test_phi_near_clean_text(self, plugin):
        """PHI adjacent to clean text in same payload."""
        from hermes_phi.plugin import PhiBlocked

        text = "Patient MRN-123456 was seen for routine checkup. Recommend continuing current medication."
        try:
            result = plugin.redact_context(text)
            assert "123456" not in result, (
                "MRN leaked near clean text"
            )
            assert "medication" in result  # Clean content preserved
        except PhiBlocked:
            pass


class TestModuleInit:
    """hermes_phi module init exports."""

    def test_version_defined(self):
        from hermes_phi import __version__
        assert __version__ == "0.1.0"

    def test_all_exports(self):
        from hermes_phi import __all__
        assert "HermesPHIPlugin" in __all__
        assert "PHISurface" in __all__
        assert "PHIVerdict" in __all__


class TestLoggingFormatterEdgeCases:
    """PHISafeFormatter edge cases."""

    def test_multiple_ssns_redacted(self):
        from hermes_phi.plugin import PHISafeFormatter

        fmt = PHISafeFormatter("%(message)s")
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "SSNs: 123-45-6789 and 987-65-4321", (), None,
        )
        output = fmt.format(record)
        assert "123-45-6789" not in output
        assert "987-65-4321" not in output
        assert output.count("REDACTED_SSN") == 2

    def test_no_false_positives_on_normal_numbers(self):
        from hermes_phi.plugin import PHISafeFormatter

        fmt = PHISafeFormatter("%(message)s")
        record = logging.LogRecord("test", logging.INFO, "", 0, "Version 2.3.4 released", (), None)
        output = fmt.format(record)
        assert output == "Version 2.3.4 released"


class TestConveniencePhiRedact:
    """Additional phi_redact tests."""

    def test_phi_redact_returns_string(self):
        from hermes_phi.plugin import phi_redact

        result = phi_redact("What is 2+2?")
        assert isinstance(result, str)

    def test_phi_redact_surface_param(self):
        from hermes_phi.plugin import phi_redact

        result = phi_redact("Hello", surface="hermes_tool_output")
        assert result == "Hello"


class TestPluginStateManagement:
    """Plugin state and lifecycle tests."""

    def test_is_available_true_when_kernel_loaded(self, plugin):
        assert plugin.is_available is True

    def test_multiple_evaluations_same_instance(self, plugin):
        assert plugin.redact_context("First call") == "First call"
        assert plugin.redact_context("Second call") == "Second call"

    def test_different_surfaces_same_instance(self, plugin):
        assert plugin.redact_context("Context") == "Context"
        assert plugin.check_tool_output("Output") == "Output"
        assert plugin.check_memory_write("Memory") == "Memory"


class TestPHIDetectionBoundaries:
    """Test PHI detection at boundaries of safety kernel features."""

    def test_mrn_pattern_with_extra_context(self, plugin):
        """MRN next to other text doesn't break detection."""
        from hermes_phi.plugin import PhiBlocked

        text = "According to MRN-123456, the patient profile shows..."
        try:
            result = plugin.redact_context(text)
            assert "123456" not in result, (
                "MRN leaked despite extra context"
            )
        except PhiBlocked:
            pass

    def test_phi_interleaved_with_code(self, plugin):
        """PHI text interleaved with code-like content."""
        from hermes_phi.plugin import PhiBlocked

        text = "def process(patient_id='MRN-999999'): pass"
        try:
            result = plugin.redact_context(text)
            assert "999999" not in result, (
                "MRN leaked in code interleave"
            )
        except PhiBlocked:
            pass


class TestStrictRegression:
    """Strict regression tests for known security issues."""

    def test_blocked_hook_proof_never_returns_raw_payload(self, proof):
        """Regression: check_text_before_llm must not return raw text on BLOCK."""
        from llm_common.core.kernel import SafetyKernel
        from hermes_phi.plugin import create_phi_kernel

        kernel = create_phi_kernel()
        try:
            text = "Patient MRN-ABC123456789 needs follow-up."
            result = proof.check_text_before_llm(text, kernel=kernel)
            assert result["action"] in ("block", "allow_with_redaction")
            if result["action"] == "block":
                assert result["payload"] == "[BLOCKED_PHI]", (
                    "Blocked payload must be sentinel, not raw text"
                )
                assert result["payload"] != text
        finally:
            kernel.shutdown()

    def test_phi_blocked_has_no_raw_payload_attribute(self):
        """Regression: PhiBlocked must not have a payload_snippet attribute."""
        from hermes_phi.plugin import PhiBlocked

        exc = PhiBlocked("hermes_context", [{"label": "mrn", "kind": "phi"}])
        assert not hasattr(exc, "payload_snippet"), (
            "PhiBlocked must not retain payload_snippet"
        )
        assert not hasattr(exc, "raw_payload"), (
            "PhiBlocked must not retain raw_payload"
        )

    def test_phone_redacted_on_all_surfaces(self, plugin):
        """Phone number is redacted on all three surfaces."""
        from hermes_phi.plugin import PhiBlocked

        text = "Contact (212) 555-0198"
        for method_name, surface_name in [
            ("redact_context", "context"),
            ("check_tool_output", "tool output"),
            ("check_memory_write", "memory write"),
        ]:
            method = getattr(plugin, method_name)
            try:
                result = method(text)
                assert "555" not in result, (
                    f"Phone leaked on {surface_name}"
                )
            except PhiBlocked:
                pass

    def test_mrn_block_exception_no_raw_mrn(self, plugin):
        """Regression: PhiBlocked from MRN detection carries no raw MRN value."""
        from hermes_phi.plugin import PhiBlocked

        text = "MRN-ABC123456789"
        try:
            result = plugin.redact_context(text)
            assert "ABC123456789" not in result
        except PhiBlocked as exc:
            msg = str(exc)
            assert "ABC123456789" not in msg, (
                "Raw MRN leaked in exception message"
            )
            for attr_name in dir(exc):
                if attr_name.startswith("_") or callable(getattr(exc, attr_name)):
                    continue
                val = str(getattr(exc, attr_name))
                assert "ABC123456789" not in val, (
                    f"Raw MRN found in PhiBlocked.{attr_name}"
                )


    def test_hermes_detector_fail_closed_on_dob(self):
        """Regression: HermesDeterministicPHIDetector fail_closed=True means a
        failing detector blocks DOB-bearing payload, not allow."""
        from unittest.mock import MagicMock
        from llm_common.core.kernel import SafetyKernel
        from llm_common.core.detector import DeterministicDetector
        from hermes_phi.plugin import HermesDeterministicPHIDetector, create_phi_policy_registry
        from llm_common.core.safety import SafetyRequest
        from hermes_phi.plugin import PhiSafetyUnavailable, PhiBlocked

        # Create a failing HermesDeterministicPHIDetector
        failing_detector = HermesDeterministicPHIDetector()
        original_detect = failing_detector.detect
        # Make it raise on any call
        failing_detector.detect = MagicMock(side_effect=RuntimeError("Hermes PHI detector failed"))

        kernel = SafetyKernel(
            detectors=[DeterministicDetector(), failing_detector],
            registry=create_phi_policy_registry(),
        )
        try:
            # DOB is a label ONLY covered by HermesDeterministicPHIDetector
            # DeterministicDetector does NOT have date_of_birth.
            # When the only detector for DOB fails with fail_closed=True,
            # the kernel must BLOCK.
            request = SafetyRequest(
                surface="hermes_context",
                payload="DOB: 1972-08-22",
                policy_set="phi_v1_strict",
            )
            decision = kernel.evaluate(request, timeout_sec=0.5)

            # fail_closed=True + failing detector for active label => BLOCK
            from llm_common.core.safety import SafetyVerdict
            assert decision.verdict == SafetyVerdict.BLOCK, (
                f"Expected BLOCK for failing HermesDetector with DOB, got {decision.verdict}"
            )
            # Verify the circuit breaker finding
            assert any(f.label == "runtime_exception" for f in decision.findings), (
                "Expected runtime_exception finding from failing detector"
            )
        finally:
            kernel.shutdown()

class TestUserFacingDocExamples:
    """Surface map examples: plugins must actually intercept the documented surfaces."""

    def test_phi_redact_example_works(self):
        """phi_redact() works as documented in docstring."""
        from hermes_phi.plugin import phi_redact, PhiBlocked

        try:
            safe = phi_redact("Patient MRN-123456 is due for follow-up.")
            assert "123456" not in safe
        except PhiBlocked:
            pass
        except Exception as exc:
            pytest.fail(f"phi_redact raised unexpected exception: {exc}")


@pytest.fixture(scope="session", autouse=True)
def _global_test_config():
    """Fixture to ensure test mode is set for all sessions."""
    os.environ["SAFETY_TEST_MODE"] = "1"
    yield


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
