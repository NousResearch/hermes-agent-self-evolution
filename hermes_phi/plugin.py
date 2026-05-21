"""
Hermes Local PHI Redaction Plugin — production plugin using llm-common Safety Kernel.

This is Part C (bd-jqrzp.3) of the Hermes PHI safety implementation.

PHI Coverage:
  - MRN (medical record numbers)
  - DOB / date of birth
  - Address / patient address
  - Phone number
  - Patient names
  - Health plan IDs
  - Accession numbers
  - Order IDs
  - SSN (handled by llm-common policy as BLOCK)
  - Email (mapped to REDACT via pii_v1)

Fail-Closed Contract:
  - If Safety Kernel is not installed (ImportError) → BLOCK all PHI-covered surfaces.
  - If SafetyKernel.evaluate() raises → BLOCK the payload.
  - If audit sink fails → the decision itself is still returned; audit failure is logged
    and appended as a finding (per Safety Kernel's own contract).

Behavior:
  - redact_context(): Evaluate assembled context before LLM call. Return redacted
    payload on ALLOW, raise PhiBlocked on BLOCK.
  - check_tool_output(): Evaluate tool output before appending to context.
    Return redacted text on ALLOW, raise PhiBlocked on BLOCK.
  - check_memory_write(): Evaluate text before persisting to durable storage.
    Return redacted on ALLOW, raise PhiBlocked on BLOCK.
"""

import enum
import logging
import re
from typing import Optional, Union

from llm_common.core.kernel import SafetyKernel
from llm_common.core.safety import SafetyRequest, SafetyVerdict

logger = logging.getLogger(__name__)


class PHISurface(str, enum.Enum):
    """Identifiers for the call site, passed to the Safety Kernel surface field."""

    HERMES_CONTEXT = "hermes_context"
    HERMES_TOOL_OUTPUT = "hermes_tool_output"
    HERMES_MEMORY_WRITE = "hermes_memory_write"


class PHIVerdict(str, enum.Enum):
    """Simplified verdict returned to the calling code."""

    ALLOW = "allow"
    BLOCK = "block"
    ALLOW_WITH_REDACTION = "allow_with_redaction"


class PhiBlocked(Exception):
    """Raised when the Safety Kernel blocks PHI-containing payload."""

    def __init__(self, surface: str, findings: list[dict], payload_snippet: str = ""):
        self.surface = surface
        self.findings = findings
        self.payload_snippet = payload_snippet
        labels = ", ".join(f["label"] for f in findings)
        super().__init__(
            f"PHI blocked on surface={surface}: {labels}"
        )


class PhiSafetyUnavailable(Exception):
    """Raised when Safety Kernel is unavailable on a PHI surface (fail-closed)."""

    def __init__(self, surface: str, reason: str):
        self.surface = surface
        super().__init__(
            f"PHI safety unavailable on surface={surface}: {reason}"
        )


class HermesPHIPlugin:
    """Plugin that evaluates Hermes text/context for PHI before LLM exposure.

    Typical usage:
        plugin = HermesPHIPlugin()

        # Before LLM call:
        safe_context = plugin.redact_context(assembled_prompt)

        # Before appending tool output:
        safe_tool_output = plugin.check_tool_output(tool_result)

        # Before persisting memory:
        safe_memory = plugin.check_memory_write(state_payload)
    """

    def __init__(
        self,
        kernel: Optional[SafetyKernel] = None,
        phi_policy_set: str = "phi_v1",
        detector_timeout: float = 0.5,
    ):
        self._phi_policy_set = phi_policy_set
        self._detector_timeout = detector_timeout

        if kernel is not None:
            self._kernel = kernel
            self._kernel_provided = True
        else:
            try:
                self._kernel = SafetyKernel()
                self._kernel_provided = True
            except ImportError as exc:
                self._kernel = None  # type: ignore[assignment]
                self._kernel_provided = False
                logger.critical(
                    "Safety Kernel unavailable (ImportError: %s). "
                    "PHI plugin WILL BLOCK all payloads (fail-closed).",
                    exc,
                )
            except Exception as exc:
                self._kernel = None  # type: ignore[assignment]
                self._kernel_provided = False
                logger.critical(
                    "Safety Kernel initialization failed: %s. "
                    "PHI plugin WILL BLOCK all payloads (fail-closed).",
                    exc,
                )

    def __del__(self) -> None:
        if hasattr(self, "_kernel") and self._kernel is not None:
            self._kernel.shutdown(wait=False)

    # ── Public API ──────────────────────────────────────────────────────

    def redact_context(
        self,
        text: str,
        surface: Union[str, PHISurface] = PHISurface.HERMES_CONTEXT,
    ) -> str:
        """Evaluate assembled context before it reaches the LLM.

        Args:
            text: The assembled prompt/context text.
            surface: The calling surface identifier.

        Returns:
            Redacted text if verdict is ALLOW.

        Raises:
            PhiBlocked: If Safety Kernel returns BLOCK.
            PhiSafetyUnavailable: If Safety Kernel is unavailable (fail-closed).
        """
        decision = self._evaluate(str(surface.value) if isinstance(surface, PHISurface) else surface, text)

        if decision.verdict == SafetyVerdict.BLOCK:
            findings = [f.model_dump() for f in decision.findings]
            raise PhiBlocked(
                surface=str(surface),
                findings=findings,
                payload_snippet=text[:200],
            )

        if decision.redacted_payload is not None:
            return str(decision.redacted_payload)

        return text

    def check_tool_output(
        self,
        text: str,
        surface: Union[str, PHISurface] = PHISurface.HERMES_TOOL_OUTPUT,
    ) -> str:
        """Evaluate tool output before it is appended to conversation context.

        Same contract as redact_context() with a different surface for policy routing.
        """
        return self.redact_context(text, surface=surface)

    def check_memory_write(
        self,
        text: str,
        surface: Union[str, PHISurface] = PHISurface.HERMES_MEMORY_WRITE,
    ) -> str:
        """Evaluate text before it is persisted to durable storage.

        Same contract as redact_context() with a different surface for policy routing.
        """
        return self.redact_context(text, surface=surface)

    # ── Internal ────────────────────────────────────────────────────────

    def _evaluate(self, surface: str, text: str) -> "SafetyDecision":
        """Execute the Safety Kernel evaluation, enforcing fail-closed."""
        if self._kernel is None:
            raise PhiSafetyUnavailable(
                surface=surface,
                reason="SafetyKernel not initialized (missing dependency or init failure)",
            )

        # Strip to safety check — avoid passing oversize context verbatim
        if len(text) == 0:
            from llm_common.core.safety import SafetyDecision, SafetyVerdict
            return SafetyDecision(verdict=SafetyVerdict.ALLOW, redacted_payload="", findings=[])

        request = SafetyRequest(
            surface=surface,
            payload=text,
            policy_set=self._phi_policy_set,
        )

        try:
            decision = self._kernel.evaluate(request, timeout_sec=self._detector_timeout)
        except Exception as exc:
            logger.error("Safety Kernel evaluate() raised %s: %s", type(exc).__name__, exc)
            raise PhiSafetyUnavailable(
                surface=surface,
                reason=f"SafetyKernel.evaluate() raised {type(exc).__name__}: {exc}",
            )

        return decision

    # ── Diagnostics ─────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Whether the Safety Kernel is loaded and operational."""
        return self._kernel is not None and self._kernel_provided


# ── Standalone Usage ─────────────────────────────────────────────────────


def phi_redact(text: str, surface: str = "hermes_context") -> str:
    """Drop-in convenience wrapper: redact text before LLM call.

    Recommended for quick integration:
        safe = phi_redact(prompt)
        response = client.chat.completions.create(messages=[{"role": "user", "content": safe}])

    Raises:
        PhiBlocked: If PHI was found and blocked.
        PhiSafetyUnavailable: If Safety Kernel is unavailable (fail-closed).
    """
    plugin = HermesPHIPlugin()
    return plugin.redact_context(text, surface=surface)


# ── Config Defaults ──────────────────────────────────────────────────────

DEFAULT_PHI_LABELS = frozenset({
    "mrn",
    "date_of_birth",
    "patient_address",
    "phone_number",
    "patient_name",
    "health_plan_id",
    "accession_number",
    "order_id",
    "ssn",
})

PLUGIN_CONFIG_DEFAULTS = {
    "enabled": True,
    "fail_closed": True,
    "phi_policy_set": "phi_v1",
    "detector_timeout_sec": 0.5,
    "surfaces": [
        "hermes_context",
        "hermes_tool_output",
        "hermes_memory_write",
    ],
}

# ── PHI Incident Logging Filter ──────────────────────────────────────────


class PHISafeFormatter(logging.Formatter):
    """Logging formatter that strips PHI patterns from log messages.

    This is a best-effort defense for Surface 10 (logs/traces). It runs after
    the message is formatted but before output. It is not a replacement for
    evaluating text before logging — it is a safety net.
    """

    _PHI_PATTERNS = [
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),  # SSN
        (re.compile(r"\bMRN[-\s]?[A-Z0-9-]{6,12}\b", re.IGNORECASE), "[REDACTED_MRN]"),
        (re.compile(r"\bHPI?[-\s]?[A-Z0-9-]{6,12}\b", re.IGNORECASE), "[REDACTED_HPI]"),
    ]

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        for pattern, replacement in self._PHI_PATTERNS:
            msg = pattern.sub(replacement, msg)
        return msg


def install_phi_safe_logging() -> None:
    """Install the PHI-safe log formatter on the root logger.

    Call once at application startup. This provides a safety net for logs
    but does not guarantee no PHI leakage through logs (known gap).
    """
    handler = logging.getLogger().handlers[0] if logging.getLogger().handlers else None
    if handler is not None:
        handler.setFormatter(PHISafeFormatter(handler.formatter._fmt or "%(message)s"))


__all__ = [
    "HermesPHIPlugin",
    "PHISurface",
    "PHIVerdict",
    "PhiBlocked",
    "PhiSafetyUnavailable",
    "phi_redact",
    "DEFAULT_PHI_LABELS",
    "PLUGIN_CONFIG_DEFAULTS",
    "PHISafeFormatter",
    "install_phi_safe_logging",
]

