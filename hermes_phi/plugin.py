"""
Hermes Local PHI Redaction Plugin — production plugin using llm-common Safety Kernel.

This is Part C (bd-jqrzp.3) of the Hermes PHI safety implementation.

PHI Coverage (deterministic, no GLiNER required):
  - MRN (medical record numbers) — BLOCK
  - Date of birth / YYYY-MM-DD — REDACT
  - Phone number — REDACT
  - Health plan IDs — BLOCK
  - Accession number (ACC-*) — REDACT
  - Order ID (ORD-*) — REDACT
  - SSN — BLOCK
  - Email — REDACT

PHI Coverage (requires GLiNER via llm-common[gliner]):
  - Patient names / patient_name
  - Patient address / patient_address
  - Location-specific PHI

Fail-Closed Contract:
  - If Safety Kernel is not installed (ImportError) → BLOCK all PHI-covered surfaces.
  - If SafetyKernel.evaluate() raises → BLOCK the payload.
  - SAFETY_AUDIT_PEPPER env var is REQUIRED for production runtime (see docs).
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

from llm_common.core.detector import BaseDetector, DeterministicDetector
from llm_common.core.kernel import SafetyKernel
from llm_common.core.policy import SafetyPolicyRegistry, PolicySet, PolicyRule
from llm_common.core.safety import SafetyAction, SafetyRequest, SafetyVerdict, SafetyFinding

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
    """Raised when the Safety Kernel blocks PHI-containing payload.

    NEVER carries raw PHI in any attribute or string representation.
    Only surface, finding labels, and finding count are exposed.
    """

    def __init__(self, surface: str, findings: list[dict]):
        self.surface = surface
        self.findings = findings
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


# ── Local PHI Detector (deterministic, no GLiNER required) ───────────────


class HermesDeterministicPHIDetector(BaseDetector):
    """Deterministic PHI detector covering date-of-birth and accession/order patterns.

    Complements llm-common DeterministicDetector by adding PHI-specific patterns
    that are not covered by the shared library's regex set. Detected labels always
    produce kind='phi' to route correctly under phi_v1_strict policy.

    Covered labels:
      - date_of_birth: YYYY-MM-DD / YYYY/MM/DD (basic pattern)
      - accession_number: ACC-XXXX-XXXX
      - order_id: ORD-XXXX-XXXX
    """

    def __init__(self) -> None:
        self.patterns: dict[str, re.Pattern] = {
            "date_of_birth": re.compile(
                r"\b\d{4}[-/]\d{2}[-/]\d{2}\b"  # YYYY-MM-DD / YYYY/MM/DD
            ),
            "accession_number": re.compile(
                r"\bACC[-\s]?[A-Z0-9-]{6,16}\b",
                re.IGNORECASE,
            ),
            "order_id": re.compile(
                r"\bORD[-\s]?[A-Z0-9-]{6,16}\b",
                re.IGNORECASE,
            ),
        }

    @property
    def fail_closed(self) -> bool:
        return True

    def get_supported_labels(self, labels: list[str]) -> list[str]:
        return [lbl for lbl in labels if lbl in self.patterns]

    def get_default_labels(self, kind: str) -> list[str]:
        if kind == "phi":
            return list(self.patterns.keys())
        return []

    def get_kind_for_label(self, label: str) -> str:
        return "phi"

    def detect(self, text: str, labels: list[str]) -> list[SafetyFinding]:
        findings: list[SafetyFinding] = []
        for label in labels:
            pattern = self.patterns.get(label)
            if not pattern:
                continue
            for match in pattern.finditer(text):
                findings.append(
                    SafetyFinding(
                        kind="phi",
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        confidence=1.0,
                        detector="hermes_deterministic_phi",
                        evidence_ref=None,
                        document_locator=None,
                    )
                )
        return findings


# ── Policy Registry ─────────────────────────────────────────────────────


def create_phi_policy_registry() -> SafetyPolicyRegistry:
    """Create a SafetyPolicyRegistry pre-configured for Hermes PHI detection.

    The 'phi_v1_strict' policy covers deterministic-only detection:
      - BLOCK: MRN, health_plan_id, SSN
      - REDACT: phone_number, email, date_of_birth, accession_number, order_id,
                and any generic phi label
    """
    registry = SafetyPolicyRegistry()

    phi_strict = PolicySet(
        name="phi_v1_strict",
        version="1.0.0",
        rules=[
            # BLOCK structured identifiers
            PolicyRule(kind="phi", label="mrn", action=SafetyAction.BLOCK, confidence_threshold=0.3),
            PolicyRule(kind="phi", label="health_plan_id", action=SafetyAction.BLOCK, confidence_threshold=0.3),
            PolicyRule(kind="phi", label="date_of_birth", action=SafetyAction.REDACT, confidence_threshold=0.3),
            PolicyRule(kind="phi", label="accession_number", action=SafetyAction.REDACT, confidence_threshold=0.3),
            PolicyRule(kind="phi", label="order_id", action=SafetyAction.REDACT, confidence_threshold=0.3),
            # PII labels that are PHI-relevant in medical context
            PolicyRule(kind="pii", label="ssn", action=SafetyAction.BLOCK, confidence_threshold=0.3),
            PolicyRule(kind="pii", label="phone_number", action=SafetyAction.REDACT, confidence_threshold=0.3),
            PolicyRule(kind="pii", label="email", action=SafetyAction.REDACT, confidence_threshold=0.3),
            # Generic catch-all for any other phi label (GLiNER patient_name, etc.)
            PolicyRule(kind="phi", label=None, action=SafetyAction.REDACT, confidence_threshold=0.3),
        ],
        fallback_action=SafetyAction.ALLOW,
    )
    registry.register(phi_strict)

    for surface in ("hermes_context", "hermes_tool_output", "hermes_memory_write"):
        registry.register_surface_policy(surface, "phi_v1_strict")

    return registry


def create_phi_kernel() -> SafetyKernel:
    """Create a SafetyKernel configured for Hermes PHI surfaces."""
    from llm_common.core.detector import DeterministicDetector
    try:
        import importlib
        has_gliner = importlib.util.find_spec("gliner") is not None
        if has_gliner:
            from llm_common.core.detector import GlinerDetector
    except ImportError:
        has_gliner = False

    detectors: list[BaseDetector] = [DeterministicDetector(), HermesDeterministicPHIDetector()]
    if has_gliner:
        detectors.append(GlinerDetector(fail_closed=False))

    registry = create_phi_policy_registry()
    return SafetyKernel(detectors=detectors, registry=registry)


# ── Plugin ────────────────────────────────────────────────────────────────


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

    Production requirement: SAFETY_AUDIT_PEPPER (or SAFETY_KERNEL_SALT) must
    be set in the environment. If missing, SafetyKernel.evaluate() will fail
    with a RuntimeError for any payload that produces findings, which the
    plugin handles by raising PhiSafetyUnavailable (fail-closed).
    """

    def __init__(
        self,
        kernel: Optional[SafetyKernel] = None,
        detector_timeout: float = 0.5,
    ):
        self._detector_timeout = detector_timeout

        if kernel is not None:
            self._kernel = kernel
            self._kernel_provided = True
        else:
            try:
                self._kernel = create_phi_kernel()
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

        if len(text) == 0:
            from llm_common.core.safety import SafetyDecision, SafetyVerdict
            return SafetyDecision(verdict=SafetyVerdict.ALLOW, redacted_payload="", findings=[])

        request = SafetyRequest(
            surface=surface,
            payload=text,
            policy_set="phi_v1_strict",
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
    "phi_policy_set": "phi_v1_strict",
    "detector_timeout_sec": 0.5,
    "surfaces": [
        "hermes_context",
        "hermes_tool_output",
        "hermes_memory_write",
    ],
    "env_requirements": {
        "SAFETY_AUDIT_PEPPER": "Required for production audit logging. Set SAFETY_TEST_MODE=1 for dev/testing only.",
    },
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
        (re.compile(r"\bACC[-\s]?[A-Z0-9-]{6,16}\b", re.IGNORECASE), "[REDACTED_ACC]"),
        (re.compile(r"\bORD[-\s]?[A-Z0-9-]{6,16}\b", re.IGNORECASE), "[REDACTED_ORD]"),
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
    "HermesDeterministicPHIDetector",
    "create_phi_policy_registry",
    "create_phi_kernel",
]
