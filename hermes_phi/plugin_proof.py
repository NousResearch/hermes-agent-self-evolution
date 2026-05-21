"""
Hermes Plugin Hook Spike — adapter proof that Safety Kernel can intercept
Hermes text/context before LLM-bound payload leaves the process.

This is Part B (bd-jqrzp.2) of the Hermes PHI safety implementation.

Purpose:
  Prove that a hook can:
  1. Receive representative Hermes text/context.
  2. Call the llm-common Safety Kernel with surface="hermes_context" or
     surface="hermes_tool_output".
  3. Return redacted text when verdict is ALLOW.
  4. Block model-bound text when verdict is BLOCK.
  5. Expose test seams for verification.

This proof does NOT install itself as a Hermes Agent plugin — it validates
the adapter contract so that the production plugin (hermes_phi/plugin.py) can
be integrated confidently.
"""

import logging
from typing import Optional

from llm_common.core.kernel import SafetyKernel
from llm_common.core.safety import SafetyRequest, SafetyVerdict

logger = logging.getLogger(__name__)


# ── Surface Identifiers ──────────────────────────────────────────────────

SURFACE_HERMES_CONTEXT = "hermes_context"
SURFACE_HERMES_TOOL_OUTPUT = "hermes_tool_output"

# ── Hook Proof ───────────────────────────────────────────────────────────


def check_text_before_llm(
    text: str,
    surface: str = SURFACE_HERMES_CONTEXT,
    kernel: Optional[SafetyKernel] = None,
    policy_set: str = "phi_v1",
    timeout_sec: float = 0.5,
) -> dict:
    """Evaluate text against the Safety Kernel before it reaches an LLM.

    This is the core hook contract tested by this spike.

    Args:
        text: The text payload to evaluate (user input, tool output, assembled context).
        surface: Safety Kernel surface identifier.
        kernel: Optional pre-configured SafetyKernel. If None, creates a default one.
        policy_set: Policy set to evaluate against.
        timeout_sec: Safety Kernel timeout per detector.

    Returns:
        dict with keys:
            - "action": "allow" | "block" | "allow_with_redaction"
            - "payload": str — the (possibly redacted) text
            - "verdict": SafetyVerdict
            - "findings": list of finding dicts
            - "audit_ref": str or None

    Raises:
        RuntimeError: If Safety Kernel is unavailable and fail-closed is required.
        ImportError: If llm_common is not installed.
    """
    if kernel is None:
        kernel = SafetyKernel()

    request = SafetyRequest(
        surface=surface,
        payload=text,
        policy_set=policy_set,
    )

    decision = kernel.evaluate(request, timeout_sec=timeout_sec)

    result: dict = {
        "verdict": decision.verdict,
        "findings": [f.model_dump() for f in decision.findings],
        "audit_ref": decision.audit_ref,
    }

    if decision.verdict == SafetyVerdict.BLOCK:
        logger.info(
            "Safety Kernel BLOCKED text on surface=%s (%d findings)",
            surface,
            len(decision.findings),
        )
        result["action"] = "block"
        result["payload"] = text  # Return original for diagnostics; caller must suppress.
    elif decision.redacted_payload is not None:
        result["action"] = "allow_with_redaction"
        result["payload"] = decision.redacted_payload
    else:
        result["action"] = "allow"
        result["payload"] = text

    return result


# ── Test Seam: Verifiable contract assertions ──────────────────────────

# These constants let tests assert contract behavior without importing
# the full plugin.

PHI_TEST_SAMPLES = {
    "mrn": (
        "Patient MRN-ABC123456 has a follow-up scheduled for next week.",
        ["mrn"],
    ),
    "ssn": (
        "The patient's SSN is 123-45-6789 and DOB is 1985-03-15.",
        ["ssn"],
    ),
    "patient_name": (
        "John Smith was admitted on 2026-01-15 with chest pain.",
        ["person", "patient_name"],
    ),
    "phone": (
        "Contact patient at (555) 123-4567 for appointment reminder.",
        ["phone_number"],
    ),
    "health_plan_id": (
        "Health plan ID: HPI-XYZ789012, group number: 4455.",
        ["health_plan_id"],
    ),
    "address": (
        "Patient lives at 123 Main Street, Springfield, IL 62701.",
        ["address"],
    ),
    "date_of_birth": (
        "Date of birth: 1972-08-22. Patient is 52 years old.",
        ["date_of_birth"],
    ),
    "accession_number": (
        "Accession number: ACC-9876-5432 for the MRI study.",
        ["accession_number"],
    ),
    "order_id": (
        "Order ID: ORD-2026-001234 for complete blood count.",
        ["order_id"],
    ),
    "clean_text": (
        "What are the current treatment guidelines for hypertension?",
        [],
    ),
}


def get_phi_test_sample(name: str) -> tuple[str, list[str]]:
    """Retrieve a PHI test sample and its expected labels."""
    return PHI_TEST_SAMPLES[name]


def prove_contract() -> bool:
    """Run a quick smoke-test of the hook contract.

    Returns True if all basic contract assertions pass.
    Intended for CI smoke tests, not full test coverage.
    """
    kernel = SafetyKernel()
    clean = check_text_before_llm("What is the capital of France?", kernel=kernel)
    assert clean["action"] == "allow", f"Expected allow, got {clean['action']}"

    phi_text = "Patient MRN-ABC123456 needs follow-up."
    result = check_text_before_llm(phi_text, kernel=kernel)
    assert result["action"] in ("allow_with_redaction", "block"), (
        f"Expected redaction or block for PHI, got {result['action']}"
    )
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prove_contract()
    print("Hook contract smoke test passed.")

