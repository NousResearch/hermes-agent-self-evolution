"""OAuth helper utilities for NousLM.

Currently only consumed by ``NousLM`` and ``_maybe_resolve_nous_lm``.
Lives as its own module so the next OAuth provider that needs in-memory
refresh has somewhere obvious to drop shared utilities without bloating
either consumer's file.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional


def parse_iso_or_epoch(value: Any) -> Optional[float]:
    """Coerce an expires_at value into Unix epoch seconds.

    Different OAuth providers serialize token expiry in different shapes:

      * Nous Portal stores ISO 8601 strings ("2026-05-15T10:30:00+00:00")
        in ``~/.hermes/auth.json``.
      * Codex stores Unix epoch floats (or decodes from a JWT ``exp`` claim).
      * Older or hand-edited entries may omit it entirely.

    Returns the equivalent Unix epoch float, or None when the value is
    missing, malformed, has no parseable shape, or fails sanity checks
    (non-finite, negative, or naive datetime that would silently pull
    in the host TZ instead of the intended UTC).

    Callers treat None as "unknown" — typically meaning "trigger a
    refresh" defensively.
    """
    if value is None or isinstance(value, bool):
        # bool is a subclass of int; reject before the numeric path so a
        # stray ``True`` isn't silently turned into 1.0 epoch seconds.
        return None
    if isinstance(value, (int, float)):
        return _validated(float(value))
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # ISO 8601 — Python's fromisoformat handles "+00:00" but not the
        # bare "Z" suffix common in OpenAI-shaped responses.
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            dt = None
        if dt is not None:
            if dt.tzinfo is None:
                # Naive datetime: Python's .timestamp() would interpret
                # in the host's local timezone, which silently corrupts
                # the skew window by hours on non-UTC hosts. Reject so
                # the caller treats it as "unknown" rather than producing
                # a confidently-wrong epoch.
                return None
            return _validated(dt.timestamp())
        # Numeric-looking string ("1747299600") — treat as epoch seconds.
        try:
            return _validated(float(s))
        except ValueError:
            return None
    return None


def _validated(epoch: float) -> Optional[float]:
    """Reject inf, nan, and negative epoch values.

    ``inf`` would make every skew check evaluate ``something >= inf`` →
    False, so the token would be treated as eternally fresh and never
    refreshed (silent "wrong answer" failure). ``nan`` has the same
    failure mode (all comparisons against nan are False). Negatives are
    structurally absurd for an expires_at and most likely indicate a
    parser bug upstream.
    """
    if not math.isfinite(epoch) or epoch < 0:
        return None
    return epoch
