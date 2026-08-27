"""Phase 4 — evolution of tool implementation code.

The mutation engine (``darwinian_evolver``) is AGPL-3.0 and cannot register an
external problem: its ``problems/registry.py`` holds a hardcoded dict and the
CLI restricts ``--problem`` to that dict's keys. Defining a Hermes problem
therefore means importing AGPL code, so that code lives in a separate
AGPL-licensed sidecar package and this MIT tree only talks to it over a
subprocess boundary. See :mod:`evolution.code.sidecar`.

What lives here is MIT and engine-independent:

* :mod:`evolution.code.admission` — the sandboxed gate a candidate must pass
  before it is scored, with held-out checks the mutator never sees.
* :mod:`evolution.code.targets` — choosing what to evolve, and turning
  recorded verification evidence into replayable checks.
* :mod:`evolution.code.evolve_code` — the orchestrator and CLI.
"""

from evolution.code.admission import (
    AdmissionGate,
    AdmissionVerdict,
    CheckResult,
    CommandCheck,
    PytestCheck,
    RecordedCommandCheck,
    build_default_gate,
    materialize_candidate,
)
from evolution.code.targets import CodeTarget, TargetError, resolve_targets, suggest_targets

__all__ = [
    "AdmissionGate",
    "AdmissionVerdict",
    "CheckResult",
    "CodeTarget",
    "CommandCheck",
    "PytestCheck",
    "RecordedCommandCheck",
    "TargetError",
    "build_default_gate",
    "materialize_candidate",
    "resolve_targets",
    "suggest_targets",
]
