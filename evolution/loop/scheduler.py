"""Scheduler — decide which evolution phases to run."""

PHASE_ORDER = ["skills", "tools", "guidance", "params"]

PHASE_DEPENDENCIES = {
    "skills": [],
    "tools": [],
    "guidance": [],
    "params": ["tools"],
}


def select_pending_phases(
    latest_per_phase: dict,
    force: bool = False,
    only: list = None,
) -> list:
    """Return ordered list of phase names that should be run."""
    candidates = PHASE_ORDER if only is None else [p for p in PHASE_ORDER if p in only]
    if force:
        return list(candidates)
    return [p for p in candidates if p not in latest_per_phase]
