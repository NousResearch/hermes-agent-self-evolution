"""Shared fixtures for tests/tools/.

`reset_root_logger_handlers` tears down FileHandlers that `evolve()` adds
to the root logger so handlers don't leak across tests. The orchestrator
attaches a per-run FileHandler unconditionally; without this fixture the
handler would survive into the next test and continue receiving log
records (slowing the suite + potentially failing once the run dir is
cleaned up).
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True)
def reset_root_logger_handlers():
    root = logging.getLogger()
    before = list(root.handlers)
    yield
    for h in list(root.handlers):
        if h not in before:
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
