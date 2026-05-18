"""Mutation engines for Phase 4 candidates.

Two engines live behind a small protocol:

- ``InternalMutator`` — Apache-clean DSPy-driven proposer that takes the baseline
  source plus a bug brief and emits a candidate file. Default engine.
- ``ExternalDarwinianEvolverMutator`` — shells out to the external Darwinian
  Evolver CLI (AGPL v3, kept off-process). Not wired yet; raises with a clear
  install hint so users know what's missing.

``InternalMutator.propose`` always returns valid Python source. If the model
returns text wrapped in a fenced code block, that wrapper is stripped before the
result is handed back to the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import shutil
import subprocess
from typing import Protocol

import dspy


class MutationProposer(Protocol):
    def propose(
        self,
        *,
        baseline_source: str,
        bug_brief: str,
        iteration: int,
    ) -> str:
        """Return a full mutated source file as a string."""
        ...


@dataclass(frozen=True)
class MutationResult:
    """Convenience wrapper if a caller wants metadata alongside the source."""

    source: str
    engine: str
    iteration: int


_FENCE_RE = re.compile(r"^```(?:python|py)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def strip_fenced_source(text: str) -> str:
    """If ``text`` is a fenced code block, return its contents; else return as-is."""
    if not text:
        return ""
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group(1)
    return stripped


class _ProposeMutationSignature(dspy.Signature):
    """Propose a minimal, surgical edit to a Python tool file.

    Constraints (the orchestrator will reject the candidate otherwise):
    - DO NOT change any public function signatures.
    - DO NOT add/remove/rename any ``registry.register(...)`` or
      ``register_tool(...)`` calls.
    - DO NOT decrease the number of ``try``/``except`` blocks (no removing
      existing error handling, even if you think it's redundant).
    - DO NOT add network calls, subprocesses, or imports outside the stdlib
      unless they already existed in the baseline.

    Output the COMPLETE mutated file as raw Python source (no prose, no fences).
    """

    baseline_source: str = dspy.InputField(desc="The original tool file in full")
    bug_brief: str = dspy.InputField(
        desc="Plain-English description of the bug / edge case to address",
    )
    iteration: int = dspy.InputField(desc="Which iteration this candidate is")
    mutated_source: str = dspy.OutputField(desc="The full mutated Python file")


class InternalMutator:
    """DSPy-driven mutator that runs on the configured optimizer model."""

    def __init__(self, *, model_name: str):
        self.model_name = model_name
        self._lm = dspy.LM(model_name)
        self._predict = dspy.Predict(_ProposeMutationSignature)

    def propose(
        self,
        *,
        baseline_source: str,
        bug_brief: str,
        iteration: int,
    ) -> str:
        with dspy.context(lm=self._lm):
            result = self._predict(
                baseline_source=baseline_source,
                bug_brief=bug_brief,
                iteration=iteration,
            )
        return strip_fenced_source(getattr(result, "mutated_source", "") or "")


class ExternalDarwinianEvolverMutator:
    """Shell-out hook for the external Darwinian Evolver CLI (AGPL v3).

    The orchestrator instantiates this lazily and only when the user explicitly
    asks for ``--engine darwinian-evolver``. If the binary is missing we surface
    a clean error instead of importing AGPL code into this Apache-licensed repo.
    """

    BINARY_NAME = "darwinian-evolver"
    INSTALL_HINT = (
        "Install the Darwinian Evolver CLI separately (AGPL v3) and ensure the "
        "`darwinian-evolver` binary is on PATH, or rerun with --engine internal."
    )

    def __init__(self):
        if shutil.which(self.BINARY_NAME) is None:
            raise RuntimeError(
                f"Darwinian Evolver binary not found on PATH. {self.INSTALL_HINT}",
            )

    def propose(
        self,
        *,
        baseline_source: str,
        bug_brief: str,
        iteration: int,
    ) -> str:
        cmd = [self.BINARY_NAME, "propose", "--iteration", str(iteration)]
        try:
            proc = subprocess.run(
                cmd,
                input=f"# BUG BRIEF\n{bug_brief}\n# BASELINE\n{baseline_source}\n",
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"darwinian-evolver invocation failed: {exc}") from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"darwinian-evolver exited {proc.returncode}: "
                f"{proc.stderr.strip() or proc.stdout.strip()}",
            )
        return strip_fenced_source(proc.stdout)


def select_engine(engine_name: str, *, model_name: str) -> MutationProposer:
    """Factory used by the CLI to pick a mutator implementation."""
    if engine_name == "internal":
        return InternalMutator(model_name=model_name)
    if engine_name == "darwinian-evolver":
        return ExternalDarwinianEvolverMutator()
    raise ValueError(f"unknown mutation engine: {engine_name!r}")
