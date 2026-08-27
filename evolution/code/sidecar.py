"""The boundary to the AGPL evolution sidecar.

This module is the entire contact surface between this MIT codebase and
``darwinian_evolver``, which is AGPL-3.0. Nothing here imports it, and nothing
here may ever import it.

Why a sidecar exists at all. The obvious plan was "call darwinian_evolver as an
external CLI", but the tool cannot do that: ``problems/registry.py`` holds a
hardcoded ``AVAILABLE_PROBLEMS`` dict and ``__main__.py`` restricts its
``problem`` argument to that dict's keys. There is no plugin path, no entry
point and no ``--problem-module``. Defining a Hermes problem means subclassing
its ``GitBasedOrganism``, ``Evaluator`` and ``Mutator`` — importing AGPL code
into the importing process.

So the AGPL-linked code lives in a separate, AGPL-licensed repository. The
dependency runs one way only:

    hermes-evolver-problems  (AGPL)  ──imports──▶  darwinian_evolver  (AGPL)
                             (AGPL)  ──imports──▶  this package       (MIT)
    this package             (MIT)   ──subprocess──▶ the sidecar

MIT flowing into AGPL is permitted; the reverse is what we are avoiding. The
sidecar may import our admission gate directly, which is why the gate lives in
:mod:`evolution.code.admission` rather than behind a socket.

Communication is a JSON job file in and a JSON result file out. Deliberately
boring: it means the sidecar can be reimplemented against a different engine
without this file changing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Bumped when the job/result contract changes incompatibly. The sidecar echoes
# it back so a version mismatch is an explicit error rather than a KeyError
# halfway through parsing someone else's format.
PROTOCOL_VERSION = 1

# Env var pointing at the sidecar executable or its checkout.
SIDECAR_ENV = "HERMES_EVOLVER_SIDECAR"

# Console script the sidecar package installs.
SIDECAR_COMMAND = "hermes-evolver-problems"

_INSTALL_HINT = (
    "Phase 4 needs the AGPL evolution sidecar, which is intentionally a "
    "separate package so that AGPL code never enters this MIT tree.\n\n"
    "  git clone https://github.com/numandev1/hermes-evolver-problems\n"
    "  pip install -e ./hermes-evolver-problems\n\n"
    f"Or set {SIDECAR_ENV} to the executable or checkout."
)


class SidecarNotAvailable(RuntimeError):
    """The sidecar is not installed or not runnable."""


class SidecarFailed(RuntimeError):
    """The sidecar ran but did not produce a usable result."""


@dataclass
class CodeCandidate:
    """One evolved variant handed back by the sidecar."""

    id: str
    files: dict[str, str]
    score: float = 0.0
    iteration: int = 0
    parent: Optional[str] = None
    diff: str = ""
    notes: str = ""

    def changed_paths(self, baseline: dict[str, str]) -> list[str]:
        return sorted(p for p, c in self.files.items() if baseline.get(p) != c)

    def total_chars(self) -> int:
        return sum(len(c) for c in self.files.values())


@dataclass
class SidecarJob:
    """What the sidecar is asked to do."""

    repo_root: str
    files: dict[str, str]
    task: str
    output_dir: str
    iterations: int = 5
    population: int = 4
    model: str = ""
    targeted_tests: list[str] = field(default_factory=list)
    protocol: int = PROTOCOL_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class SidecarResult:
    """What came back."""

    candidates: list[CodeCandidate]
    best_id: Optional[str] = None
    elapsed_s: float = 0.0
    log_path: Optional[str] = None
    raw: dict = field(default_factory=dict)

    def best(self) -> Optional[CodeCandidate]:
        if self.best_id:
            for candidate in self.candidates:
                if candidate.id == self.best_id:
                    return candidate
        return max(self.candidates, key=lambda c: c.score, default=None)


def find_sidecar(explicit: Optional[str] = None) -> list[str]:
    """Resolve the sidecar invocation, or explain how to get one.

    Returns the argv prefix to run. Accepts an executable, or a checkout
    directory which is run as a module.
    """
    for source in (explicit, os.getenv(SIDECAR_ENV)):
        if not source:
            continue
        path = Path(source).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return [str(path)]
        if path.is_dir():
            # A checkout: run the package from inside it.
            return ["python", "-m", "hermes_problems"]
        raise SidecarNotAvailable(
            f"{SIDECAR_ENV} points at {source}, which is neither an executable "
            f"nor a directory.\n\n{_INSTALL_HINT}"
        )

    found = shutil.which(SIDECAR_COMMAND)
    if found:
        return [found]

    # Importable without us importing it: ask the interpreter, in a subprocess.
    probe = subprocess.run(
        ["python", "-c", "import importlib.util,sys;"
         "sys.exit(0 if importlib.util.find_spec('hermes_problems') else 1)"],
        capture_output=True,
    )
    if probe.returncode == 0:
        return ["python", "-m", "hermes_problems"]

    raise SidecarNotAvailable(_INSTALL_HINT)


def sidecar_available(explicit: Optional[str] = None) -> tuple[bool, str]:
    """Non-raising probe, for CLIs that want to report rather than fail."""
    try:
        argv = find_sidecar(explicit)
    except SidecarNotAvailable as exc:
        return False, str(exc)
    return True, " ".join(argv)


def run_sidecar(
    job: SidecarJob,
    explicit: Optional[str] = None,
    timeout_s: int = 7200,
    stream_output: bool = True,
) -> SidecarResult:
    """Run one evolution job and parse what comes back.

    The job and result travel as files rather than pipes so a long run's output
    survives the process, and so a failed run leaves something to read.
    """
    argv = list(find_sidecar(explicit))

    output_dir = Path(job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    job_path = output_dir / "job.json"
    result_path = output_dir / "result.json"
    log_path = output_dir / "sidecar.log"

    job_path.write_text(job.to_json())

    argv += ["--job", str(job_path), "--result", str(result_path)]

    started = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(
                argv,
                stdout=log if not stream_output else None,
                stderr=subprocess.STDOUT if not stream_output else None,
                timeout=timeout_s,
                cwd=str(output_dir),
            )
    except subprocess.TimeoutExpired as exc:
        raise SidecarFailed(
            f"sidecar exceeded {timeout_s}s. Partial output: {log_path}"
        ) from exc
    except OSError as exc:
        raise SidecarNotAvailable(f"could not run {' '.join(argv)}: {exc}") from exc

    elapsed = time.time() - started

    if not result_path.is_file():
        raise SidecarFailed(
            f"sidecar exited {proc.returncode} without writing {result_path}. "
            f"See {log_path}."
        )

    return parse_result(result_path, elapsed=elapsed, log_path=log_path)


def parse_result(
    result_path: Path,
    elapsed: float = 0.0,
    log_path: Optional[Path] = None,
) -> SidecarResult:
    """Read and validate a sidecar result file."""
    try:
        payload = json.loads(Path(result_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SidecarFailed(f"unreadable sidecar result {result_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise SidecarFailed(f"sidecar result must be an object, got {type(payload).__name__}")

    protocol = payload.get("protocol")
    if protocol != PROTOCOL_VERSION:
        raise SidecarFailed(
            f"sidecar speaks protocol {protocol}, this build expects "
            f"{PROTOCOL_VERSION}. Update whichever is older."
        )

    if payload.get("error"):
        raise SidecarFailed(f"sidecar reported: {payload['error']}")

    candidates: list[CodeCandidate] = []
    for raw in payload.get("organisms", []):
        if not isinstance(raw, dict):
            continue
        files = raw.get("files")
        if not isinstance(files, dict) or not files:
            # A candidate with no files is not a candidate.
            continue
        candidates.append(
            CodeCandidate(
                id=str(raw.get("id") or f"organism-{len(candidates)}"),
                files={str(k): str(v) for k, v in files.items()},
                score=_as_float(raw.get("score")),
                iteration=int(raw.get("iteration") or 0),
                parent=raw.get("parent"),
                diff=str(raw.get("diff") or ""),
                notes=str(raw.get("notes") or ""),
            )
        )

    return SidecarResult(
        candidates=candidates,
        best_id=payload.get("best"),
        elapsed_s=elapsed,
        log_path=str(log_path) if log_path else None,
        raw=payload,
    )


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
