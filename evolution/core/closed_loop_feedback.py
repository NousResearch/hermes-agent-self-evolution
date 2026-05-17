"""ClosedLoopFeedbackCache — surface real-execution verdicts into GEPA reflection.

The LM-judge that scores GEPA candidates internally is a synthetic signal;
on hand-tuned baselines it can saturate near 1.0, leaving the reflection
LM with no gradient to mutate against. The closed-loop validator runs the
real agent on a task suite and produces a verdict the judge can't see.
This module wires that verdict into the reflection LM's input via the
existing ``Feedback`` string channel of GEPA's reflective dataset, without
touching the metric signature, the proposer, or the GEPA acceptance rule.
The verdict's effect is on what the LM *proposes*; it does not drive
GEPA's accept/reject choice — that remains ``sum(judge_scores)`` on the
minibatch.

The cache is the integration seam: it owns the validator, the task suite,
the saturation gate (so closed-loop only fires when the judge is at ceiling
or a periodic floor is reached), the candidate-keyed cache (so the same
candidate text doesn't pay validation cost twice), and a deterministic
feedback-block renderer (GEPA hashes feedback strings — non-determinism
would thrash its cache).
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import threading
from pathlib import Path
from typing import Callable, Literal, Optional

from evolution.validation.report import TaskResult, ValidationReport
from evolution.validation.task import TaskSuite
from evolution.validation.validator import (
    ChecksumDriftError,
    ClosedLoopValidator,
    ConcurrentRunError,
    StaleBackupError,
    ValidationInputs,
)


GateMode = Literal["sampled", "always"]

ArtifactWriter = Callable[[str, Path], None]
"""Write candidate text to a path in the format the installer consumes.

The cache calls this with ``(baseline_or_candidate_text, target_path)``
before each validator run. The default writes a single-tool MCP manifest
JSON (the tool-side shape); skill-side passes a writer that drops raw
text directly into the path.
"""

logger = logging.getLogger(__name__)


_NOISE_FLOOR = 0.15  # below this |Δpass_rate|, mark the block [CLOSED_LOOP-NOISY]
_INPUT_SCHEMA_STUB = {"type": "object", "properties": {}, "required": []}


class ClosedLoopFeedbackCache:
    """Run-bounded cache of closed-loop verdicts keyed by candidate text.

    One instance per ``evolve_tool`` / ``evolve_skill`` invocation. The
    tmp dir lives for the cache's lifetime; the OS reclaims it at process
    exit (no explicit cleanup).

    The cache is shared across metric calls within a run, including across
    DSPy's parallel ``Evaluate`` workers. The threading lock prevents
    workers from racing each other into ``ConcurrentRunError`` from the
    validator's cross-process flock.
    """

    def __init__(
        self,
        *,
        validator: ClosedLoopValidator,
        suite: TaskSuite,
        artifact_name: str,
        baseline_artifact_text: str,
        saturation_threshold: float = 0.95,
        min_iters: int = 3,
        window_size: int = 8,
        gate_mode: GateMode = "sampled",
        artifact_writer: Optional[ArtifactWriter] = None,
        artifact_suffix: str = ".json",
    ) -> None:
        if not (0.0 <= saturation_threshold <= 1.0):
            raise ValueError(
                f"saturation_threshold must be in [0, 1], got {saturation_threshold}"
            )
        if min_iters < 1:
            raise ValueError(f"min_iters must be >= 1, got {min_iters}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if gate_mode not in ("sampled", "always"):
            raise ValueError(
                f"gate_mode must be 'sampled' or 'always', got {gate_mode!r}"
            )
        self._validator = validator
        self._suite = suite
        self._artifact_name = artifact_name
        self.saturation_threshold = saturation_threshold
        self.min_iters = min_iters
        self.window_size = window_size
        self.gate_mode = gate_mode

        self._artifact_writer: ArtifactWriter = (
            artifact_writer
            if artifact_writer is not None
            else _make_default_tool_writer(artifact_name)
        )

        self._tmp_dir = Path(tempfile.mkdtemp(prefix="cl_feedback_"))
        self._baseline_path = self._tmp_dir / f"baseline{artifact_suffix}"
        self._evolved_path = self._tmp_dir / f"evolved{artifact_suffix}"
        self._artifact_writer(baseline_artifact_text, self._baseline_path)

        self._cache: dict[str, ValidationReport] = {}
        self._judge_history: list[float] = []
        self._iters_since_last_run = self.min_iters  # allow first fire
        self._lock = threading.Lock()

    def record_judge_score(self, score: float) -> None:
        """Track judge scores so the saturation gate can read recent history."""
        self._judge_history.append(score)
        self._iters_since_last_run += 1

    def should_run(self) -> bool:
        """Saturation gate: fire when the recent window saturates OR a periodic floor is hit.

        In ``gate_mode="always"`` the gate is unconditionally open — used when
        closed-loop is a selection-affecting score channel (behavioral-example
        trainset mode), where every novel candidate text must score every
        time it's sampled.

        Caller is responsible for cache-hit short-circuiting before this —
        a cache hit always returns the cached report regardless of the gate.
        """
        if self.gate_mode == "always":
            return True
        recent = self._judge_history[-self.window_size :]
        if not recent:
            return False
        saturated = min(recent) >= self.saturation_threshold
        periodic = self._iters_since_last_run >= self.min_iters
        return saturated or periodic

    def get_or_run(self, candidate_text: str) -> Optional[ValidationReport]:
        """Return a cached report; on cache miss, fire the validator if the gate is open.

        Returns ``None`` if the gate is closed (no fresh fire) or if the
        validator raised one of its expected errors. Closed-loop failure
        must never propagate up — GEPA runs without the feedback rather
        than aborting.
        """
        key = self._key(candidate_text)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            if not self.should_run():
                return None
            try:
                self._artifact_writer(candidate_text, self._evolved_path)
                inputs = ValidationInputs(
                    tool_name=self._artifact_name,
                    suite=self._suite,
                    baseline_artifact=self._baseline_path,
                    evolved_artifact=self._evolved_path,
                )
                report = self._validator.validate(inputs)
            except (ConcurrentRunError, StaleBackupError, ChecksumDriftError) as exc:
                logger.warning(
                    "closed-loop run skipped — %s: %s", type(exc).__name__, exc
                )
                return None
            self._cache[key] = report
            self._iters_since_last_run = 0
            return report

    def get_task_verdict(
        self, candidate_text: str, task_id: str
    ) -> Optional[TaskResult]:
        """Return the per-task ``TaskResult`` for ``candidate_text`` and ``task_id``.

        Used by the behavioral-example branch of the fitness metric:
        ``score = float(verdict.passed)`` if a verdict is available, else
        ``0.0`` so a candidate isn't credited for a non-result.

        Returns ``None`` if ``get_or_run`` returns ``None`` (gate closed in
        ``sampled`` mode, or validator raised a swallowed error) or if the
        requested ``task_id`` isn't present in the report's evolved phase.
        """
        report = self.get_or_run(candidate_text)
        if report is None:
            return None
        for task_result in report.evolved.tasks:
            if task_result.task_id == task_id:
                return task_result
        return None

    def _key(self, candidate_text: str) -> str:
        hasher = hashlib.sha256()
        hasher.update(candidate_text.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(self._suite.sha256.encode("utf-8"))
        return hasher.hexdigest()


def render_feedback_block(report: ValidationReport) -> str:
    """Render a deterministic ``[CLOSED_LOOP]`` block from a ValidationReport.

    The reflection LM reads this string verbatim. Determinism matters —
    GEPA hashes reflective-dataset entries for caching; a feedback string
    that varies run-to-run for the same report would cause cache thrash
    and unstable proposer behavior.

    Marks the block ``[CLOSED_LOOP-NOISY]`` when ``|Δpass_rate|`` is within
    the documented LM non-determinism floor — see
    ``ClosedLoopValidator``'s class docstring on that floor.
    """
    delta = report.delta
    is_noisy = abs(delta.pass_rate_change) < _NOISE_FLOOR
    header = "[CLOSED_LOOP-NOISY]" if is_noisy else "[CLOSED_LOOP]"

    lines: list[str] = [
        f"{header} decision={report.decision}",
    ]
    for reason in report.decision_reasons:
        lines.append(f"  reason: {reason}")
    lines.append(
        f"  win/loss/tie: {delta.n_wins} / {delta.n_losses} / {delta.n_ties}"
        f" (Δpass_rate {delta.pass_rate_change:+.2f})"
    )

    # Index baseline tasks by id so we can pair with evolved tasks.
    baseline_by_id = {t.task_id: t for t in report.baseline.tasks}
    suite_tasks_by_id = _load_suite_task_messages(report.task_suite_path)

    for ev in report.evolved.tasks:
        b = baseline_by_id.get(ev.task_id)
        if b is None:
            continue
        # Only surface tasks whose verdict changed — ties carry no signal.
        if b.passed == ev.passed and b.tool_calls_seq == ev.tool_calls_seq:
            continue
        verdict_change = _describe_verdict_change(b, ev)
        lines.append(f"  task {ev.task_id} ({verdict_change}):")
        msg = suite_tasks_by_id.get(ev.task_id)
        if msg is not None:
            lines.append(f"    user_message: {msg!r}")
        lines.append(f"    baseline_invoked: {list(b.tool_calls_seq)!r}")
        lines.append(f"    evolved_invoked:  {list(ev.tool_calls_seq)!r}")

    return "\n".join(lines)


def _describe_verdict_change(baseline_task, evolved_task) -> str:
    if baseline_task.passed and not evolved_task.passed:
        return "loss"
    if not baseline_task.passed and evolved_task.passed:
        return "win"
    return "tool_call_diff"


def _load_suite_task_messages(suite_path: str) -> dict[str, str]:
    """Load the suite's user_messages keyed by task_id.

    Empty dict if the file is unreadable — we degrade gracefully rather
    than failing the whole feedback render over a missing fixture.
    """
    path = Path(suite_path)
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = obj.get("task_id")
            msg = obj.get("user_message")
            if tid and isinstance(msg, str):
                result[tid] = msg
    except OSError:
        return {}
    return result


def _manifest_json(tool_name: str, description: str) -> str:
    """Single-tool MCP manifest JSON, the shape ToolManifest.from_json_file expects."""
    return json.dumps(
        {
            "tools": [
                {
                    "name": tool_name,
                    "description": description,
                    "inputSchema": _INPUT_SCHEMA_STUB,
                }
            ]
        },
        indent=2,
    )


def _make_default_tool_writer(tool_name: str) -> ArtifactWriter:
    """Default ``artifact_writer`` for tool-side closed-loop.

    Writes a single-tool MCP manifest JSON — the shape
    ``HermesToolDescriptionInstaller._extract_description`` consumes when
    ``artifact_source.suffix == ".json"``.
    """

    def write(candidate_text: str, path: Path) -> None:
        path.write_text(_manifest_json(tool_name, candidate_text))

    return write


def write_text_artifact(candidate_text: str, path: Path) -> None:
    """``artifact_writer`` for skill-side closed-loop: drop raw text into the path.

    The skill installer reads the whole file as the candidate SKILL.md
    body, so no envelope is needed.
    """
    path.write_text(candidate_text)
