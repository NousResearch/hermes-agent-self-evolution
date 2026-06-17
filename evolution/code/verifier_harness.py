"""Verifier gate for Phase 4 code-evolution candidates.

This module is a small, repo-native adaptation of the verifier pattern used in
``sunghunkwag/rsi-metaforge-core``: candidate code sees public files and visible
checks, while the harness keeps hidden checks in memory and accepts a candidate
only when it beats a frozen baseline without failing the regression gate.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence


CheckFn = Callable[[Path], "CheckResult"]
CandidateRunner = Callable[["ToolActionSession"], None]


@dataclass(frozen=True)
class CheckResult:
    """Result from a visible, hidden, or regression-suite check."""

    passed: bool
    score: float = 0.0
    detail: str = ""
    residues: tuple[str, ...] = ()

    @classmethod
    def pass_(cls, detail: str = "passed", score: float = 1.0) -> "CheckResult":
        return cls(True, score, detail, ())

    @classmethod
    def fail(
        cls,
        detail: str,
        residues: Sequence[str] = (),
        score: float = 0.0,
    ) -> "CheckResult":
        return cls(False, score, detail, tuple(residues))


@dataclass(frozen=True)
class ActionTrace:
    """Deterministic audit record for one tool-layer action."""

    action: str
    input_hash: str
    output_hash: str
    status: str
    failure_type: str = ""
    cost: float = 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "status": self.status,
            "failure_type": self.failure_type,
            "cost": self.cost,
        }


@dataclass(frozen=True)
class CodeEvalTask:
    """A file-backed code repair task with visible and sealed hidden checks.

    ``workspace_files`` are the only files provisioned for the candidate. The
    hidden check remains a harness-side callable and is never written into the
    workspace.
    """

    name: str
    workspace_files: Mapping[str, str | bytes]
    visible_check: CheckFn
    hidden_check: CheckFn
    full_suite_check: Optional[CheckFn] = None
    allowed_tools: tuple[str, ...] = (
        "read_file",
        "write_file",
        "apply_patch",
        "revert_file",
        "run_visible_check",
    )

    def provision(self, root: Path) -> Path:
        workspace = Path(root) / self.name
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        for rel_path, payload in sorted(self.workspace_files.items()):
            target = _safe_child(workspace, rel_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(payload, bytes):
                target.write_bytes(payload)
            else:
                target.write_text(payload, encoding="utf-8")
        return workspace


@dataclass(frozen=True)
class RunResult:
    """Complete result for one arm: frozen or adaptive."""

    arm: str
    visible: CheckResult
    hidden: CheckResult
    full_suite: CheckResult
    traces: tuple[ActionTrace, ...] = ()
    trace_digest: str = ""
    skipped: bool = False

    @classmethod
    def skipped_result(cls, arm: str, reason: str) -> "RunResult":
        skipped = CheckResult.fail(reason, ("skipped",))
        return cls(arm, skipped, skipped, skipped, (), "", True)


@dataclass(frozen=True)
class GateDecision:
    """Final admission decision for a candidate against one task."""

    accepted: bool
    reason: str
    adaptive: RunResult
    frozen: RunResult
    candidate_key: str
    cached: bool = False


class ToolActionSession:
    """Restricted file-operation surface exposed to candidate runners."""

    def __init__(
        self,
        root: Path,
        visible_check: CheckFn,
        allowed_tools: Sequence[str],
    ) -> None:
        self.root = Path(root).resolve()
        self.visible_check = visible_check
        self.allowed_tools = tuple(allowed_tools)
        self.traces: list[ActionTrace] = []

    def read_file(self, rel_path: str) -> str:
        action = "read_file"
        payload = {"path": rel_path}
        before = self._tree_digest()
        self._ensure_tool(action, payload, before)
        path = self._resolve(action, payload, before, rel_path)
        try:
            out = path.read_text(encoding="utf-8")
        except Exception:
            self._reject(action, payload, before, "read_failed")
            raise
        self._record(action, payload, before, self._tree_digest(), "observed")
        return out

    def write_file(self, rel_path: str, content: str) -> None:
        action = "write_file"
        payload = {
            "path": rel_path,
            "content_hash": _digest_bytes(content.encode("utf-8")),
        }
        before = self._tree_digest()
        self._ensure_tool(action, payload, before)
        path = self._resolve(action, payload, before, rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self._record(action, payload, before, self._tree_digest(), "committed")

    def apply_patch(self, rel_path: str, old: str, new: str) -> bool:
        action = "apply_patch"
        payload = {
            "path": rel_path,
            "old_hash": _digest_bytes(old.encode("utf-8")),
            "new_hash": _digest_bytes(new.encode("utf-8")),
        }
        before = self._tree_digest()
        self._ensure_tool(action, payload, before)
        path = self._resolve(action, payload, before, rel_path)
        text = path.read_text(encoding="utf-8")
        if text.count(old) != 1:
            self._record(
                action,
                payload,
                before,
                before,
                "reverted",
                failure_type="patch_not_unique",
            )
            return False
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        self._record(action, payload, before, self._tree_digest(), "committed")
        return True

    def revert_file(self, rel_path: str, content: str) -> None:
        action = "revert_file"
        payload = {
            "path": rel_path,
            "content_hash": _digest_bytes(content.encode("utf-8")),
        }
        before = self._tree_digest()
        self._ensure_tool(action, payload, before)
        path = self._resolve(action, payload, before, rel_path)
        path.write_text(content, encoding="utf-8")
        self._record(action, payload, before, self._tree_digest(), "reverted")

    def run_visible_check(self) -> CheckResult:
        action = "run_visible_check"
        payload = {"checker": getattr(self.visible_check, "__name__", "visible_check")}
        before = self._tree_digest()
        self._ensure_tool(action, payload, before)
        try:
            result = self.visible_check(self.root)
        except Exception as exc:
            result = CheckResult.fail(str(exc), ("visible_check_exception",))
        self._record(
            action,
            payload,
            before,
            self._tree_digest(),
            "observed",
            failure_type="" if result.passed else "visible_check_failed",
        )
        return result

    def _tree_digest(self) -> str:
        entries: list[tuple[str, str]] = []
        for path in sorted(p for p in self.root.rglob("*") if p.is_file()):
            rel = path.relative_to(self.root).as_posix()
            entries.append((rel, _digest_bytes(path.read_bytes())))
        return _digest_json(entries)

    def _record(
        self,
        action: str,
        payload: object,
        before: str,
        after: str,
        status: str,
        failure_type: str = "",
    ) -> None:
        self.traces.append(
            ActionTrace(
                action=action,
                input_hash=_digest_json(
                    {"action": action, "payload": payload, "before": before}
                ),
                output_hash=_digest_json(
                    {"after": after, "failure_type": failure_type}
                ),
                status=status,
                failure_type=failure_type,
            )
        )

    def _reject(
        self,
        action: str,
        payload: object,
        before: str,
        failure_type: str,
    ) -> None:
        self._record(action, payload, before, before, "reverted", failure_type)
        raise ValueError(f"{action} rejected: {failure_type}")

    def _ensure_tool(self, action: str, payload: object, before: str) -> None:
        if action not in self.allowed_tools:
            self._reject(action, payload, before, "tool_not_allowed")

    def _resolve(
        self,
        action: str,
        payload: object,
        before: str,
        rel_path: str,
    ) -> Path:
        try:
            return _safe_child(self.root, rel_path)
        except ValueError:
            self._reject(action, payload, before, "path_escape")
            raise


class CodeFitnessHarness:
    """Admission gate for a Phase 4 candidate versus a frozen baseline."""

    def __init__(
        self,
        *,
        min_hidden_delta: float = 0.0,
        cache_rejections: bool = True,
    ) -> None:
        self.min_hidden_delta = min_hidden_delta
        self.cache_rejections = cache_rejections
        self.rejected_candidate_keys: set[str] = set()

    def evaluate(
        self,
        task: CodeEvalTask,
        *,
        adaptive_runner: CandidateRunner,
        frozen_runner: CandidateRunner,
        candidate_id: str,
        root: Optional[Path] = None,
    ) -> GateDecision:
        candidate_key = _digest_json({"task": task.name, "candidate": candidate_id})
        if candidate_key in self.rejected_candidate_keys:
            return GateDecision(
                accepted=False,
                reason="rejected_cached",
                adaptive=RunResult.skipped_result("adaptive", "rejected_cached"),
                frozen=RunResult.skipped_result("frozen", "rejected_cached"),
                candidate_key=candidate_key,
                cached=True,
            )

        if root is None:
            with tempfile.TemporaryDirectory(prefix="phase4_harness_") as tmp:
                return self._evaluate_in_root(
                    task, adaptive_runner, frozen_runner, candidate_id, Path(tmp)
                )
        return self._evaluate_in_root(
            task, adaptive_runner, frozen_runner, candidate_id, Path(root)
        )

    def _evaluate_in_root(
        self,
        task: CodeEvalTask,
        adaptive_runner: CandidateRunner,
        frozen_runner: CandidateRunner,
        candidate_id: str,
        root: Path,
    ) -> GateDecision:
        candidate_key = _digest_json({"task": task.name, "candidate": candidate_id})
        frozen = _run_arm(task, frozen_runner, root / "frozen", "frozen")
        adaptive = _run_arm(task, adaptive_runner, root / "adaptive", "adaptive")

        accepted, reason = self._decide(frozen, adaptive)
        if not accepted and self.cache_rejections:
            self.rejected_candidate_keys.add(candidate_key)
        return GateDecision(
            accepted=accepted,
            reason=reason,
            adaptive=adaptive,
            frozen=frozen,
            candidate_key=candidate_key,
        )

    def _decide(self, frozen: RunResult, adaptive: RunResult) -> tuple[bool, str]:
        if not adaptive.visible.passed:
            return False, "visible_gate_failed"
        if not adaptive.hidden.passed:
            return False, "hidden_gate_failed"
        if not adaptive.full_suite.passed:
            return False, "full_suite_failed"
        required = frozen.hidden.score + self.min_hidden_delta
        if adaptive.hidden.score <= required:
            return False, "no_hidden_improvement"
        return True, "accepted"


def _run_arm(
    task: CodeEvalTask,
    runner: CandidateRunner,
    root: Path,
    arm: str,
) -> RunResult:
    workspace = task.provision(root)
    session = ToolActionSession(workspace, task.visible_check, task.allowed_tools)
    runner_error: Optional[Exception] = None
    try:
        runner(session)
    except Exception as exc:
        runner_error = exc

    visible = _safe_check(task.visible_check, workspace, "visible_check_exception")
    hidden = _safe_check(task.hidden_check, workspace, "hidden_check_exception")
    full_suite = (
        _safe_check(task.full_suite_check, workspace, "full_suite_exception")
        if task.full_suite_check is not None
        else CheckResult.pass_("full suite not configured")
    )
    if runner_error is not None:
        visible = CheckResult.fail(str(runner_error), ("runner_exception",))

    traces = tuple(session.traces)
    return RunResult(
        arm=arm,
        visible=visible,
        hidden=hidden,
        full_suite=full_suite,
        traces=traces,
        trace_digest=_digest_json([trace.as_dict() for trace in traces]),
    )


def _safe_check(
    check: CheckFn,
    workspace: Path,
    residue: str,
) -> CheckResult:
    try:
        return check(workspace)
    except Exception as exc:
        return CheckResult.fail(str(exc), (residue,))


def _safe_child(root: Path, rel_path: str) -> Path:
    rel = Path(str(rel_path))
    if rel.is_absolute():
        raise ValueError(f"absolute path rejected: {rel_path}")
    base = Path(root).resolve()
    target = (base / rel).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {rel_path}") from exc
    return target


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _digest_json(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return _digest_bytes(raw)
