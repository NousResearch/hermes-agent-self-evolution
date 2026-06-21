"""Verified, proposer-agnostic search for Phase 4 code evolution.

The mutation engine is deliberately outside this module.  It may be a local
model, Darwinian Evolver running as an external CLI, or a deterministic test
proposer.  The engine receives only :class:`PublicSearchContext`; sealed checks
stay in the verifier process and outside candidate workspaces.

The search loop combines four properties that are otherwise easy to lose in a
mutate-and-pytest implementation:

* patches are applied only in disposable repository copies;
* candidates with observationally equivalent visible behaviour are collapsed,
  keeping the smallest patch deterministically;
* a candidate is admitted only when sealed score improves over the current
  frozen tree and the full-suite gate passes; and
* accepted operator tags and public failure residues feed the next proposal
  round without exposing sealed check details or weakening thresholds.

This is process isolation, not a security sandbox for hostile code.  Untrusted
candidate execution still belongs in a container or VM.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Protocol, Sequence


_PATCH_HEADER = re.compile(r"^diff --git a/(.+) b/(.+)$", re.MULTILINE)


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_json(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return _digest_bytes(raw.encode("utf-8"))


def _safe_environment(extra: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Return a small environment without API keys or repository credentials."""

    keep = (
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "HOME",
        "USERPROFILE",
    )
    env = {key: os.environ[key] for key in keep if key in os.environ}
    env.update({"PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8"})
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return env


@dataclass(frozen=True)
class CommandCheck:
    """One command-based gate.

    ``score_pattern`` may capture a floating-point score from stdout.  Without
    it, a successful command scores 1 and a failed command scores 0.
    ``{repo}`` and ``{python}`` placeholders are expanded without invoking a
    shell.
    """

    name: str
    argv: tuple[str, ...]
    timeout: float = 60.0
    score_pattern: str = ""
    minimum: float = 1.0
    weight: float = 1.0

    def public_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "argv": list(self.argv),
            "timeout": self.timeout,
            "minimum": self.minimum,
        }


@dataclass(frozen=True)
class CheckOutcome:
    name: str
    passed: bool
    score: float
    returncode: int
    stdout_digest: str
    stderr_digest: str
    residue: str = ""
    timed_out: bool = False

    def signature_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "score": round(self.score, 8),
            "returncode": self.returncode,
            "stdout_digest": self.stdout_digest,
            "stderr_digest": self.stderr_digest,
            "timed_out": self.timed_out,
        }

    def manifest_dict(self) -> dict[str, object]:
        out = self.signature_dict()
        out["residue"] = self.residue
        return out


@dataclass(frozen=True)
class PatchCandidate:
    patch: str
    operator: str = "unclassified"
    rationale: str = ""
    parent_id: Optional[str] = None

    @property
    def candidate_id(self) -> str:
        return "patch-" + _digest_bytes(self.patch.encode("utf-8"))[:16]

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object], *, parent_id: Optional[str] = None
    ) -> "PatchCandidate":
        patch = str(value.get("patch", ""))
        if not patch.strip():
            raise ValueError("proposal is missing a unified diff in 'patch'")
        supplied_parent = value.get("parent_id", parent_id)
        return cls(
            patch=patch,
            operator=str(value.get("operator", "unclassified")),
            rationale=str(value.get("rationale", "")),
            parent_id=None if supplied_parent is None else str(supplied_parent),
        )


@dataclass(frozen=True)
class GateTask:
    task_id: str
    visible_checks: tuple[CommandCheck, ...]
    sealed_checks: tuple[CommandCheck, ...]
    full_suite_check: CommandCheck
    allowed_paths: tuple[str, ...]
    max_changed_files: int = 4
    max_changed_lines: int = 200
    min_sealed_delta: float = 0.0

    def __post_init__(self) -> None:
        if not self.visible_checks:
            raise ValueError("at least one visible check is required")
        if not self.sealed_checks:
            raise ValueError("at least one sealed check is required")
        if not self.allowed_paths:
            raise ValueError("at least one allowed path pattern is required")

    def public_dict(self) -> dict[str, object]:
        """Candidate-facing task view.  Sealed and full-suite commands omitted."""

        return {
            "task_id": self.task_id,
            "visible_checks": [check.public_dict() for check in self.visible_checks],
            "allowed_paths": list(self.allowed_paths),
            "max_changed_files": self.max_changed_files,
            "max_changed_lines": self.max_changed_lines,
        }


@dataclass(frozen=True)
class PublicSearchContext:
    task: Mapping[str, object]
    cycle: int
    budget: int
    seed: int
    parent_id: Optional[str]
    promoted_operators: tuple[str, ...]
    public_residues: Mapping[str, int]
    sealed_rejections: int

    def as_dict(self) -> dict[str, object]:
        return {
            "task": dict(self.task),
            "cycle": self.cycle,
            "budget": self.budget,
            "seed": self.seed,
            "parent_id": self.parent_id,
            "promoted_operators": list(self.promoted_operators),
            "public_residues": dict(self.public_residues),
            # Coarse count only: no hidden names, commands, outputs, or scores.
            "sealed_rejections": self.sealed_rejections,
        }


class Proposer(Protocol):
    def propose(
        self, context: PublicSearchContext, budget: int
    ) -> Sequence[PatchCandidate]: ...


class ExternalCLIProposer:
    """Call an external proposer over a JSON-lines protocol.

    The public context is written to stdin as one JSON object.  The command must
    print one proposal JSON object per line with ``patch``, and may include
    ``operator``, ``rationale``, and ``parent_id``.  No shell is used.
    """

    def __init__(
        self,
        argv: Sequence[str],
        *,
        timeout: float = 180.0,
        cwd: Optional[Path] = None,
    ) -> None:
        if not argv:
            raise ValueError("proposer command is empty")
        self.argv = tuple(str(item) for item in argv)
        self.timeout = timeout
        self.cwd = None if cwd is None else Path(cwd).resolve()

    def propose(
        self, context: PublicSearchContext, budget: int
    ) -> Sequence[PatchCandidate]:
        payload = json.dumps(context.as_dict(), sort_keys=True) + "\n"
        if self.cwd is None:
            with tempfile.TemporaryDirectory(prefix="phase4_proposer_") as tmp:
                return self._invoke(payload, budget, Path(tmp), context.parent_id)
        self.cwd.mkdir(parents=True, exist_ok=True)
        return self._invoke(payload, budget, self.cwd, context.parent_id)

    def _invoke(
        self,
        payload: str,
        budget: int,
        cwd: Path,
        parent_id: Optional[str],
    ) -> Sequence[PatchCandidate]:
        private_home = cwd / ".home"
        private_tmp = cwd / ".tmp"
        private_home.mkdir(exist_ok=True)
        private_tmp.mkdir(exist_ok=True)
        completed = subprocess.run(
            list(self.argv),
            cwd=cwd,
            input=payload,
            text=True,
            capture_output=True,
            timeout=self.timeout,
            check=False,
            env=_safe_environment(
                {
                    "HOME": str(private_home),
                    "USERPROFILE": str(private_home),
                    "TEMP": str(private_tmp),
                    "TMP": str(private_tmp),
                }
            ),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "proposer failed with exit "
                f"{completed.returncode}: {completed.stderr[-1000:]}"
            )
        values: list[Mapping[str, object]] = []
        stripped = completed.stdout.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            if not isinstance(parsed, list):
                raise ValueError("proposer JSON array expected")
            values.extend(item for item in parsed if isinstance(item, Mapping))
        else:
            for line in stripped.splitlines():
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError("proposal line must be a JSON object")
                values.append(value)
        return [
            PatchCandidate.from_dict(value, parent_id=parent_id)
            for value in values[:budget]
        ]


@dataclass
class SearchState:
    parent_id: Optional[str] = None
    promoted_operators: list[str] = field(default_factory=list)
    public_residues: Counter[str] = field(default_factory=Counter)
    sealed_rejections: int = 0
    rejected_ids: set[str] = field(default_factory=set)
    accepted_ids: list[str] = field(default_factory=list)

    def context(self, task: GateTask, cycle: int, budget: int, seed: int) -> PublicSearchContext:
        return PublicSearchContext(
            task=task.public_dict(),
            cycle=cycle,
            budget=budget,
            seed=seed,
            parent_id=self.parent_id,
            promoted_operators=tuple(self.promoted_operators),
            public_residues=dict(self.public_residues),
            sealed_rejections=self.sealed_rejections,
        )


@dataclass
class CandidateEvaluation:
    candidate: PatchCandidate
    cycle: int
    changed_files: tuple[str, ...] = ()
    changed_lines: int = 0
    visible: tuple[CheckOutcome, ...] = ()
    sealed: tuple[CheckOutcome, ...] = ()
    full_suite: Optional[CheckOutcome] = None
    visible_signature: str = ""
    tree_digest: str = ""
    source_unchanged: bool = True
    accepted: bool = False
    reason: str = "pending"
    duplicate_of: Optional[str] = None
    trace: list[str] = field(default_factory=list)

    @property
    def visible_passed(self) -> bool:
        return bool(self.visible) and all(item.passed for item in self.visible)

    @property
    def sealed_passed(self) -> bool:
        return bool(self.sealed) and all(item.passed for item in self.sealed)

    def manifest_dict(self) -> dict[str, object]:
        return {
            "event": "candidate",
            "cycle": self.cycle,
            "candidate_id": self.candidate.candidate_id,
            "parent_id": self.candidate.parent_id,
            "operator": self.candidate.operator,
            "rationale": self.candidate.rationale,
            "patch_digest": _digest_bytes(self.candidate.patch.encode("utf-8")),
            "changed_files": list(self.changed_files),
            "changed_lines": self.changed_lines,
            "visible": [item.manifest_dict() for item in self.visible],
            "sealed": [item.manifest_dict() for item in self.sealed],
            "full_suite": None
            if self.full_suite is None
            else self.full_suite.manifest_dict(),
            "visible_signature": self.visible_signature,
            "tree_digest": self.tree_digest,
            "source_unchanged": self.source_unchanged,
            "accepted": self.accepted,
            "reason": self.reason,
            "duplicate_of": self.duplicate_of,
            "trace": list(self.trace),
            "trace_digest": _digest_json(self.trace),
        }


@dataclass(frozen=True)
class SearchReport:
    task_id: str
    adaptive: bool
    proposal_budget: int
    proposals_evaluated: int
    accepted_ids: tuple[str, ...]
    initial_sealed_score: float
    final_sealed_score: float
    hidden_gain: float
    promoted_operators: tuple[str, ...]
    public_residues: Mapping[str, int]
    sealed_rejections: int
    replay_verified: bool
    manifest_path: str

    def as_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "adaptive": self.adaptive,
            "proposal_budget": self.proposal_budget,
            "proposals_evaluated": self.proposals_evaluated,
            "accepted_ids": list(self.accepted_ids),
            "initial_sealed_score": self.initial_sealed_score,
            "final_sealed_score": self.final_sealed_score,
            "hidden_gain": self.hidden_gain,
            "promoted_operators": list(self.promoted_operators),
            "public_residues": dict(self.public_residues),
            "sealed_rejections": self.sealed_rejections,
            "replay_verified": self.replay_verified,
            "manifest_path": self.manifest_path,
        }


class AdmissionManifest:
    """Append-only JSONL decisions plus content-addressed patch files."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.patch_dir = self.run_dir / "patches"
        self.path = self.run_dir / "admission_manifest.jsonl"
        self.patch_dir.mkdir(parents=True, exist_ok=True)

    def save_patch(self, candidate: PatchCandidate) -> Path:
        path = self.patch_dir / f"{candidate.candidate_id}.patch"
        if path.exists() and path.read_text(encoding="utf-8") != candidate.patch:
            raise ValueError("candidate id collision")
        path.write_text(candidate.patch, encoding="utf-8")
        return path

    def append(self, value: Mapping[str, object]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")


@dataclass(frozen=True)
class _PatchStats:
    files: tuple[str, ...]
    changed_lines: int


class _Workspace:
    def __init__(self, source: Path, task: GateTask, temp_parent: Path) -> None:
        self.source = Path(source).resolve()
        self.task = task
        self.temp_parent = temp_parent
        self._temp: Optional[tempfile.TemporaryDirectory[str]] = None
        self.root: Optional[Path] = None

    def __enter__(self) -> "_Workspace":
        self.temp_parent.mkdir(parents=True, exist_ok=True)
        self._temp = tempfile.TemporaryDirectory(
            prefix="phase4_candidate_", dir=str(self.temp_parent)
        )
        self.root = Path(self._temp.name) / "repo"
        shutil.copytree(
            self.source,
            self.root,
            ignore=shutil.ignore_patterns(
                ".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"
            ),
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._temp is not None:
            self._temp.cleanup()

    def apply(self, patch: str) -> _PatchStats:
        if self.root is None:
            raise RuntimeError("workspace is not open")
        stats = _validate_patch(patch, self.task)
        checked = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", "-"],
            cwd=self.root,
            input=patch,
            text=True,
            capture_output=True,
            check=False,
            env=_safe_environment(),
        )
        if checked.returncode != 0:
            raise ValueError(f"patch does not apply: {checked.stderr[-500:]}")
        applied = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=self.root,
            input=patch,
            text=True,
            capture_output=True,
            check=False,
            env=_safe_environment(),
        )
        if applied.returncode != 0:
            raise ValueError(f"patch apply failed: {applied.stderr[-500:]}")
        return stats


def _validate_patch(patch: str, task: GateTask) -> _PatchStats:
    if "new file mode 120000" in patch or "new mode 120000" in patch:
        raise ValueError("symbolic-link patches are not supported")
    pairs = _PATCH_HEADER.findall(patch)
    if not pairs:
        raise ValueError("proposal is not a git unified diff")
    files: set[str] = set()
    for before, after in pairs:
        if before != after:
            raise ValueError("renames are not supported by the Phase 4 MVP")
        rel = Path(after)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"patch path escapes repository: {after}")
        normalized = rel.as_posix()
        if not any(fnmatch.fnmatch(normalized, pattern) for pattern in task.allowed_paths):
            raise ValueError(f"patch path is outside allowed targets: {normalized}")
        files.add(normalized)
    body_paths = re.findall(r"^(?:--- a/|\+\+\+ b/)(.+)$", patch, re.MULTILINE)
    if not body_paths or any(Path(path).as_posix() not in files for path in body_paths):
        raise ValueError("patch body paths do not match allowed diff headers")
    changed_lines = sum(
        1
        for line in patch.splitlines()
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )
    if len(files) > task.max_changed_files:
        raise ValueError("patch changes too many files")
    if changed_lines > task.max_changed_lines:
        raise ValueError("patch changes too many lines")
    return _PatchStats(tuple(sorted(files)), changed_lines)


def _tree_digest(root: Path) -> str:
    entries: list[tuple[str, str]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel = path.relative_to(root)
        if any(
            part
            in {
                ".git",
                ".venv",
                ".phase4_home",
                ".phase4_tmp",
                "__pycache__",
                ".pytest_cache",
                ".mypy_cache",
            }
            for part in rel.parts
        ):
            continue
        entries.append((rel.as_posix(), _digest_bytes(path.read_bytes())))
    return _digest_json(entries)


def _run_check(root: Path, check: CommandCheck) -> CheckOutcome:
    argv = [
        part.format(repo=str(root), python=sys.executable) for part in check.argv
    ]
    private_home = root / ".phase4_home"
    private_tmp = root / ".phase4_tmp"
    private_home.mkdir(exist_ok=True)
    private_tmp.mkdir(exist_ok=True)
    try:
        completed = subprocess.run(
            argv,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=check.timeout,
            check=False,
            env=_safe_environment(
                {
                    "PHASE4_CANDIDATE_REPO": str(root),
                    "HOME": str(private_home),
                    "USERPROFILE": str(private_home),
                    "TEMP": str(private_tmp),
                    "TMP": str(private_tmp),
                }
            ),
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        score = 1.0 if completed.returncode == 0 else 0.0
        residue = "" if completed.returncode == 0 else f"{check.name}:exit-{completed.returncode}"
        if check.score_pattern:
            match = re.search(check.score_pattern, stdout)
            if match is None:
                score = 0.0
                residue = f"{check.name}:score-missing"
            else:
                score = float(match.group(1))
                if score < check.minimum:
                    residue = f"{check.name}:score-below-minimum"
        passed = completed.returncode == 0 and score >= check.minimum
        return CheckOutcome(
            name=check.name,
            passed=passed,
            score=score,
            returncode=completed.returncode,
            stdout_digest=_digest_bytes(stdout.encode("utf-8")),
            stderr_digest=_digest_bytes(stderr.encode("utf-8")),
            residue=residue,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return CheckOutcome(
            name=check.name,
            passed=False,
            score=0.0,
            returncode=-1,
            stdout_digest=_digest_bytes(stdout.encode("utf-8")),
            stderr_digest=_digest_bytes(stderr.encode("utf-8")),
            residue=f"{check.name}:timeout",
            timed_out=True,
        )
    except OSError as exc:
        message = f"{type(exc).__name__}:{exc}"
        return CheckOutcome(
            name=check.name,
            passed=False,
            score=0.0,
            returncode=-2,
            stdout_digest=_digest_bytes(b""),
            stderr_digest=_digest_bytes(message.encode("utf-8")),
            residue=f"{check.name}:command-error",
        )


def _weighted_score(
    checks: Sequence[CommandCheck], outcomes: Sequence[CheckOutcome]
) -> float:
    total = sum(check.weight for check in checks)
    if total <= 0:
        return 0.0
    return sum(
        check.weight * outcome.score for check, outcome in zip(checks, outcomes)
    ) / total


class VerifiedPatchSearch:
    """Iterative patch search with sealed admission and persistent evidence."""

    def __init__(
        self,
        repository: Path,
        task: GateTask,
        run_dir: Path,
        *,
        seed: int = 20260621,
    ) -> None:
        self.repository = Path(repository).resolve()
        if not self.repository.is_dir():
            raise FileNotFoundError(self.repository)
        self.task = task
        self.seed = seed
        resolved_run_dir = Path(run_dir).resolve()
        try:
            resolved_run_dir.relative_to(self.repository)
        except ValueError:
            pass
        else:
            raise ValueError("run_dir must be outside the stable repository")
        self.manifest = AdmissionManifest(resolved_run_dir)
        self.temp_parent = self.manifest.run_dir / "workspaces"
        self.state = SearchState()
        self._accepted_patches: list[PatchCandidate] = []
        self._accepted_digests: dict[str, str] = {}

    def _source_digest(self) -> str:
        return _tree_digest(self.repository)

    def _evaluate(
        self,
        candidate: PatchCandidate,
        cycle: int,
        *,
        include_sealed: bool,
    ) -> CandidateEvaluation:
        before_source = self._source_digest()
        evaluation = CandidateEvaluation(candidate=candidate, cycle=cycle)
        try:
            with _Workspace(self.repository, self.task, self.temp_parent) as workspace:
                evaluation.trace.append("copy_stable_tree")
                assert workspace.root is not None
                for accepted in self._accepted_patches:
                    workspace.apply(accepted.patch)
                    evaluation.trace.append(f"apply_parent:{accepted.candidate_id}")
                stats = workspace.apply(candidate.patch)
                evaluation.trace.append(f"apply_candidate:{candidate.candidate_id}")
                evaluation.changed_files = stats.files
                evaluation.changed_lines = stats.changed_lines
                evaluation.visible = tuple(
                    _run_check(workspace.root, check)
                    for check in self.task.visible_checks
                )
                evaluation.trace.append("run_visible_checks")
                evaluation.visible_signature = _digest_json(
                    [item.signature_dict() for item in evaluation.visible]
                )
                if include_sealed and evaluation.visible_passed:
                    evaluation.sealed = tuple(
                        _run_check(workspace.root, check)
                        for check in self.task.sealed_checks
                    )
                    evaluation.trace.append("run_sealed_checks")
                    evaluation.full_suite = _run_check(
                        workspace.root, self.task.full_suite_check
                    )
                    evaluation.trace.append("run_full_suite")
                evaluation.tree_digest = _tree_digest(workspace.root)
        except Exception as exc:
            evaluation.reason = f"patch_rejected:{type(exc).__name__}:{exc}"
            evaluation.trace.append("reject_and_discard")
        else:
            evaluation.trace.append("discard_workspace")
        evaluation.source_unchanged = before_source == self._source_digest()
        if not evaluation.source_unchanged:
            evaluation.reason = "stable_repository_mutated"
        return evaluation

    def _baseline(self) -> tuple[float, str]:
        before_source = self._source_digest()
        with _Workspace(self.repository, self.task, self.temp_parent) as workspace:
            assert workspace.root is not None
            for accepted in self._accepted_patches:
                workspace.apply(accepted.patch)
            sealed = tuple(
                _run_check(workspace.root, check) for check in self.task.sealed_checks
            )
            digest = _tree_digest(workspace.root)
        if before_source != self._source_digest():
            raise RuntimeError("baseline evaluation mutated the stable repository")
        return _weighted_score(self.task.sealed_checks, sealed), digest

    def _sealed_evaluate(
        self, candidate: PatchCandidate, cycle: int, baseline_score: float
    ) -> CandidateEvaluation:
        evaluation = self._evaluate(candidate, cycle, include_sealed=True)
        if evaluation.reason != "pending":
            return evaluation
        if not evaluation.visible_passed:
            evaluation.reason = "visible_gate_failed"
            return evaluation
        if not evaluation.sealed_passed:
            evaluation.reason = "sealed_gate_failed"
            return evaluation
        if evaluation.full_suite is None or not evaluation.full_suite.passed:
            evaluation.reason = "full_suite_failed"
            return evaluation
        score = _weighted_score(self.task.sealed_checks, evaluation.sealed)
        if score <= baseline_score + self.task.min_sealed_delta:
            evaluation.reason = "no_sealed_improvement"
            return evaluation
        evaluation.accepted = True
        evaluation.reason = "admissible"
        return evaluation

    def run(
        self,
        proposer: Proposer,
        *,
        cycles: int = 2,
        budget: int = 4,
        adaptive: bool = True,
    ) -> SearchReport:
        if cycles < 1 or budget < 1:
            raise ValueError("cycles and budget must be positive")
        initial_score, _ = self._baseline()
        current_score = initial_score
        proposals_evaluated = 0

        self.manifest.append(
            {
                "event": "run_start",
                "task": self.task.public_dict(),
                "adaptive": adaptive,
                "cycles": cycles,
                "budget": budget,
                "seed": self.seed,
                "sealed_check_count": len(self.task.sealed_checks),
                "full_suite_configured": True,
                "initial_sealed_score": initial_score,
            }
        )

        for cycle in range(1, cycles + 1):
            context = self.state.context(
                self.task, cycle, budget, self.seed + cycle - 1
            )
            proposed = list(proposer.propose(context, budget))[:budget]
            unique: dict[str, PatchCandidate] = {}
            for candidate in proposed:
                unique.setdefault(candidate.candidate_id, candidate)
            candidates = list(unique.values())
            proposals_evaluated += len(candidates)
            visible_evaluations: list[CandidateEvaluation] = []

            for candidate in candidates:
                self.manifest.save_patch(candidate)
                if candidate.candidate_id in self.state.rejected_ids:
                    cached = CandidateEvaluation(
                        candidate=candidate,
                        cycle=cycle,
                        reason="rejected_cached",
                    )
                    self.manifest.append(cached.manifest_dict())
                    continue
                if candidate.parent_id not in (None, self.state.parent_id):
                    stale = CandidateEvaluation(
                        candidate=candidate,
                        cycle=cycle,
                        reason="stale_parent",
                    )
                    self.state.rejected_ids.add(candidate.candidate_id)
                    self.manifest.append(stale.manifest_dict())
                    continue
                evaluation = self._evaluate(candidate, cycle, include_sealed=False)
                if evaluation.reason == "pending" and not evaluation.visible_passed:
                    evaluation.reason = "visible_gate_failed"
                if evaluation.reason != "pending":
                    for outcome in evaluation.visible:
                        if outcome.residue:
                            self.state.public_residues[outcome.residue] += 1
                    self.state.rejected_ids.add(candidate.candidate_id)
                    self.manifest.append(evaluation.manifest_dict())
                    continue
                visible_evaluations.append(evaluation)

            groups: dict[str, list[CandidateEvaluation]] = {}
            for evaluation in visible_evaluations:
                groups.setdefault(evaluation.visible_signature, []).append(evaluation)

            representatives: list[PatchCandidate] = []
            for group in groups.values():
                group.sort(
                    key=lambda item: (
                        item.changed_lines,
                        len(item.candidate.patch),
                        item.candidate.candidate_id,
                    )
                )
                winner = group[0]
                representatives.append(winner.candidate)
                for duplicate in group[1:]:
                    duplicate.reason = "visible_behavior_duplicate"
                    duplicate.duplicate_of = winner.candidate.candidate_id
                    self.state.rejected_ids.add(duplicate.candidate.candidate_id)
                    self.manifest.append(duplicate.manifest_dict())

            sealed_evaluations = [
                self._sealed_evaluate(candidate, cycle, current_score)
                for candidate in representatives
            ]
            admissible = [item for item in sealed_evaluations if item.accepted]
            admissible.sort(
                key=lambda item: (
                    -_weighted_score(self.task.sealed_checks, item.sealed),
                    item.changed_lines,
                    item.candidate.candidate_id,
                )
            )
            selected = admissible[0] if admissible else None

            for evaluation in sealed_evaluations:
                if evaluation is selected and adaptive:
                    evaluation.reason = "accepted"
                    self._accepted_patches.append(evaluation.candidate)
                    self._accepted_digests[evaluation.candidate.candidate_id] = (
                        evaluation.tree_digest
                    )
                    self.state.parent_id = evaluation.candidate.candidate_id
                    self.state.accepted_ids.append(evaluation.candidate.candidate_id)
                    if evaluation.candidate.operator not in self.state.promoted_operators:
                        self.state.promoted_operators.append(evaluation.candidate.operator)
                    current_score = _weighted_score(
                        self.task.sealed_checks, evaluation.sealed
                    )
                elif evaluation is selected:
                    evaluation.accepted = False
                    evaluation.reason = "frozen_control_no_adoption"
                elif evaluation.accepted:
                    evaluation.accepted = False
                    evaluation.reason = "dominated_admissible_candidate"
                    self.state.rejected_ids.add(evaluation.candidate.candidate_id)
                else:
                    self.state.rejected_ids.add(evaluation.candidate.candidate_id)
                    if evaluation.reason in {"sealed_gate_failed", "no_sealed_improvement"}:
                        self.state.sealed_rejections += 1
                self.manifest.append(evaluation.manifest_dict())

        replay_verified = self._verify_replay()
        report = SearchReport(
            task_id=self.task.task_id,
            adaptive=adaptive,
            proposal_budget=cycles * budget,
            proposals_evaluated=proposals_evaluated,
            accepted_ids=tuple(self.state.accepted_ids),
            initial_sealed_score=round(initial_score, 8),
            final_sealed_score=round(current_score, 8),
            hidden_gain=round(current_score - initial_score, 8),
            promoted_operators=tuple(self.state.promoted_operators),
            public_residues=dict(self.state.public_residues),
            sealed_rejections=self.state.sealed_rejections,
            replay_verified=replay_verified,
            manifest_path=str(self.manifest.path),
        )
        self.manifest.append({"event": "run_end", **report.as_dict()})
        return report

    def _verify_replay(self) -> bool:
        if not self._accepted_patches:
            return True
        before_source = self._source_digest()
        with _Workspace(self.repository, self.task, self.temp_parent) as workspace:
            assert workspace.root is not None
            for candidate in self._accepted_patches:
                workspace.apply(candidate.patch)
                expected = self._accepted_digests[candidate.candidate_id]
                if _tree_digest(workspace.root) != expected:
                    return False
            visible = tuple(
                _run_check(workspace.root, check) for check in self.task.visible_checks
            )
            sealed = tuple(
                _run_check(workspace.root, check) for check in self.task.sealed_checks
            )
            full = _run_check(workspace.root, self.task.full_suite_check)
        return (
            before_source == self._source_digest()
            and all(item.passed for item in visible)
            and all(item.passed for item in sealed)
            and full.passed
        )


@dataclass(frozen=True)
class CounterfactualReport:
    adaptive: SearchReport
    frozen: SearchReport
    equal_budget: bool
    recursive_gain: float

    def as_dict(self) -> dict[str, object]:
        return {
            "adaptive": self.adaptive.as_dict(),
            "frozen": self.frozen.as_dict(),
            "equal_budget": self.equal_budget,
            "recursive_gain": self.recursive_gain,
        }


def compare_adaptive_frozen(
    repository: Path,
    task: GateTask,
    run_dir: Path,
    proposer_factory: Callable[[], Proposer],
    *,
    cycles: int = 2,
    budget: int = 4,
    seed: int = 20260621,
) -> CounterfactualReport:
    """Run adaptive and frozen arms with identical cycle/proposal budgets."""

    root = Path(run_dir)
    adaptive = VerifiedPatchSearch(
        repository, task, root / "adaptive", seed=seed
    ).run(proposer_factory(), cycles=cycles, budget=budget, adaptive=True)
    frozen = VerifiedPatchSearch(
        repository, task, root / "frozen", seed=seed
    ).run(proposer_factory(), cycles=cycles, budget=budget, adaptive=False)
    return CounterfactualReport(
        adaptive=adaptive,
        frozen=frozen,
        equal_budget=adaptive.proposal_budget == frozen.proposal_budget,
        recursive_gain=round(
            adaptive.final_sealed_score - frozen.final_sealed_score, 8
        ),
    )
