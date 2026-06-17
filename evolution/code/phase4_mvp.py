"""Phase 4 MVP: the fitness/verifier *layer* assembled on the verifier_harness primitives.

``verifier_harness.py`` provides the executable primitives. This module turns them
into the full set of guarantees advertised in issue #118, so that the issue's
claims and the code match one-to-one:

    issue #118 layer                     delivered by
    ----------------------------------   -------------------------------------------
    sealed hidden evaluations            verifier_harness (CodeEvalTask.hidden_check)
    adaptive-vs-frozen baseline          verifier_harness (CodeFitnessHarness._decide)
    patch/revert traces                  verifier_harness (ToolActionSession/ActionTrace)
    rejection caching                    verifier_harness (rejected_candidate_keys)
    full-suite gating before accept      StrictCodeFitnessHarness          (here)
    unseen-seed held-out evaluation      evaluate_on_unseen_seed           (here)
    residue-driven curriculum            ResidueCurriculum                 (here)
    deterministic lineage/archive        LineageArchive                    (here)
    deterministic replay of accepted     LineageArchive.verify_replay /
                                         LineageArchive.replay_evaluation  (here)
    issue/PR summary emission            summarize_decision / render_markdown (here)

GATE-PRESERVATION INVARIANT
    The curriculum only *reorders and generates tasks*. It holds no reference to,
    and never mutates, the acceptance thresholds in the harness. Failed attempts
    make the search harder-targeted (a more representative visible surface), never
    the gate softer. This is the same discipline as negative-grammar induction:
    reorder exploration, do not lower the bar.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from evolution.code.verifier_harness import (
    CodeEvalTask,
    CodeFitnessHarness,
    GateDecision,
    RunResult,
)

# A task factory takes (visible_seed, hidden_seed) and builds a held-out task whose
# visible checks key off visible_seed and whose sealed hidden checks key off hidden_seed.
TaskFactory = Callable[[int, int], CodeEvalTask]

# Maps a failure residue string to a follow-up task that specifically targets it.
ResidueTaskFactory = Callable[[str], Optional[CodeEvalTask]]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _sha_json(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return _sha(raw.encode("utf-8"))


# --------------------------------------------------------------------------- #
# 1. full-suite gating: refuse the free pass when no full suite is configured  #
# --------------------------------------------------------------------------- #
class StrictCodeFitnessHarness(CodeFitnessHarness):
    """CodeFitnessHarness that refuses to accept a candidate that was never run
    through a full suite.

    The base harness treats a missing ``full_suite_check`` as a pass
    (``CheckResult.pass_("full suite not configured")``). For a real Phase 4
    target the MVP requires "full test-suite pass after accepted patch", so an
    unconfigured suite must be a *non-acceptance*, not a silent free pass.
    """

    _NOT_CONFIGURED = "full suite not configured"

    def __init__(
        self,
        *,
        min_hidden_delta: float = 0.0,
        cache_rejections: bool = True,
        require_full_suite: bool = True,
    ) -> None:
        super().__init__(
            min_hidden_delta=min_hidden_delta, cache_rejections=cache_rejections
        )
        self.require_full_suite = require_full_suite

    def _decide(self, frozen: RunResult, adaptive: RunResult) -> tuple[bool, str]:
        if (
            self.require_full_suite
            and adaptive.full_suite.passed
            and adaptive.full_suite.detail == self._NOT_CONFIGURED
        ):
            return False, "full_suite_required"
        return super()._decide(frozen, adaptive)


# --------------------------------------------------------------------------- #
# 2. unseen-seed held-out evaluation                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HeldOutEvaluation:
    decision: GateDecision
    visible_seed: int
    hidden_seed: int


def evaluate_on_unseen_seed(
    harness: CodeFitnessHarness,
    factory: TaskFactory,
    *,
    adaptive_runner,
    frozen_runner,
    candidate_id: str,
    visible_seed: int,
    hidden_seed: int,
    root: Optional[Path] = None,
) -> HeldOutEvaluation:
    """Evaluate adaptive vs frozen where the hidden/full-suite seed is *unseen*.

    The candidate only ever observes the visible-seed checks during its run; the
    gate scores it on the hidden-seed checks it never had access to. We refuse to
    run with ``hidden_seed == visible_seed`` so the held-out property cannot be
    accidentally collapsed.
    """
    if visible_seed == hidden_seed:
        raise ValueError(
            "hidden_seed must differ from visible_seed; otherwise the evaluation "
            "is not held-out and the gate proves nothing about unseen behavior"
        )
    task = factory(visible_seed, hidden_seed)
    decision = harness.evaluate(
        task,
        adaptive_runner=adaptive_runner,
        frozen_runner=frozen_runner,
        candidate_id=candidate_id,
        root=root,
    )
    return HeldOutEvaluation(decision, visible_seed, hidden_seed)


# --------------------------------------------------------------------------- #
# 3. residue-driven curriculum (gate-preserving)                              #
# --------------------------------------------------------------------------- #
@dataclass
class _Pending:
    task: CodeEvalTask
    residue: str
    priority: int


class ResidueCurriculum:
    """Turns failure residues into a prioritized queue of follow-up tasks.

    Contract: this object never sees or changes acceptance thresholds. It only
    decides *which task to attempt next*. A residue is "covered" once it has
    produced a follow-up task, so the same failure is not rediscovered into the
    same task twice.
    """

    def __init__(
        self, residue_factories: Optional[Mapping[str, ResidueTaskFactory]] = None
    ) -> None:
        self._factories: dict[str, ResidueTaskFactory] = dict(residue_factories or {})
        self._base: list[CodeEvalTask] = []
        self._pending: dict[str, _Pending] = {}
        self._covered: set[str] = set()
        self._residue_counts: Counter = Counter()

    def register(self, residue: str, factory: ResidueTaskFactory) -> None:
        self._factories[residue] = factory

    def seed(self, tasks: Sequence[CodeEvalTask]) -> None:
        self._base.extend(tasks)

    def observe(self, decision: GateDecision) -> list[CodeEvalTask]:
        """Consume an evaluation result; schedule targeted follow-ups for any
        new failure residue. Returns the tasks newly scheduled (for logging)."""
        if decision.accepted:
            return []
        new_tasks: list[CodeEvalTask] = []
        for residue in self._collect_residues(decision):
            self._residue_counts[residue] += 1
            if residue in self._covered:
                continue
            factory = self._factories.get(residue)
            if factory is None:
                continue
            task = factory(residue)
            if task is None:
                continue
            self._covered.add(residue)
            self._pending[task.name] = _Pending(
                task, residue, self._residue_counts[residue]
            )
            new_tasks.append(task)
        return new_tasks

    def next_task(self) -> Optional[CodeEvalTask]:
        """Highest-priority pending follow-up (impact x frequency proxy), then the
        base queue. Deterministic tie-break by task name."""
        if self._pending:
            name = sorted(
                self._pending, key=lambda n: (-self._pending[n].priority, n)
            )[0]
            return self._pending.pop(name).task
        if self._base:
            return self._base.pop(0)
        return None

    def residue_counts(self) -> dict[str, int]:
        return dict(self._residue_counts)

    @staticmethod
    def _collect_residues(decision: GateDecision) -> list[str]:
        arm = decision.adaptive
        residues: list[str] = []
        for check in (arm.visible, arm.hidden, arm.full_suite):
            residues.extend(check.residues)
        for trace in arm.traces:
            if trace.failure_type:
                residues.append(trace.failure_type)
        # stable de-dup preserving first-seen order
        seen: set[str] = set()
        ordered: list[str] = []
        for r in residues:
            if r not in seen:
                seen.add(r)
                ordered.append(r)
        return ordered


# --------------------------------------------------------------------------- #
# 4. deterministic lineage/archive + replay                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LineageEntry:
    candidate_id: str
    parent_id: Optional[str]
    task_name: str
    accepted: bool
    reason: str
    frozen_hidden: float
    adaptive_hidden: float
    hidden_delta: float
    trace_digest: str
    initial_files: tuple[tuple[str, str], ...]
    final_files: tuple[tuple[str, str], ...]
    final_tree_digest: str

    def as_dict(self) -> dict[str, object]:
        # bodies excluded for compact audit records; digests preserved
        return {
            "candidate_id": self.candidate_id,
            "parent_id": self.parent_id,
            "task_name": self.task_name,
            "accepted": self.accepted,
            "reason": self.reason,
            "frozen_hidden": self.frozen_hidden,
            "adaptive_hidden": self.adaptive_hidden,
            "hidden_delta": self.hidden_delta,
            "trace_digest": self.trace_digest,
            "final_tree_digest": self.final_tree_digest,
        }


class LineageArchive:
    """Append-only, deterministic archive of every candidate evaluated.

    Preserves accepted *and* rejected candidates, parent->child lineage, patch
    trace digests, and the initial/final workspace contents needed to audit a
    diff and to deterministically replay an accepted candidate.
    """

    def __init__(self) -> None:
        self._entries: list[LineageEntry] = []
        self._by_id: dict[str, LineageEntry] = {}

    def record(
        self,
        *,
        decision: GateDecision,
        task: CodeEvalTask,
        final_files: Sequence[tuple[str, str]],
        parent_id: Optional[str] = None,
    ) -> LineageEntry:
        initial = _normalize_files(
            (name, payload)
            for name, payload in task.workspace_files.items()
        )
        final = _normalize_files(final_files)
        entry = LineageEntry(
            candidate_id=decision.candidate_key,
            parent_id=parent_id,
            task_name=task.name,
            accepted=decision.accepted,
            reason=decision.reason,
            frozen_hidden=decision.frozen.hidden.score,
            adaptive_hidden=decision.adaptive.hidden.score,
            hidden_delta=round(
                decision.adaptive.hidden.score - decision.frozen.hidden.score, 6
            ),
            trace_digest=decision.adaptive.trace_digest,
            initial_files=initial,
            final_files=final,
            final_tree_digest=_sha_json(list(final)),
        )
        self._entries.append(entry)
        self._by_id[entry.candidate_id] = entry
        return entry

    def entries(self) -> tuple[LineageEntry, ...]:
        return tuple(self._entries)

    def get(self, candidate_id: str) -> LineageEntry:
        return self._by_id[candidate_id]

    def lineage_of(self, candidate_id: str) -> list[LineageEntry]:
        chain: list[LineageEntry] = []
        cur: Optional[str] = candidate_id
        seen: set[str] = set()
        while cur is not None and cur in self._by_id and cur not in seen:
            seen.add(cur)
            entry = self._by_id[cur]
            chain.append(entry)
            cur = entry.parent_id
        chain.reverse()
        return chain

    def rejected_count(self) -> int:
        return sum(1 for e in self._entries if not e.accepted)

    def unified_diff(self, entry: LineageEntry) -> str:
        initial = dict(entry.initial_files)
        final = dict(entry.final_files)
        chunks: list[str] = []
        for name in sorted(set(initial) | set(final)):
            before = initial.get(name, "").splitlines(keepends=True)
            after = final.get(name, "").splitlines(keepends=True)
            diff = difflib.unified_diff(
                before, after, fromfile=f"a/{name}", tofile=f"b/{name}"
            )
            chunks.append("".join(diff))
        return "".join(c for c in chunks if c)

    def verify_replay(self, entry: LineageEntry) -> bool:
        """Tamper-evident integrity replay: recompute the digest of the archived
        final artifact and confirm it matches what was recorded at accept time."""
        return _sha_json(list(entry.final_files)) == entry.final_tree_digest

    def replay_evaluation(self, entry: LineageEntry, task: CodeEvalTask) -> bool:
        """Stronger replay: re-materialize the archived final artifact and re-run
        the task's hidden + full-suite checks, confirming the recorded adaptive
        hidden score reproduces. Deterministic given deterministic checks."""
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory(prefix="phase4_replay_") as tmp:
            workspace = Path(tmp) / task.name
            if workspace.exists():
                shutil.rmtree(workspace)
            workspace.mkdir(parents=True, exist_ok=True)
            for name, content in entry.final_files:
                target = workspace / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            hidden = task.hidden_check(workspace)
            if task.full_suite_check is not None:
                full = task.full_suite_check(workspace)
                if not full.passed:
                    return False
        return abs(hidden.score - entry.adaptive_hidden) < 1e-9


def _normalize_files(
    items,
) -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    for name, payload in items:
        if isinstance(payload, bytes):
            out.append((str(name), f"<bytes:{_sha(payload)}>"))
        else:
            out.append((str(name), str(payload)))
    out.sort(key=lambda kv: kv[0])
    return tuple(out)


def evaluate_with_lineage(
    harness: CodeFitnessHarness,
    task: CodeEvalTask,
    *,
    adaptive_runner,
    frozen_runner,
    candidate_id: str,
    archive: LineageArchive,
    root: Path,
    parent_id: Optional[str] = None,
) -> tuple[GateDecision, LineageEntry]:
    """Run an evaluation under a caller-owned root so the adaptive workspace can
    be snapshotted into the archive *without modifying the core harness*.

    The base harness, when given an explicit ``root``, leaves the adaptive
    workspace on disk at ``root/adaptive/<task.name>``; we read it back after the
    decision and hand it to the archive.
    """
    decision = harness.evaluate(
        task,
        adaptive_runner=adaptive_runner,
        frozen_runner=frozen_runner,
        candidate_id=candidate_id,
        root=root,
    )
    adaptive_ws = Path(root) / "adaptive" / task.name
    final_files = _read_tree(adaptive_ws)
    entry = archive.record(
        decision=decision, task=task, final_files=final_files, parent_id=parent_id
    )
    return decision, entry


def _read_tree(root: Path) -> list[tuple[str, str]]:
    root = Path(root)
    if not root.exists():
        return []
    out: list[tuple[str, str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        try:
            out.append((rel, path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            out.append((rel, f"<bytes:{_sha(path.read_bytes())}>"))
    return out


# --------------------------------------------------------------------------- #
# 5. issue/PR summary emission                                                #
# --------------------------------------------------------------------------- #
def summarize_decision(
    decision: GateDecision,
    *,
    rejected_count: int = 0,
    lineage_entry: Optional[LineageEntry] = None,
    diff_summary: Optional[str] = None,
) -> dict[str, object]:
    """Build the MVP acceptance summary required by issue #118: hidden score
    delta, frozen vs adaptive comparison, patch trace digest, rejected count,
    final diff summary."""
    adaptive, frozen = decision.adaptive, decision.frozen
    summary: dict[str, object] = {
        "accepted": decision.accepted,
        "reason": decision.reason,
        "candidate_key": decision.candidate_key,
        "hidden_score_delta": round(
            adaptive.hidden.score - frozen.hidden.score, 6
        ),
        "frozen_vs_adaptive": {
            "frozen": {
                "visible_passed": frozen.visible.passed,
                "hidden_score": frozen.hidden.score,
                "full_suite_passed": frozen.full_suite.passed,
            },
            "adaptive": {
                "visible_passed": adaptive.visible.passed,
                "hidden_score": adaptive.hidden.score,
                "full_suite_passed": adaptive.full_suite.passed,
            },
        },
        "patch_trace_digest": {
            "adaptive": adaptive.trace_digest,
            "frozen": frozen.trace_digest,
        },
        "rejected_candidate_count": rejected_count,
        "diff_summary": diff_summary
        if diff_summary is not None
        else _trace_diff_summary(adaptive),
    }
    if lineage_entry is not None:
        summary["lineage"] = {
            "candidate_id": lineage_entry.candidate_id,
            "parent_id": lineage_entry.parent_id,
            "final_tree_digest": lineage_entry.final_tree_digest,
        }
    return summary


def _trace_diff_summary(run: RunResult) -> dict[str, object]:
    committed = Counter()
    files: set[str] = set()
    for trace in run.traces:
        if trace.status == "committed":
            committed[trace.action] += 1
    return {
        "writes": committed.get("write_file", 0),
        "patches": committed.get("apply_patch", 0),
        "reverts": committed.get("revert_file", 0),
        "committed_actions": int(sum(committed.values())),
    }


def render_markdown(summary: Mapping[str, object]) -> str:
    """Render the summary as the issue/PR comment block #118 asks for."""
    fa = summary["frozen_vs_adaptive"]  # type: ignore[index]
    frozen = fa["frozen"]  # type: ignore[index]
    adaptive = fa["adaptive"]  # type: ignore[index]
    digest = summary["patch_trace_digest"]  # type: ignore[index]
    lines = [
        f"### Phase 4 verifier decision: {'ACCEPTED' if summary['accepted'] else 'REJECTED'} ({summary['reason']})",
        "",
        f"- hidden score delta: `{summary['hidden_score_delta']}`",
        f"- frozen hidden: `{frozen['hidden_score']}`  |  adaptive hidden: `{adaptive['hidden_score']}`",
        f"- frozen full-suite: `{frozen['full_suite_passed']}`  |  adaptive full-suite: `{adaptive['full_suite_passed']}`",
        f"- patch trace digest (adaptive): `{digest['adaptive']}`",
        f"- rejected candidate count: `{summary['rejected_candidate_count']}`",
        f"- diff summary: `{json.dumps(summary['diff_summary'], sort_keys=True)}`",
    ]
    if "lineage" in summary:
        lineage = summary["lineage"]  # type: ignore[index]
        lines.append(
            f"- lineage: `{lineage['candidate_id']}` <- parent `{lineage['parent_id']}` "
            f"(final tree `{lineage['final_tree_digest']}`)"
        )
    return "\n".join(lines)
