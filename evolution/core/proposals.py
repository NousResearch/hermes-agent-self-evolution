"""Proposal writer — persists evolved-skill proposals for human review.

When `evolve_skill` runs in `--mode propose` (or when the auto-merge
gate rejects an evolved variant), a ProposalRecord is written under
`proposals_dir/{skill_name}/{timestamp}/` containing everything a
reviewer needs to decide: baseline text, evolved text, unified diff,
per-constraint results, and the gate decision.

The on-disk layout is intentionally simple so the future ProposalReviewer
CLI (Task 6) can list + approve + reject with nothing but `pathlib`.

Layout:
    proposals_dir/
      {skill_name}/
        {YYYYMMDD_HHMMSS}/
          baseline_skill.md        # exact original skill text
          evolved_skill.md         # reassembled evolved skill (frontmatter + body)
          diff.patch               # unified diff baseline -> evolved
          decision.json            # GateDecision + scores + delta + metadata
          constraints.json         # list of {name, passed, message} for evolved
          review.md                # human-readable summary with approval checklist
          STATUS                   # single-line state: PENDING / APPROVED / REJECTED
"""

from __future__ import annotations

import difflib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROPOSAL_STATUS_PENDING = "PENDING"
PROPOSAL_STATUS_APPROVED = "APPROVED"
PROPOSAL_STATUS_REJECTED = "REJECTED"


@dataclass
class ConstraintRecord:
    """Serializable shape of a ConstraintResult."""
    name: str
    passed: bool
    message: str

    @classmethod
    def from_result(cls, result: Any) -> "ConstraintRecord":
        return cls(
            name=getattr(result, "constraint_name", "unknown"),
            passed=bool(getattr(result, "passed", False)),
            message=str(getattr(result, "message", "")),
        )


@dataclass
class ProposalRecord:
    """Everything a reviewer needs for one evolved-skill proposal."""
    skill_name: str
    timestamp: str
    baseline_text: str
    evolved_text: str
    baseline_score: float
    evolved_score: float
    improvement: float
    mode: str
    auto_merge: bool
    gate_reason: str
    regression: bool
    constraints: List[ConstraintRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def baseline_size(self) -> int:
        return len(self.baseline_text)

    @property
    def evolved_size(self) -> int:
        return len(self.evolved_text)

    @property
    def size_delta(self) -> int:
        return self.evolved_size - self.baseline_size


class ProposalWriter:
    """Write ProposalRecord to disk in the standard layout."""

    def __init__(self, proposals_dir: Path | str):
        self.root = Path(proposals_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, record: ProposalRecord) -> Path:
        """Persist the proposal and return the directory written to."""
        skill_dir = self.root / record.skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        proposal_dir = skill_dir / record.timestamp
        proposal_dir.mkdir(parents=True, exist_ok=True)

        # Raw artifacts
        (proposal_dir / "baseline_skill.md").write_text(record.baseline_text)
        (proposal_dir / "evolved_skill.md").write_text(record.evolved_text)

        # Unified diff
        diff = self._unified_diff(record.baseline_text, record.evolved_text)
        (proposal_dir / "diff.patch").write_text(diff)

        # Gate decision + scores
        decision_payload = {
            "skill_name": record.skill_name,
            "timestamp": record.timestamp,
            "mode": record.mode,
            "baseline_score": record.baseline_score,
            "evolved_score": record.evolved_score,
            "improvement": record.improvement,
            "auto_merge": record.auto_merge,
            "gate_reason": record.gate_reason,
            "regression": record.regression,
            "baseline_size": record.baseline_size,
            "evolved_size": record.evolved_size,
            "size_delta": record.size_delta,
            "metadata": record.metadata,
        }
        (proposal_dir / "decision.json").write_text(json.dumps(decision_payload, indent=2))

        # Constraints
        constraints_payload = [asdict(c) for c in record.constraints]
        (proposal_dir / "constraints.json").write_text(json.dumps(constraints_payload, indent=2))

        # Human-readable review doc
        (proposal_dir / "review.md").write_text(self._render_review(record, proposal_dir))

        # Status tombstone — ProposalReviewer (Task 6) updates this file.
        (proposal_dir / "STATUS").write_text(PROPOSAL_STATUS_PENDING + "\n")

        return proposal_dir

    # ───────────────────────── internal helpers ─────────────────────────
    @staticmethod
    def _unified_diff(baseline: str, evolved: str) -> str:
        return "".join(
            difflib.unified_diff(
                baseline.splitlines(keepends=True),
                evolved.splitlines(keepends=True),
                fromfile="baseline_skill.md",
                tofile="evolved_skill.md",
                n=3,
            )
        )

    @staticmethod
    def _render_review(record: ProposalRecord, proposal_dir: Path) -> str:
        pct = (record.improvement / record.baseline_score * 100.0) if record.baseline_score else 0.0
        constraint_lines = []
        for c in record.constraints:
            icon = "✅" if c.passed else "❌"
            constraint_lines.append(f"- {icon} **{c.name}** — {c.message}")
        constraints_block = "\n".join(constraint_lines) if constraint_lines else "_(no constraint results)_"

        auto_merge_line = "🟢 eligible for auto-merge" if record.auto_merge else "🔴 NOT eligible for auto-merge"
        regression_warning = "\n> ⚠️  **Regression detected** — this variant scored materially worse than the baseline.\n" if record.regression else ""

        return f"""# Evolution Proposal — `{record.skill_name}`

**Timestamp:** `{record.timestamp}`
**Mode:** `{record.mode}`
**Status:** PENDING human review

## Scores

| Metric | Baseline | Evolved | Δ |
|---|---:|---:|---:|
| Holdout score | `{record.baseline_score:.3f}` | `{record.evolved_score:.3f}` | `{record.improvement:+.3f}` ({pct:+.1f}%) |
| Size (chars) | `{record.baseline_size:,}` | `{record.evolved_size:,}` | `{record.size_delta:+,}` |

## Gate Decision
{auto_merge_line}

> {record.gate_reason}
{regression_warning}
## Constraints (evolved variant)

{constraints_block}

## Review Checklist

- [ ] Diff preserves skill's core purpose (no silent scope drift)
- [ ] No secrets, paths, or user-specific tokens leaked into the skill body
- [ ] Frontmatter (`name`, `description`, `category`) is intact and accurate
- [ ] Tone/voice matches existing Hermes skills
- [ ] Examples and commands still correct
- [ ] No hallucinated APIs, flags, or file paths

## Artifacts

- `baseline_skill.md` — original skill as it exists in hermes-agent
- `evolved_skill.md` — proposed replacement
- `diff.patch` — unified diff (`diff -u baseline evolved`)
- `decision.json` — gate decision + scores
- `constraints.json` — per-constraint pass/fail for the evolved variant

## Approving

When satisfied, update `STATUS` to `APPROVED` (or use the ProposalReviewer CLI).
To reject, update `STATUS` to `REJECTED` with an optional reason on a second line.
"""


def build_proposal_record(
    *,
    skill_name: str,
    baseline_text: str,
    evolved_text: str,
    baseline_score: float,
    evolved_score: float,
    decision: Any,  # GateDecision
    constraint_results: List[Any],
    mode: str,
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> ProposalRecord:
    """Convenience assembler used by evolve_skill.py."""
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return ProposalRecord(
        skill_name=skill_name,
        timestamp=ts,
        baseline_text=baseline_text,
        evolved_text=evolved_text,
        baseline_score=float(baseline_score),
        evolved_score=float(evolved_score),
        improvement=float(evolved_score) - float(baseline_score),
        mode=mode,
        auto_merge=bool(getattr(decision, "auto_merge", False)),
        gate_reason=str(getattr(decision, "reason", "")),
        regression=bool(getattr(decision, "regression", False)),
        constraints=[ConstraintRecord.from_result(r) for r in constraint_results],
        metadata=metadata or {},
    )
