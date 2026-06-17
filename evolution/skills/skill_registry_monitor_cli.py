#!/usr/bin/env python3
"""Read-only Hermes skill registry monitor CLI.

This is intentionally conservative: it inventories SKILL.md files, compares the
current registry with a saved snapshot, and writes snapshots only when explicitly
requested. It never edits runtime skill files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RISK_PATTERNS: tuple[tuple[str, str], ...] = (
    ("credential mutation", r"\b(rotate|create|delete|modify|update|write|set|change)\b.{0,80}\b(api key|secret|credential|token|oauth|client|vault|oauth client)\b"),
    ("public mutation", r"\b(public post|publish|tweet|send email|gmail send|post to|comment on|reply publicly|upload to|deploy to production|release to)\b"),
    ("regulated data", r"\b(PHI|HIPAA|payroll|tax return|medical record|SSN|social security)\b"),
)
MENTION_TERMS = (
    "calendar", "credential", "drive", "github", "gmail", "oauth", "phone",
    "publish", "send email", "sms", "vault", "voice", "patient", "phi",
)


@dataclass(frozen=True)
class Diff:
    new: list[str]
    removed: list[str]
    changed_hashes: list[str]
    changed_paths: list[str]
    readiness_regressions: list[str]
    high_risk: list[str]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_frontmatter(text: str) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    if not text.startswith("---\n"):
        return {}, ["missing YAML frontmatter"]
    end = text.find("\n---", 4)
    if end == -1:
        return {}, ["unterminated YAML frontmatter"]
    raw = text[4:end]
    data: dict[str, Any] = {}
    # Tiny YAML subset sufficient for Hermes skill metadata. Avoid dependency on PyYAML.
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#") or line.startswith(" "):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"\'')
        if key and key not in data:
            data[key] = value
    if "name" not in data:
        failures.append("frontmatter missing name")
    return data, failures


def dataset_status(docs_root: Path | None, skill_name: str) -> tuple[bool, list[str]]:
    if docs_root is None:
        return False, ["docs root absent"]
    dataset_dir = docs_root / "datasets" / skill_name
    if not dataset_dir.exists():
        return False, ["dataset directory absent"]
    required = ["train.jsonl", "val.jsonl", "holdout.jsonl", "rubric.md"]
    missing = [name for name in required if not (dataset_dir / name).exists()]
    if missing:
        return False, ["missing dataset files: " + ", ".join(missing)]
    return True, []


def risk_labels_for(text: str) -> tuple[list[str], list[str]]:
    labels: set[str] = set()
    blockers: set[str] = set()
    lower = text.lower()
    for label, pattern in RISK_PATTERNS:
        if re.search(pattern, lower, flags=re.IGNORECASE | re.DOTALL):
            labels.add(label)
            blockers.add(label)
    for term in MENTION_TERMS:
        if term in lower:
            labels.add(f"mention:{term}")
    return sorted(labels), sorted(blockers)


def readiness_state(frontmatter_valid: bool, dataset_ready: bool, risk_blockers: list[str]) -> str:
    if not frontmatter_valid:
        return "invalid_frontmatter"
    if risk_blockers:
        return "high_risk_review_required"
    if not dataset_ready:
        return "needs_dataset"
    return "ready_for_monitoring"


def inventory(skills_root: Path, docs_root: Path | None) -> dict[str, Any]:
    skills: list[dict[str, Any]] = []
    for skill_file in sorted(skills_root.rglob("SKILL.md")):
        text = skill_file.read_text(encoding="utf-8", errors="replace")
        fm, fm_failures = extract_frontmatter(text)
        name = str(fm.get("name") or skill_file.parent.name)
        dataset_ready, dataset_failures = dataset_status(docs_root, name)
        labels, blockers = risk_labels_for(text)
        frontmatter_valid = not fm_failures
        state = readiness_state(frontmatter_valid, dataset_ready, blockers)
        ref_dir = skill_file.parent / "references"
        reference_count = sum(1 for p in ref_dir.rglob("*") if p.is_file()) if ref_dir.exists() else 0
        skills.append({
            "dataset_failures": dataset_failures,
            "dataset_ready": dataset_ready,
            "frontmatter_failures": fm_failures,
            "frontmatter_valid": frontmatter_valid,
            "name": name,
            "path": str(skill_file),
            "promotion_eligible_by_default": state == "ready_for_monitoring",
            "readiness_state": state,
            "reference_count": reference_count,
            "risk_blockers": blockers,
            "risk_labels": labels,
            "sha256": sha256_file(skill_file),
            "size_bytes": skill_file.stat().st_size,
        })
    return {"schema_version": 1, "skill_count": len(skills), "skills": skills}


def load_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def by_name(snapshot: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not snapshot:
        return {}
    return {str(item.get("name")): item for item in snapshot.get("skills", [])}


def compare(current: dict[str, Any], previous: dict[str, Any] | None) -> Diff:
    cur = by_name(current)
    prev = by_name(previous)
    new = sorted(set(cur) - set(prev))
    removed = sorted(set(prev) - set(cur))
    shared = sorted(set(cur) & set(prev))
    changed_hashes = [name for name in shared if cur[name].get("sha256") != prev[name].get("sha256")]
    changed_paths = [name for name in shared if cur[name].get("path") != prev[name].get("path")]
    readiness_regressions = [
        name for name in shared
        if prev[name].get("readiness_state") == "ready_for_monitoring"
        and cur[name].get("readiness_state") in {"invalid_frontmatter", "needs_dataset"}
    ]
    high_risk = sorted(name for name, row in cur.items() if row.get("readiness_state") == "high_risk_review_required")
    return Diff(new, removed, changed_hashes, changed_paths, readiness_regressions, high_risk)


def write_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_report(profile: str, status: str, snapshot_written: bool, count: int, diff: Diff) -> None:
    print(f"Skill registry monitor dry-run for {profile}: {status}")
    print(f"- Snapshot written: {'yes' if snapshot_written else 'no'}")
    print(f"- Skill count: {count}")
    print(f"Skill registry monitor: {status}")
    if diff.new:
        print("- New skills: " + ", ".join(diff.new))
    if diff.removed:
        print("- Removed skills: " + ", ".join(diff.removed))
    if diff.changed_paths:
        print("- Changed skill paths: " + ", ".join(diff.changed_paths))
    if diff.changed_hashes:
        print("- Changed skill hashes: " + ", ".join(diff.changed_hashes))
    if diff.readiness_regressions:
        print("- New readiness regressions: " + ", ".join(diff.readiness_regressions))
    if diff.high_risk:
        print("- High-risk monitor-only skills: " + ", ".join(diff.high_risk))
    if status != "clean":
        print("Next actions:")
        for name in diff.new:
            print(f"- Inventory new skill {name} and add prepared golden dataset before optimization")
        for name in diff.changed_hashes:
            print(f"- Review hash change before optimization/promotions for {name}")
        for name in diff.removed:
            print(f"- Confirm removed skill {name} was intentionally archived or pruned")
        for name in diff.changed_paths:
            print(f"- Confirm path change for {name} before refreshing registry")
        for name in diff.readiness_regressions:
            print(f"- Repair readiness regression for {name} or document monitor-only state")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Hermes skill registry monitor")
    parser.add_argument("--skills-root", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--docs-root")
    parser.add_argument("--write-snapshot", action="store_true")
    parser.add_argument("--silence-on-clean", action="store_true")
    parser.add_argument("--change-only", action="store_true")
    args = parser.parse_args(argv)

    skills_root = Path(args.skills_root)
    snapshot_path = Path(args.snapshot)
    docs_root = Path(args.docs_root) if args.docs_root else None
    if not skills_root.exists() or not skills_root.is_dir():
        print(f"Skill registry monitor FAILED: skills root not found: {skills_root}", file=sys.stderr)
        return 2
    if docs_root is not None and not docs_root.exists():
        print(f"Skill registry monitor FAILED: docs root not found: {docs_root}", file=sys.stderr)
        return 2

    current = inventory(skills_root, docs_root)
    previous = load_snapshot(snapshot_path)
    diff = compare(current, previous)
    status = "baseline_created" if previous is None else "clean"
    actionable = bool(diff.new or diff.removed or diff.changed_hashes or diff.changed_paths or diff.readiness_regressions)
    if previous is not None and actionable:
        status = "action_needed"

    snapshot_written = False
    if args.write_snapshot:
        write_snapshot(snapshot_path, current)
        snapshot_written = True

    if status == "clean" and args.silence_on_clean:
        print("[SILENT]")
        return 0
    if args.change_only and status == "clean" and not snapshot_written:
        if args.silence_on_clean:
            print("[SILENT]")
        return 0

    print_report(args.profile, status, snapshot_written, current["skill_count"], diff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
