"""Evolve a Hermes Agent skill using DSPy + GEPA.

Usage:
    python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/
"""

import difflib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

import click
import dspy
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evolution.core.candidate_bundle import (
    create_candidate_bundle,
    write_bundle_json,
    write_bundle_text,
    write_decision,
)
from evolution.core.config import EvolutionConfig, get_hermes_agent_path
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader, expand_objective_examples as expand_objective_dataset_examples
from evolution.core.external_importers import build_dataset_from_external
from evolution.core.fitness import (
    skill_fitness_metric,
    skill_fitness_feedback_metric,
    LLMJudge,
    FitnessScore,
    _score_rubric_checks,
    _skill_size_penalty,
)
from evolution.core.constraints import ConstraintValidator
from evolution.core.hermes_lm import make_lm
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)

console = Console()


def write_skill_candidate_bundle(
    *,
    skill_name: str,
    baseline_skill: str,
    evolved_skill: str,
    metrics: dict,
    constraint_results: list,
    seed_skill: str | None = None,
    candidate_selection: Mapping[str, Any] | None = None,
    decision_summary: str | None = None,
) -> Path:
    """Write a Phase 1 result using the local candidate bundle contract."""

    bundle = create_candidate_bundle(
        phase="Phase 1: Skill Evolution",
        target=skill_name,
        run_id=datetime.now().strftime("%Y%m%d_%H%M%S_%f_phase1"),
    )
    sanitized_metrics = _sanitize_skill_candidate_metrics(metrics)
    if candidate_selection is not None:
        sanitized_metrics["candidate_selection"] = dict(candidate_selection)
    write_bundle_text(bundle, "candidates/baseline_skill.md", baseline_skill)
    write_bundle_text(bundle, "candidates/evolved_skill.md", evolved_skill)
    if seed_skill is not None:
        write_bundle_text(bundle, "candidates/seed_skill.md", seed_skill)
    diff_text = _skill_diff(baseline_skill, evolved_skill)
    write_bundle_text(bundle, "candidates/candidate.patch", diff_text)
    write_bundle_json(bundle, "eval/metrics.json", sanitized_metrics)
    write_bundle_json(
        bundle,
        "eval/constraint_results.json",
        [asdict(result) for result in constraint_results],
    )
    artifacts = {
        "baseline": "candidates/baseline_skill.md",
        "evolved": "candidates/evolved_skill.md",
        "patch": "candidates/candidate.patch",
        "metrics": "eval/metrics.json",
        "constraints": "eval/constraint_results.json",
        "report": "reports/report.md",
        "rollback": "reports/rollback.md",
    }
    if candidate_selection is not None:
        write_bundle_json(bundle, "eval/candidate_selection.json", dict(candidate_selection))
        artifacts["candidate_selection"] = "eval/candidate_selection.json"
        quality_repair_evidence = candidate_selection.get("quality_repair_evidence")
        if isinstance(quality_repair_evidence, Mapping):
            write_bundle_json(bundle, "eval/quality_repair_evidence.json", dict(quality_repair_evidence))
            artifacts["quality_repair_evidence"] = "eval/quality_repair_evidence.json"
    write_bundle_text(bundle, "reports/report.md", _render_skill_candidate_report(skill_name, sanitized_metrics, diff_text))
    write_bundle_text(bundle, "reports/verification.log", _render_skill_verification_log(constraint_results, sanitized_metrics))
    write_bundle_text(bundle, "reports/rollback.md", _render_skill_rollback_note(skill_name))
    decision_status = _decision_status_for_skill_candidate(
        diff_text=diff_text,
        improvement=float(sanitized_metrics.get("improvement", 0.0) or 0.0),
        constraints_passed=bool(sanitized_metrics.get("constraints_passed", False)),
        candidate_selection=candidate_selection,
    )
    write_decision(
        bundle,
        status=decision_status,
        summary=decision_summary
        or "Phase 1 skill candidate generated locally; active skills and GitHub PRs were not modified.",
        metrics=sanitized_metrics,
        artifacts=artifacts,
    )
    return bundle.root


def _sanitize_skill_candidate_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = dict(metrics)
    if sanitized.get("seed_skill_path"):
        sanitized["seed_skill_path"] = "[redacted-seed-skill-path]"
    return sanitized


def _resolve_skill_path(
    skill_name: str,
    hermes_agent_path: Path,
    *,
    skills_root: str | None = None,
    skill_path: str | None = None,
) -> Path | None:
    """Resolve a skill path from an explicit SKILL.md, skills root, or Hermes repo.

    User-local/profile-local skills live under ``~/.hermes/skills`` and are not
    necessarily present under the Hermes source checkout. Explicit ``--skill-path``
    and ``--skills-root`` keep candidate generation profile-aware without a
    temporary facade.
    """

    if skill_path:
        resolved = Path(skill_path).expanduser()
        return resolved if resolved.exists() and resolved.is_file() else None

    if skills_root:
        root = Path(skills_root).expanduser()
        if root.exists():
            for skill_md in root.rglob("SKILL.md"):
                if skill_md.parent.name == skill_name:
                    return skill_md
            for skill_md in root.rglob("SKILL.md"):
                try:
                    content = skill_md.read_text()[:500]
                except Exception:
                    continue
                if f"name: {skill_name}" in content or f'name: "{skill_name}"' in content:
                    return skill_md

    return find_skill(skill_name, hermes_agent_path)


def _display_skill_path(skill_path: Path, hermes_agent_path: Path) -> str:
    try:
        return str(skill_path.relative_to(hermes_agent_path))
    except ValueError:
        return str(skill_path)


def _decision_status_for_skill_candidate(
    *,
    diff_text: str,
    improvement: float,
    constraints_passed: bool,
    candidate_selection: Mapping[str, Any] | None = None,
) -> str:
    if not diff_text.strip() or diff_text.startswith("No candidate skill changes."):
        return "NO_DIFF_NO_GO"
    if not constraints_passed:
        return "REGRESSION_NO_GO"
    if candidate_selection and candidate_selection.get("selected_candidate_val_score_below_gepa_best") is True:
        return "REGRESSION_NO_GO"
    if improvement > 0:
        return "PASS_CANDIDATE_ONLY"
    if improvement < 0:
        return "REGRESSION_NO_GO"
    return "INCONCLUSIVE"


def _skill_diff(baseline_skill: str, evolved_skill: str) -> str:
    if baseline_skill == evolved_skill:
        return "No candidate skill changes.\n"
    return "".join(
        difflib.unified_diff(
            baseline_skill.splitlines(keepends=True),
            evolved_skill.splitlines(keepends=True),
            fromfile="baseline_skill.md",
            tofile="evolved_skill.md",
        )
    )


def _select_non_empty_export_candidate(
    optimized_module: Any,
    *,
    baseline_body: str,
    quality_examples: Sequence[Any] | None = None,
    content_level_repair: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Select the skill body to export from a GEPA run.

    GEPA can legitimately decide that the best validation program is the seed
    program. In Phase 1 skill evolution this creates an empty candidate diff,
    while later stochastic holdout calls may still show a tiny score movement.
    When ``track_stats=True`` gives access to proposed candidates, export the
    highest-validation non-empty proposal as a candidate artifact instead of
    silently exporting the seed again. If that proposal scored below GEPA's best
    program, the bundle decision remains ``REGRESSION_NO_GO``.
    """

    best_text = _module_skill_text(optimized_module)
    detailed = getattr(optimized_module, "detailed_results", None)
    candidates = list(getattr(detailed, "candidates", []) or [])
    scores = list(getattr(detailed, "val_aggregate_scores", []) or [])
    best_idx = _gepa_best_index(detailed, scores)
    best_score = _score_at(scores, best_idx)

    non_empty_candidates: list[tuple[int, Any, float | None]] = []
    for idx, candidate in enumerate(candidates):
        try:
            candidate_text = _module_skill_text(candidate)
        except AttributeError:
            continue
        if candidate_text != baseline_body:
            non_empty_candidates.append((idx, candidate, _score_at(scores, idx)))

    base_metadata: dict[str, Any] = {
        "candidate_count": len(candidates),
        "non_empty_candidate_count": len(non_empty_candidates),
        "gepa_best_candidate_index": best_idx,
        "gepa_best_val_score": best_score,
        "gepa_best_was_empty_diff": best_text == baseline_body,
    }

    if best_text != baseline_body:
        selected_idx = best_idx
        selected_score = _score_at(scores, selected_idx)
        return optimized_module, {
            **base_metadata,
            "strategy": "gepa_best",
            "selected_candidate_index": selected_idx,
            "selected_candidate_val_score": selected_score,
            "selected_candidate_has_non_empty_diff": True,
            "selected_candidate_val_score_below_gepa_best": False,
        }

    if not non_empty_candidates:
        if content_level_repair:
            content_repaired = _quality_repair_content_candidate(
                baseline_body=baseline_body,
                selected_body=baseline_body,
                quality_examples=quality_examples,
            )
            if content_repaired is not None:
                repaired_body, repair_metadata = content_repaired
                return SkillModule(repaired_body), {
                    **base_metadata,
                    **repair_metadata,
                    "strategy": "quality_repaired_compact_content_from_empty_gepa",
                    "raw_selected_candidate_index": best_idx,
                    "raw_selected_candidate_val_score": best_score,
                    "raw_selected_candidate_val_score_below_gepa_best": False,
                    "selected_candidate_index": "quality_repaired_compact_content",
                    "selected_candidate_val_score": best_score,
                    "selected_candidate_has_non_empty_diff": True,
                    "selected_candidate_val_score_below_gepa_best": False,
                }
        strategy = "gepa_best_empty_no_candidate_stats" if detailed is None else "gepa_best_empty_no_non_empty_candidate"
        return optimized_module, {
            **base_metadata,
            "strategy": strategy,
            "selected_candidate_index": best_idx,
            "selected_candidate_val_score": best_score,
            "selected_candidate_has_non_empty_diff": False,
            "selected_candidate_val_score_below_gepa_best": False,
        }

    selected_idx, selected_candidate, selected_score = max(
        non_empty_candidates,
        key=lambda item: (float("-inf") if item[2] is None else item[2], -item[0]),
    )
    below_best = best_score is not None and selected_score is not None and selected_score < best_score
    metadata = {
        **base_metadata,
        "strategy": "best_non_empty_gepa_candidate",
        "selected_candidate_index": selected_idx,
        "selected_candidate_val_score": selected_score,
        "selected_candidate_has_non_empty_diff": True,
        "selected_candidate_val_score_below_gepa_best": below_best,
    }
    if below_best:
        selected_body = _module_skill_text(selected_candidate)
        if content_level_repair:
            content_repaired = _quality_repair_content_candidate(
                baseline_body=baseline_body,
                selected_body=selected_body,
                quality_examples=quality_examples,
            )
            if content_repaired is not None:
                repaired_body, repair_metadata = content_repaired
                return SkillModule(repaired_body), {
                    **metadata,
                    **repair_metadata,
                    "strategy": "quality_repaired_compact_content",
                    "raw_selected_candidate_index": selected_idx,
                    "raw_selected_candidate_val_score": selected_score,
                    "raw_selected_candidate_val_score_below_gepa_best": below_best,
                    "selected_candidate_index": "quality_repaired_compact_content",
                    "selected_candidate_val_score": best_score,
                    "selected_candidate_has_non_empty_diff": True,
                    "selected_candidate_val_score_below_gepa_best": False,
                }
        repaired = _quality_repair_regressive_candidate(
            baseline_body=baseline_body,
            selected_body=selected_body,
            quality_examples=quality_examples,
        )
        if repaired is not None:
            repaired_body, repair_metadata = repaired
            return SkillModule(repaired_body), {
                **metadata,
                **repair_metadata,
                "strategy": "quality_repaired_non_disruptive_evidence",
                "raw_selected_candidate_index": selected_idx,
                "raw_selected_candidate_val_score": selected_score,
                "raw_selected_candidate_val_score_below_gepa_best": below_best,
                "selected_candidate_index": "quality_repaired_baseline",
                "selected_candidate_val_score": best_score,
                "selected_candidate_has_non_empty_diff": False,
                "selected_candidate_val_score_below_gepa_best": False,
            }
    return selected_candidate, metadata


def _quality_repair_content_candidate(
    *,
    baseline_body: str,
    selected_body: str,
    quality_examples: Sequence[Any] | None,
    max_size_delta: int = 900,
) -> tuple[str, dict[str, Any]] | None:
    """Create a compact runtime-body candidate from missing rubric evidence.

    This is stronger than the evidence-only sidecar repair: it writes a bounded,
    human-reviewable section into the candidate skill body while preserving the
    soft-size penalty budget. If the repair would worsen size penalty or fail to
    improve deterministic document coverage, it fails closed and callers can
    fall back to non-disruptive sidecar evidence.
    """

    examples = list(quality_examples or [])
    if not examples:
        return None

    baseline_score = _aggregate_skill_doc_rubric_score(baseline_body, examples)
    selected_score = _aggregate_skill_doc_rubric_score(selected_body, examples)
    evidence_block = _render_quality_repair_addendum(baseline_body, examples)
    evidence_lines = _quality_repair_evidence_lines(evidence_block)
    compact_section = _render_compact_content_repair_section(evidence_lines)
    if not compact_section:
        return None

    repaired_body = _replace_common_pitfalls_with_content_repair(baseline_body, compact_section)
    if repaired_body.rstrip() == baseline_body.rstrip():
        return None

    size_delta = len(repaired_body) - len(baseline_body)
    baseline_size_penalty = _skill_size_penalty(baseline_body)
    repaired_size_penalty = _skill_size_penalty(repaired_body)
    if size_delta > max_size_delta or repaired_size_penalty > baseline_size_penalty:
        return None

    repaired_score = _aggregate_skill_doc_rubric_score(repaired_body, examples)
    if repaired_score < baseline_score or repaired_score <= selected_score:
        return None

    return repaired_body, {
        "quality_repair_applied": True,
        "quality_repair_gate_passed": True,
        "quality_repair_strategy": "compact_content_repair",
        "quality_repair_evidence_placement": "runtime_compact_section_and_metadata_sidecar",
        "quality_repair_doc_score_baseline": baseline_score,
        "quality_repair_doc_score_before": selected_score,
        "quality_repair_doc_score_after": repaired_score,
        "quality_repair_runtime_doc_score_after": repaired_score,
        "quality_repair_size_delta": size_delta,
        "quality_repair_size_penalty_baseline": baseline_size_penalty,
        "quality_repair_size_penalty_after": repaired_size_penalty,
        "quality_repair_size_penalty_delta": repaired_size_penalty - baseline_size_penalty,
        "quality_repair_evidence": {
            "placement": "runtime_compact_section_and_metadata_sidecar",
            "format": "compact_content_repair",
            "lines": evidence_lines,
        },
    }


def _render_compact_content_repair_section(evidence_lines: Sequence[str]) -> str:
    evidence_text = "\n".join(evidence_lines)
    if not evidence_text.strip():
        return ""

    lines = ["## Common Pitfalls and Evidence Hints", ""]
    cues: list[str] = []
    if any(token in evidence_text for token in ("git log main..HEAD", "--name-only", "/pulls/$PR_NUMBER/files", "pull/$PR_NUMBER/head", "gh pr checkout")):
        cues.append(
            "use `git log main..HEAD`; `gh pr diff ... --name-only` or `/pulls/$PR_NUMBER/files`; `git fetch origin pull/$PR_NUMBER/head` or `gh pr checkout`"
        )
    if any(token in evidence_text for token in ("Never fabricate", "PR number", "pulls/$PR_NUMBER/comments", "side=RIGHT", "line is not part of the diff")):
        cues.append(
            "Never fabricate; inline comments need PR number, File path, Line number, Comment body; use `pulls/$PR_NUMBER/comments`, `side=RIGHT`; GitHub rejects comments when the line is not part of the diff"
        )
    if not cues:
        return ""
    lines.append("; ".join(cues) + ".")
    return "\n".join(lines).rstrip() + "\n"


def _replace_common_pitfalls_with_content_repair(baseline_body: str, compact_section: str) -> str:
    section_start = baseline_body.find("\n## Common Pitfalls")
    if section_start == -1 and baseline_body.startswith("## Common Pitfalls"):
        section_start = 0
    if section_start != -1:
        prefix = baseline_body[:section_start].rstrip()
        suffix = baseline_body[section_start:].lstrip("\n")
        return f"{prefix}\n\n{compact_section.rstrip()}\n\n{suffix}"
    verification_start = baseline_body.find("\n## Verification Checklist")
    if verification_start != -1:
        prefix = baseline_body[:verification_start].rstrip()
        suffix = baseline_body[verification_start:].lstrip("\n")
        return f"{prefix}\n\n{compact_section.rstrip()}\n\n{suffix}"
    return f"{baseline_body.rstrip()}\n\n{compact_section}"


def _quality_repair_regressive_candidate(
    *,
    baseline_body: str,
    selected_body: str,
    quality_examples: Sequence[Any] | None,
) -> tuple[str, dict[str, Any]] | None:
    """Build non-disruptive quality evidence for a regressive candidate.

    Earlier repairs appended a rubric-derived HTML comment to the exported
    skill body. That preserved audit evidence, but it also polluted the runtime
    instruction surface and could trigger the metric's soft size penalty. Keep
    the runtime candidate at the baseline body and place the audit evidence in
    bundle metadata/sidecar JSON instead.
    """

    examples = list(quality_examples or [])
    if not examples:
        return None

    baseline_score = _aggregate_skill_doc_rubric_score(baseline_body, examples)
    selected_score = _aggregate_skill_doc_rubric_score(selected_body, examples)
    evidence_block = _render_quality_repair_addendum(baseline_body, examples)
    if not evidence_block:
        return None

    evidence_augmented_body = f"{baseline_body.rstrip()}\n\n{evidence_block.rstrip()}\n"
    evidence_augmented_score = _aggregate_skill_doc_rubric_score(evidence_augmented_body, examples)
    if evidence_augmented_score < baseline_score or evidence_augmented_score <= selected_score:
        return None

    evidence_lines = _quality_repair_evidence_lines(evidence_block)
    return baseline_body, {
        "quality_repair_applied": True,
        "quality_repair_gate_passed": True,
        "quality_repair_strategy": "baseline_with_metadata_sidecar_evidence",
        "quality_repair_evidence_placement": "metadata_sidecar",
        "quality_repair_doc_score_baseline": baseline_score,
        "quality_repair_doc_score_before": selected_score,
        "quality_repair_doc_score_after": evidence_augmented_score,
        "quality_repair_runtime_doc_score_after": baseline_score,
        "quality_repair_size_delta": 0,
        "quality_repair_sidecar_size_bytes": len(evidence_block.encode()),
        "quality_repair_evidence": {
            "placement": "metadata_sidecar",
            "format": "rubric_missing_lines",
            "lines": evidence_lines,
        },
    }


def _quality_repair_evidence_lines(evidence_block: str) -> list[str]:
    return [line for line in evidence_block.splitlines() if line.startswith("- ")]


def _aggregate_skill_doc_rubric_score(skill_text: str, examples: Sequence[Any]) -> float:
    scores: list[float] = []
    for example in examples:
        checks = getattr(example, "rubric_checks", None) or []
        if checks:
            scores.append(_score_rubric_checks(skill_text, checks))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _render_quality_repair_addendum(baseline_body: str, examples: Sequence[Any]) -> str:
    missing_lines: list[str] = []
    seen: set[str] = set()
    for example in examples:
        for check in getattr(example, "rubric_checks", None) or []:
            if not isinstance(check, Mapping):
                continue
            if _score_rubric_checks(baseline_body, [check]) >= 1.0:
                continue
            description = str(check.get("description") or check.get("id") or "quality check")
            examples_for_check = _rubric_check_examples(check)
            if not examples_for_check:
                continue
            key = f"{description}|{'|'.join(examples_for_check)}"
            if key in seen:
                continue
            seen.add(key)
            cue_text = "; ".join(f"`{example_text}`" for example_text in examples_for_check[:4])
            missing_lines.append(f"- {description}: preserve explicit guidance for {cue_text}.")

    if not missing_lines:
        return ""

    return "\n".join(
        [
            "<!-- HSE candidate-quality evidence (candidate-only; preserve baseline behavior; do not treat this as external-write approval):",
            *missing_lines,
            "-->",
        ]
    )


def _rubric_check_examples(check: Mapping[str, Any]) -> list[str]:
    examples: list[str] = []
    for key in ("pattern_all", "pattern_any"):
        raw_patterns = check.get(key)
        if not isinstance(raw_patterns, list | tuple):
            continue
        for pattern in raw_patterns:
            if isinstance(pattern, str) and pattern:
                examples.append(_rubric_pattern_example(pattern))
    return [example for example in examples if example]


def _rubric_pattern_example(pattern: str) -> str:
    example = pattern.strip().strip("^").strip("$")
    replacements = {
        r"\$": "$",
        r"\.": ".",
        r"\?": "?",
        r"\"": '"',
        r"\\": "",
        ".?": "",
        ".*": " ... ",
    }
    for old, new in replacements.items():
        example = example.replace(old, new)
    example = example.replace("|", " or ")
    return " ".join(example.split())


def _module_skill_text(module: Any) -> str:
    skill_text = getattr(module, "skill_text")
    if not isinstance(skill_text, str):
        raise AttributeError("candidate module does not expose string skill_text")
    return skill_text


def _gepa_best_index(detailed: Any, scores: list[Any]) -> int | None:
    raw_best_idx = getattr(detailed, "best_idx", None) if detailed is not None else None
    if isinstance(raw_best_idx, int):
        return raw_best_idx
    if scores:
        return max(range(len(scores)), key=lambda idx: scores[idx])
    return None


def _score_at(scores: list[Any], idx: int | None) -> float | None:
    if idx is None or idx < 0 or idx >= len(scores):
        return None
    score = scores[idx]
    return float(score) if isinstance(score, int | float) else None


def _render_skill_candidate_report(skill_name: str, metrics: Mapping[str, object], diff_text: str) -> str:
    lines = [
        f"# Phase 1 Skill Candidate Bundle — {skill_name}",
        "",
        "Status: candidate-only; active Hermes skills were not modified.",
        "",
        "## Metrics",
        "",
        f"- baseline_score: `{metrics.get('baseline_score')}`",
        f"- evolved_score: `{metrics.get('evolved_score')}`",
        f"- improvement: `{metrics.get('improvement')}`",
        f"- constraints_passed: `{metrics.get('constraints_passed')}`",
        "",
    ]
    candidate_selection = metrics.get("candidate_selection")
    if isinstance(candidate_selection, Mapping):
        lines.extend(
            [
                "## Candidate selection",
                "",
                f"- strategy: `{candidate_selection.get('strategy')}`",
                f"- non_empty_candidate_count: `{candidate_selection.get('non_empty_candidate_count')}`",
                f"- selected_candidate_val_score: `{candidate_selection.get('selected_candidate_val_score')}`",
                f"- gepa_best_val_score: `{candidate_selection.get('gepa_best_val_score')}`",
                f"- selected_candidate_val_score_below_gepa_best: `{candidate_selection.get('selected_candidate_val_score_below_gepa_best')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Diff",
            "",
            "```diff",
            diff_text.rstrip(),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _render_skill_verification_log(constraint_results: list, metrics: Mapping[str, object]) -> str:
    lines = ["Phase 1 skill candidate verification", "", "Constraints:"]
    for result in constraint_results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"- {status} {result.constraint_name}: {result.message}")
    lines.extend(
        [
            "",
            f"constraints_passed={metrics.get('constraints_passed')}",
            f"holdout_examples={metrics.get('holdout_examples')}",
            f"holdout_total_examples={metrics.get('holdout_total_examples')}",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_skill_rollback_note(skill_name: str) -> str:
    return (
        f"# Rollback note — {skill_name}\n\n"
        "No active skill file was modified by this candidate-only run. "
        "If a human later applies `candidates/candidate.patch`, rollback by "
        "restoring the prior active SKILL.md from `candidates/baseline_skill.md` "
        "or by reverting the local apply commit.\n"
    )


def _build_gepa_optimizer(*, iterations: int, optimizer_model: str, hermes_repo: str):
    """Build GEPA with the feedback-rich skill objective.

    Returning score-only metrics gives GEPA generic reflective feedback. For HSE
    skill evolution, use ``skill_fitness_feedback_metric`` so reflection receives
    missing rubric and External Write Gate guidance, not just a scalar.
    """

    return dspy.GEPA(
        metric=skill_fitness_feedback_metric,
        max_metric_calls=iterations,
        reflection_lm=make_lm(optimizer_model, hermes_repo=hermes_repo),
        track_stats=True,
    )


def evolve(
    skill_name: str,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    hermes_repo: Optional[str] = None,
    seed_skill_path: Optional[str] = None,
    skills_root: Optional[str] = None,
    skill_path: Optional[str] = None,
    holdout_limit: Optional[int] = None,
    run_tests: bool = False,
    dry_run: bool = False,
    expand_objective_examples: bool = False,
    content_level_repair: bool = False,
):
    """Main evolution function — orchestrates the full optimization loop."""

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,  # Use same model for dataset generation
        run_pytest=run_tests,
    )
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    # ── 1. Find and load the skill ──────────────────────────────────────
    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Evolving skill: [bold]{skill_name}[/bold]\n")

    resolved_skill_path = _resolve_skill_path(
        skill_name,
        config.hermes_agent_path,
        skills_root=skills_root,
        skill_path=skill_path,
    )
    if not resolved_skill_path:
        search_locations = [str(config.hermes_agent_path / "skills")]
        if skills_root:
            search_locations.insert(0, str(Path(skills_root).expanduser()))
        if skill_path:
            search_locations.insert(0, str(Path(skill_path).expanduser()))
        console.print(
            f"[red]✗ Skill '{skill_name}' not found in {', '.join(search_locations)}[/red]"
        )
        sys.exit(1)

    skill = load_skill(resolved_skill_path)
    console.print(f"  Loaded: {_display_skill_path(resolved_skill_path, config.hermes_agent_path)}")
    console.print(f"  Name: {skill['name']}")
    console.print(f"  Size: {len(skill['raw']):,} chars")
    console.print(f"  Description: {skill['description'][:80]}...")

    seed_skill = None
    if seed_skill_path:
        seed_path = Path(seed_skill_path).expanduser()
        if not seed_path.exists():
            console.print(f"[red]✗ Seed skill not found: {seed_path}[/red]")
            sys.exit(1)
        seed_skill = load_skill(seed_path)
        console.print(f"  Seed: {seed_path} ({len(seed_skill['raw']):,} chars)")

    if dry_run:
        console.print(f"\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        console.print(f"  Would validate constraints and create PR")
        return

    # ── 2. Build or load evaluation dataset ─────────────────────────────
    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")

    if eval_source == "golden" and dataset_path:
        dataset = GoldenDatasetLoader.load(Path(dataset_path))
        console.print(f"  Loaded golden dataset: {len(dataset.all_examples)} examples")
    elif eval_source == "golden":
        dataset_path_resolved = GoldenDatasetLoader.find_regression_fixture(skill_name)
        dataset = GoldenDatasetLoader.load(dataset_path_resolved)
        console.print(
            f"  Loaded promoted regression fixture: {dataset_path_resolved} "
            f"({len(dataset.all_examples)} examples)"
        )
    elif eval_source == "sessiondb":
        save_path = Path(dataset_path) if dataset_path else Path("datasets") / "skills" / skill_name
        dataset = build_dataset_from_external(
            skill_name=skill_name,
            skill_text=skill["raw"],
            sources=["claude-code", "copilot", "hermes"],
            output_path=save_path,
            model=eval_model,
        )
        if not dataset.all_examples:
            console.print("[red]✗ No relevant examples found from session history[/red]")
            sys.exit(1)
        console.print(f"  Mined {len(dataset.all_examples)} examples from session history")
    elif eval_source == "synthetic":
        builder = SyntheticDatasetBuilder(config)
        dataset = builder.generate(
            artifact_text=skill["raw"],
            artifact_type="skill",
        )
        # Save for reuse
        save_path = Path("datasets") / "skills" / skill_name
        dataset.save(save_path)
        console.print(f"  Generated {len(dataset.all_examples)} synthetic examples")
        console.print(f"  Saved to {save_path}/")
    elif dataset_path:
        dataset = EvalDataset.load(Path(dataset_path))
        console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
    else:
        console.print("[red]✗ Specify --dataset-path or use --eval-source synthetic[/red]")
        sys.exit(1)

    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")
    objective_expansion_metadata: dict[str, Any] = {"enabled": False}
    if expand_objective_examples:
        dataset, objective_expansion_metadata = expand_objective_dataset_examples(dataset)
        console.print(
            "  Objective expansion: "
            f"+{objective_expansion_metadata['added_train_examples']} train / "
            f"+{objective_expansion_metadata['added_val_examples']} val examples; "
            "holdout unchanged"
        )
        console.print(f"  Expanded split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    # ── 3. Validate constraints on baseline ─────────────────────────────
    console.print(f"\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    # Validate the complete SKILL.md, not only the body. The skill-structure
    # constraint intentionally checks YAML frontmatter, so passing only the body
    # creates a false baseline violation for valid skills.
    baseline_constraints = validator.validate_all(skill["raw"], "skill")
    all_pass = True
    for c in baseline_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[yellow]⚠ Baseline skill has constraint violations — proceeding anyway[/yellow]")

    optimization_seed = seed_skill or skill
    if seed_skill:
        console.print(f"\n[bold]Validating seed constraints[/bold]")
        seed_constraints = validator.validate_all(seed_skill["raw"], "skill", baseline_text=skill["raw"])
        seed_pass = True
        for c in seed_constraints:
            icon = "✓" if c.passed else "✗"
            color = "green" if c.passed else "red"
            console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
            if not c.passed:
                seed_pass = False
        if not seed_pass:
            console.print("[red]✗ Seed skill FAILED constraints — refusing seeded optimization[/red]")
            sys.exit(1)

    # ── 4. Set up DSPy + GEPA optimizer ─────────────────────────────────
    console.print(f"\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer: GEPA ({iterations} iterations)")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")

    # Configure DSPy
    lm = make_lm(eval_model, hermes_repo=str(config.hermes_agent_path))
    dspy.configure(lm=lm)

    # Create the optimizer seed skill module. This may be a compressed seed
    # candidate when the active baseline violates hard size constraints.
    baseline_module = SkillModule(optimization_seed["body"])

    # Prepare DSPy examples
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    # ── 5. Run GEPA optimization ────────────────────────────────────────
    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")

    start_time = time.time()

    try:
        optimizer = _build_gepa_optimizer(
            iterations=iterations,
            optimizer_model=optimizer_model,
            hermes_repo=str(config.hermes_agent_path),
        )

        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
            valset=valset,
        )
    except Exception as e:
        # Fall back to MIPROv2 if GEPA isn't available in this DSPy version
        console.print(f"[yellow]GEPA not available ({e}), falling back to MIPROv2[/yellow]")
        optimizer = dspy.MIPROv2(
            metric=skill_fitness_metric,
            auto="light",
        )
        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
        )

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    # ── 6. Extract evolved skill text ───────────────────────────────────
    # Prefer GEPA's best program, but keep a non-empty candidate artifact when
    # the best validation program is the unchanged seed and GEPA proposed other
    # candidates. Such alternates remain No-Go if they scored below the GEPA
    # best; the bundle records that in eval/candidate_selection.json.
    export_module, candidate_selection = _select_non_empty_export_candidate(
        optimized_module,
        baseline_body=optimization_seed["body"],
        quality_examples=[*trainset, *valset],
        content_level_repair=content_level_repair,
    )
    if candidate_selection["strategy"] != "gepa_best":
        console.print(
            f"  [yellow]Candidate export selection: {candidate_selection['strategy']} "
            f"(non-empty candidates: {candidate_selection['non_empty_candidate_count']})[/yellow]"
        )
    evolved_body = export_module.skill_text
    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

    # ── 7. Validate evolved skill ───────────────────────────────────────
    console.print(f"\n[bold]Validating evolved skill[/bold]")
    evolved_constraints = validator.validate_all(
        evolved_full,
        "skill",
        baseline_text=optimization_seed["raw"],
    )
    all_pass = True
    for c in evolved_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[red]✗ Evolved skill FAILED constraints — not deploying[/red]")
        metrics = {
            "skill_name": skill_name,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "iterations": iterations,
            "optimizer_model": optimizer_model,
            "eval_model": eval_model,
            "baseline_score": None,
            "evolved_score": None,
            "improvement": 0.0,
            "baseline_size": len(optimization_seed["body"]),
            "active_baseline_size": len(skill["body"]),
            "seed_skill_path": str(Path(seed_skill_path).expanduser()) if seed_skill_path else None,
            "seed_size": len(seed_skill["body"]) if seed_skill else None,
            "evolved_size": len(evolved_body),
            "train_examples": len(dataset.train),
            "val_examples": len(dataset.val),
            "holdout_examples": 0,
            "holdout_total_examples": len(dataset.holdout),
            "elapsed_seconds": elapsed,
            "constraints_passed": False,
            "objective_expansion": objective_expansion_metadata,
            "content_level_repair_enabled": content_level_repair,
        }
        bundle_dir = write_skill_candidate_bundle(
            skill_name=skill_name,
            baseline_skill=skill["raw"],
            evolved_skill=evolved_full,
            seed_skill=seed_skill["raw"] if seed_skill else None,
            metrics=metrics,
            constraint_results=evolved_constraints,
            candidate_selection=candidate_selection,
            decision_summary="Phase 1 skill candidate failed constraints; no active skill or GitHub PR mutation performed.",
        )
        console.print(f"  Saved failed candidate bundle to {bundle_dir}")
        return

    # ── 8. Evaluate on holdout set ──────────────────────────────────────
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")

    holdout_examples = dataset.to_dspy_examples("holdout")
    if holdout_limit is not None and holdout_limit > 0:
        holdout_examples = holdout_examples[:holdout_limit]
        console.print(f"  Limited holdout evaluation to {len(holdout_examples)} examples")

    baseline_scores = []
    evolved_scores = []
    for ex in holdout_examples:
        # Score baseline
        with dspy.context(lm=lm):
            baseline_pred = baseline_module(task_input=ex.task_input)
            baseline_score = skill_fitness_metric(ex, baseline_pred)
            baseline_scores.append(baseline_score)

            evolved_pred = export_module(task_input=ex.task_input)
            evolved_score = skill_fitness_metric(ex, evolved_pred)
            evolved_scores.append(evolved_score)

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

    # ── 9. Report results ───────────────────────────────────────────────
    table = Table(title="Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    change_color = "green" if improvement > 0 else "red"
    table.add_row(
        "Holdout Score",
        f"{avg_baseline:.3f}",
        f"{avg_evolved:.3f}",
        f"[{change_color}]{improvement:+.3f}[/{change_color}]",
    )
    table.add_row(
        "Skill Size",
        f"{len(optimization_seed['body']):,} chars",
        f"{len(evolved_body):,} chars",
        f"{len(evolved_body) - len(optimization_seed['body']):+,} chars",
    )
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    table.add_row("Iterations", "", str(iterations), "")

    console.print()
    console.print(table)

    # ── 10. Save local candidate bundle ─────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = {
        "skill_name": skill_name,
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "baseline_size": len(optimization_seed["body"]),
        "active_baseline_size": len(skill["body"]),
        "seed_skill_path": str(Path(seed_skill_path).expanduser()) if seed_skill_path else None,
        "seed_size": len(seed_skill["body"]) if seed_skill else None,
        "evolved_size": len(evolved_body),
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(holdout_examples),
        "holdout_total_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
        "constraints_passed": all_pass,
        "objective_expansion": objective_expansion_metadata,
        "content_level_repair_enabled": content_level_repair,
        "candidate_selection": candidate_selection,
    }
    output_dir = write_skill_candidate_bundle(
        skill_name=skill_name,
        baseline_skill=skill["raw"],
        evolved_skill=evolved_full,
        seed_skill=seed_skill["raw"] if seed_skill else None,
        metrics=metrics,
        constraint_results=evolved_constraints,
        candidate_selection=candidate_selection,
    )

    console.print(f"\n  Candidate bundle saved to {output_dir}/")

    bundle_decision = json.loads((output_dir / "decision.json").read_text())
    decision_status = bundle_decision.get("status")
    if decision_status == "PASS_CANDIDATE_ONLY":
        console.print(f"\n[bold green]✓ Evolution produced a passing candidate by {improvement:+.3f} ({improvement/max(0.001, avg_baseline)*100:+.1f}%)[/bold green]")
        console.print(f"  Review the patch: {output_dir}/candidates/candidate.patch")
    elif decision_status == "REGRESSION_NO_GO":
        console.print("\n[yellow]⚠ Non-empty candidate generated but it did not beat GEPA's best validation program — No-Go[/yellow]")
        console.print(f"  Review the patch: {output_dir}/candidates/candidate.patch")
    elif decision_status == "NO_DIFF_NO_GO":
        console.print("\n[yellow]⚠ GEPA did not produce an exportable non-empty skill diff — No-Go[/yellow]")
    else:
        console.print(f"\n[yellow]⚠ Evolution did not produce a passing skill candidate (status: {decision_status})[/yellow]")
        console.print("  Try: more iterations, better eval dataset, or different optimizer model")

@click.command()
@click.option("--skill", required=True, help="Name of the skill to evolve")
@click.option("--iterations", default=10, help="Number of GEPA iterations")
@click.option("--eval-source", default="synthetic", type=click.Choice(["synthetic", "golden", "sessiondb"]),
              help="Source for evaluation dataset")
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL)")
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for GEPA reflections")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model for evaluations")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--seed-skill-path", default=None, help="Existing SKILL.md candidate to use as optimizer seed")
@click.option("--skills-root", default=None, help="Root directory containing user/profile skills")
@click.option("--skill-path", default=None, help="Explicit path to a SKILL.md file to evolve")
@click.option("--holdout-limit", default=None, type=int, help="Limit holdout examples for bounded retry runs")
@click.option("--expand-objective-examples", is_flag=True, help="Expand train/val with rubric-focused objective examples while preserving holdout")
@click.option("--content-level-repair", is_flag=True, help="Convert regressive evidence-only repairs into compact runtime candidate sections when size-safe")
@click.option("--run-tests", is_flag=True, help="Run full pytest suite as constraint gate")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
def main(
    skill,
    iterations,
    eval_source,
    dataset_path,
    optimizer_model,
    eval_model,
    hermes_repo,
    seed_skill_path,
    skills_root,
    skill_path,
    holdout_limit,
    expand_objective_examples,
    content_level_repair,
    run_tests,
    dry_run,
):
    """Evolve a Hermes Agent skill using DSPy + GEPA optimization."""
    evolve(
        skill_name=skill,
        iterations=iterations,
        eval_source=eval_source,
        dataset_path=dataset_path,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        hermes_repo=hermes_repo,
        seed_skill_path=seed_skill_path,
        skills_root=skills_root,
        skill_path=skill_path,
        holdout_limit=holdout_limit,
        expand_objective_examples=expand_objective_examples,
        content_level_repair=content_level_repair,
        run_tests=run_tests,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()
