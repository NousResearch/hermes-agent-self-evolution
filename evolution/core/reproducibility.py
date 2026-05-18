"""Reproducibility manifest helpers for optimization runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import EvalDataset, EvalExample


def _hash_payload(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _hash_examples(examples: list[EvalExample]) -> str:
    normalized = [
        json.dumps(example.to_dict(), sort_keys=True, separators=(",", ":"))
        for example in examples
    ]
    payload = "\n".join(normalized).encode("utf-8")
    return _hash_payload(payload)


def _hash_file(path: Path) -> str:
    return _hash_payload(path.read_bytes())


def _git_head(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(path),
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return "unknown"

    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


@dataclass(frozen=True)
class ReproducibilityManifest:
    """Canonical run metadata for deterministic replay and audit."""

    generated_at: str
    skill_name: str
    skill_path: str
    skill_sha256: str
    evolution_repo_git_sha: str
    hermes_repo_git_sha: str
    optimizer_model: str
    eval_model: str
    judge_model: str
    iterations: int
    dataset_train_sha256: str
    dataset_val_sha256: str
    dataset_holdout_sha256: str
    dataset_counts: dict[str, int]
    baseline_score: float
    evolved_score: float
    improvement: float
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def build_reproducibility_manifest(
    *,
    skill_name: str,
    skill_path: Path,
    dataset: EvalDataset,
    config: EvolutionConfig,
    baseline_score: float,
    evolved_score: float,
    improvement: float,
    elapsed_seconds: float,
    evolution_repo_path: Path,
) -> ReproducibilityManifest:
    """Build the canonical reproducibility manifest for a run."""
    return ReproducibilityManifest(
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        skill_name=skill_name,
        skill_path=str(skill_path),
        skill_sha256=_hash_file(skill_path),
        evolution_repo_git_sha=_git_head(evolution_repo_path),
        hermes_repo_git_sha=_git_head(config.hermes_agent_path),
        optimizer_model=config.optimizer_model,
        eval_model=config.eval_model,
        judge_model=config.judge_model,
        iterations=config.iterations,
        dataset_train_sha256=_hash_examples(dataset.train),
        dataset_val_sha256=_hash_examples(dataset.val),
        dataset_holdout_sha256=_hash_examples(dataset.holdout),
        dataset_counts={
            "train": len(dataset.train),
            "val": len(dataset.val),
            "holdout": len(dataset.holdout),
        },
        baseline_score=baseline_score,
        evolved_score=evolved_score,
        improvement=improvement,
        elapsed_seconds=elapsed_seconds,
    )


def write_manifest(path: Path, manifest: ReproducibilityManifest | dict) -> None:
    """Write manifest to JSON with deterministic key ordering."""
    data = manifest.to_dict() if hasattr(manifest, "to_dict") else dict(manifest)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class ToolReproducibilityManifest:
    """Run metadata for a tool-description optimization run."""

    generated_at: str
    artifact_type: str
    tool_name: str
    tool_source_path: str
    baseline_description_sha256: str
    evolution_repo_git_sha: str
    hermes_repo_git_sha: str
    optimizer_model: str
    eval_model: str
    judge_model: str
    iterations: int
    dataset_train_sha256: str
    dataset_val_sha256: str
    dataset_holdout_sha256: str
    dataset_counts: dict[str, int]
    baseline_score: float
    evolved_score: float
    improvement: float
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PromptSectionReproducibilityManifest:
    """Run metadata for a system-prompt section optimization run."""

    generated_at: str
    artifact_type: str
    section_name: str
    section_source_path: str
    baseline_text_sha256: str
    evolution_repo_git_sha: str
    hermes_repo_git_sha: str
    optimizer_model: str
    eval_model: str
    judge_model: str
    iterations: int
    dataset_train_sha256: str
    dataset_val_sha256: str
    dataset_holdout_sha256: str
    dataset_counts: dict[str, int]
    baseline_score: float
    evolved_score: float
    improvement: float
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def build_prompt_section_reproducibility_manifest(
    *,
    section_name: str,
    section_source_path: Path,
    baseline_text: str,
    dataset: EvalDataset,
    config: EvolutionConfig,
    baseline_score: float,
    evolved_score: float,
    improvement: float,
    elapsed_seconds: float,
    evolution_repo_path: Path,
) -> PromptSectionReproducibilityManifest:
    return PromptSectionReproducibilityManifest(
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        artifact_type="prompt_section",
        section_name=section_name,
        section_source_path=str(section_source_path),
        baseline_text_sha256=_hash_payload((baseline_text or "").encode("utf-8")),
        evolution_repo_git_sha=_git_head(evolution_repo_path),
        hermes_repo_git_sha=_git_head(config.hermes_agent_path),
        optimizer_model=config.optimizer_model,
        eval_model=config.eval_model,
        judge_model=config.judge_model,
        iterations=config.iterations,
        dataset_train_sha256=_hash_examples(dataset.train),
        dataset_val_sha256=_hash_examples(dataset.val),
        dataset_holdout_sha256=_hash_examples(dataset.holdout),
        dataset_counts={
            "train": len(dataset.train),
            "val": len(dataset.val),
            "holdout": len(dataset.holdout),
        },
        baseline_score=baseline_score,
        evolved_score=evolved_score,
        improvement=improvement,
        elapsed_seconds=elapsed_seconds,
    )


def build_tool_reproducibility_manifest(
    *,
    tool_name: str,
    tool_source_path: Path,
    baseline_description: str,
    dataset: EvalDataset,
    config: EvolutionConfig,
    baseline_score: float,
    evolved_score: float,
    improvement: float,
    elapsed_seconds: float,
    evolution_repo_path: Path,
) -> ToolReproducibilityManifest:
    return ToolReproducibilityManifest(
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        artifact_type="tool_description",
        tool_name=tool_name,
        tool_source_path=str(tool_source_path),
        baseline_description_sha256=_hash_payload((baseline_description or "").encode("utf-8")),
        evolution_repo_git_sha=_git_head(evolution_repo_path),
        hermes_repo_git_sha=_git_head(config.hermes_agent_path),
        optimizer_model=config.optimizer_model,
        eval_model=config.eval_model,
        judge_model=config.judge_model,
        iterations=config.iterations,
        dataset_train_sha256=_hash_examples(dataset.train),
        dataset_val_sha256=_hash_examples(dataset.val),
        dataset_holdout_sha256=_hash_examples(dataset.holdout),
        dataset_counts={
            "train": len(dataset.train),
            "val": len(dataset.val),
            "holdout": len(dataset.holdout),
        },
        baseline_score=baseline_score,
        evolved_score=evolved_score,
        improvement=improvement,
        elapsed_seconds=elapsed_seconds,
    )


@dataclass(frozen=True)
class CodeReproducibilityManifest:
    """Run metadata for a Phase 4 code-evolution run."""

    generated_at: str
    artifact_type: str
    tool_name: str
    tool_source_path: str
    baseline_source_sha256: str
    evolution_repo_git_sha: str
    hermes_repo_git_sha: str
    optimizer_model: str
    iterations: int
    engine: str
    best_fitness_score: float
    best_iteration: int

    def to_dict(self) -> dict:
        return asdict(self)


def build_code_reproducibility_manifest(
    *,
    tool_name: str,
    tool_source_path: Path,
    baseline_source: str,
    config: EvolutionConfig,
    best_fitness_score: float,
    best_iteration: int,
    iterations: int,
    engine: str,
    evolution_repo_path: Path,
) -> CodeReproducibilityManifest:
    return CodeReproducibilityManifest(
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        artifact_type="tool_code",
        tool_name=tool_name,
        tool_source_path=str(tool_source_path),
        baseline_source_sha256=_hash_payload((baseline_source or "").encode("utf-8")),
        evolution_repo_git_sha=_git_head(evolution_repo_path),
        hermes_repo_git_sha=_git_head(config.hermes_agent_path),
        optimizer_model=config.optimizer_model,
        iterations=iterations,
        engine=engine,
        best_fitness_score=best_fitness_score,
        best_iteration=best_iteration,
    )
