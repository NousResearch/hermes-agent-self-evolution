#!/usr/bin/env python3
"""Run an optional live Phase 1 skill-evolution smoke test.

This script is intentionally outside the default pytest suite. It calls real
DSPy language models, so it may incur provider cost and needs credentials for
whatever model names you pass.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from evolution.skills.skill_module import find_skill


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SKILL = "demo-skill"
DEFAULT_DATASET = REPO_ROOT / "examples" / "golden-datasets" / DEFAULT_SKILL
DEFAULT_HERMES_REPO = REPO_ROOT / "examples" / "hermes-agent-fixture"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "live-smoke"
DEFAULT_OPTIMIZER_MODEL = "openai/gpt-4.1"
DEFAULT_EVAL_MODEL = "openai/gpt-4.1-mini"


PROVIDER_ENV_HINTS = {
    "openai/": ["OPENAI_API_KEY"],
    "anthropic/": ["ANTHROPIC_API_KEY"],
    "gemini/": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "google/": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "openrouter/": ["OPENROUTER_API_KEY"],
}


def provider_env_names(model_names: list[str]) -> set[str]:
    """Return likely credential env vars for the requested DSPy model names."""
    env_names: set[str] = set()
    for model_name in model_names:
        for prefix, names in PROVIDER_ENV_HINTS.items():
            if model_name.startswith(prefix):
                env_names.update(names)
    return env_names


def missing_provider_env(model_names: list[str]) -> list[str]:
    """Return env groups that appear missing for known provider prefixes."""
    missing: list[str] = []
    for model_name in model_names:
        for prefix, names in PROVIDER_ENV_HINTS.items():
            if model_name.startswith(prefix) and not any(os.environ.get(name) for name in names):
                missing.append(" or ".join(names))
    return sorted(set(missing))


def build_command(args: argparse.Namespace) -> list[str]:
    """Build the underlying evolution CLI command."""
    return [
        sys.executable,
        "-m",
        "evolution.skills.evolve_skill",
        "--skill",
        args.skill,
        "--iterations",
        str(args.iterations),
        "--eval-source",
        "golden",
        "--dataset-path",
        str(args.dataset_path),
        "--hermes-repo",
        str(args.hermes_repo),
        "--output-dir",
        str(args.output_dir),
        "--optimizer-model",
        args.optimizer_model,
        "--eval-model",
        args.eval_model,
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optional live smoke test for Phase 1 skill evolution."
    )
    parser.add_argument("--skill", default=DEFAULT_SKILL)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--hermes-repo", type=Path, default=DEFAULT_HERMES_REPO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--optimizer-model", default=DEFAULT_OPTIMIZER_MODEL)
    parser.add_argument("--eval-model", default=DEFAULT_EVAL_MODEL)
    parser.add_argument(
        "--allow-missing-provider-env",
        action="store_true",
        help="Run even if common provider credential env vars are not detected.",
    )
    parser.add_argument(
        "--print-command-only",
        action="store_true",
        help="Print the underlying command and exit without calling providers.",
    )
    return parser.parse_args(argv)


def validate_inputs(args: argparse.Namespace) -> None:
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    if not args.dataset_path.exists():
        raise SystemExit(f"Dataset path does not exist: {args.dataset_path}")
    if not args.hermes_repo.exists():
        raise SystemExit(f"Hermes repo fixture/path does not exist: {args.hermes_repo}")
    skill_path = find_skill(args.skill, args.hermes_repo)
    if not skill_path:
        raise SystemExit(
            f"Skill '{args.skill}' not found recursively under: "
            f"{args.hermes_repo / 'skills'}"
        )

    missing = missing_provider_env([args.optimizer_model, args.eval_model])
    if missing and not args.allow_missing_provider_env and not args.print_command_only:
        hint = ", ".join(missing)
        raise SystemExit(
            "Provider credentials were not detected for the requested model names. "
            f"Set one of: {hint}. To bypass this check, pass "
            "--allow-missing-provider-env."
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_inputs(args)
    command = build_command(args)

    print("Live skill evolution smoke command:")
    print(shlex.join(command))
    print()
    print("This may call external model providers and incur cost.")

    if args.print_command_only:
        return 0

    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
