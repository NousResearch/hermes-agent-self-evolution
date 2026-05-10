"""Generate a single large holdout pool for Study A.

Study A picks the (eval_dataset_size, holdout_ratio) pair that minimizes
the bootstrap CI half-width. To support that without re-running GEPA,
we generate one large pool per skill and let the analysis script
subsample at N ∈ {50, 100, 150, 250, 400}.

Usage:
    uv run python scripts/generate_large_holdout.py \\
        --skill nano-pdf --n 400 --seed 42

Writes to `datasets/skills/<skill>/holdout_n<N>_seed<seed>/holdout.jsonl`.
The split JSONL files (train/val) are also written but empty — the
generated examples all land in `holdout` because we set the split ratios
to 0/0/1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import SyntheticDatasetBuilder
from evolution.core.skill_sources import discover_skill_sources
from evolution.skills.skill_module import find_skill, load_skill


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--skill", required=True, help="Skill name (e.g. nano-pdf)")
    parser.add_argument("--n", type=int, default=400, help="Pool size (default 400)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    parser.add_argument(
        "--judge-model",
        default="openai/gpt-4.1-mini",
        help="LLM used by the synthetic generator (default: gpt-4.1-mini, "
        "matches the eval model used in actual runs)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override default datasets/skills/<skill>/holdout_n<N>_seed<seed>",
    )
    args = parser.parse_args()

    config = EvolutionConfig(
        eval_dataset_size=args.n,
        train_ratio=0.0,
        val_ratio=0.0,
        holdout_ratio=1.0,
        seed=args.seed,
        judge_model=args.judge_model,
    )

    skill_path = find_skill(args.skill, config.skill_sources)
    if not skill_path:
        searched = ", ".join(s.name for s in config.skill_sources) or "(none)"
        print(f"✗ Skill '{args.skill}' not found across sources: {searched}", file=sys.stderr)
        sys.exit(1)
    skill = load_skill(skill_path)
    print(f"  Loaded: {skill_path}  ({len(skill['raw']):,} chars)")

    builder = SyntheticDatasetBuilder(config)
    print(f"  Generating {args.n} cases via {args.judge_model}…")
    dataset = builder.generate(skill["raw"], artifact_type="skill", num_cases=args.n)

    out_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path("datasets") / "skills" / args.skill / f"holdout_n{args.n}_seed{args.seed}"
    )
    dataset.save(out_root)
    print(f"  Wrote {len(dataset.holdout)} examples to {out_root}/holdout.jsonl")
    if len(dataset.holdout) < args.n:
        print(
            f"  ⚠ {args.n - len(dataset.holdout)} examples dropped (LLM output filtered "
            "by missing task_input/expected_behavior)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
