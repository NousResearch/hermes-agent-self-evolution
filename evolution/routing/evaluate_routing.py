"""Evaluate skill discovery against a labelled routing dataset.

Usage:
    python -m evolution.routing.evaluate_routing \
        --skills-dir ../hermes-agent/skills --dataset routing.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
import dspy

from evolution.core.routing import (
    RoutingDataset,
    SkillCatalog,
    build_neighbor_map,
    evaluate_routing,
)


class SkillSelector(dspy.Signature):
    """Select the single best skill for a user request, or reject it."""

    task_input: str = dspy.InputField()
    skill_catalog: str = dspy.InputField()
    selected_skill: str = dspy.OutputField(
        desc="Exact skill name from the catalog, or NONE when no skill applies"
    )


def dspy_selector(model: str):
    predictor = dspy.Predict(SkillSelector)
    lm = dspy.LM(model)

    def select(query: str, catalog: SkillCatalog) -> Optional[str]:
        catalog_text = "\n".join(
            f"- {entry.name}: {entry.description}" for entry in catalog.entries
        )
        with dspy.context(lm=lm):
            result = predictor(task_input=query, skill_catalog=catalog_text)
        selected = str(result.selected_skill).strip()
        if selected.upper() == "NONE" or selected not in catalog.names:
            return None
        return selected

    return select


@click.command()
@click.option("--skills-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--dataset", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--model", default="openai/gpt-4.1-mini")
@click.option("--output", type=click.Path(path_type=Path), default=None)
def main(skills_dir: Path, dataset: Path, model: str, output: Optional[Path]) -> None:
    """Measure routing quality for a complete skills catalog."""
    catalog = SkillCatalog.from_skills_dir(skills_dir)
    routing_dataset = RoutingDataset.load(dataset)
    metrics = evaluate_routing(
        routing_dataset,
        catalog,
        dspy_selector(model),
        neighboring_skills=build_neighbor_map(catalog),
    )
    payload = metrics.as_dict()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2))
    click.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

