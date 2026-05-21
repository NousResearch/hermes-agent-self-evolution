"""L5 Constraint Analyzer — Analyze recent evolution results and propose constraint adjustments.

Constraints are rules that validate artifacts (size limits, growth limits, etc.).
This analyzer looks at recent evolution history to suggest tuning:
- Should size limits be relaxed or tightened?
- Are growth limits causing too many rejections?
- What constraint patterns correlate with successful evolutions?

Usage:
    python -m evolution.artifacts.constraint_analyzer
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import dspy
from rich.console import Console

console = Console()

OUTPUT_DIR = Path("output")


def load_recent_results(days: int = 14) -> List[Dict]:
    """Load all recent evolution metrics."""
    results = []
    cutoff = datetime.now().timestamp() - (days * 86400)

    for artifact_dir in OUTPUT_DIR.glob("*/*"):
        if not artifact_dir.is_dir():
            continue
        mfile = artifact_dir / "metrics.json"
        if mfile.exists() and mfile.stat().st_mtime > cutoff:
            try:
                data = json.loads(mfile.read_text())
                data["_source"] = str(mfile)
                results.append(data)
            except Exception:
                pass

    return results


class ConstraintTuningProposal(dspy.Signature):
    """Analyze evolution history and propose constraint rule adjustments.

    Given metrics from recent self-evolution runs, identify:
    - Constraints that are too strict (causing false rejections of good artifacts)
    - Constraints that are too loose (allowing bad artifacts through)
    - Missing constraints that would have caught problems

    Output specific, actionable constraint adjustments.
    """

    evolution_history = dspy.InputField(desc="Summary of recent evolution runs with pass/fail and scores")
    current_constraints = dspy.InputField(desc="Current constraint rules being used")
    proposals = dspy.OutputField(desc="JSON array of proposals, each with: constraint_name, current_value, proposed_value, reasoning")


def analyze_constraints():
    """Main analysis function."""
    console.print("\n[bold cyan]⚖️ L5 Constraint Analysis[/bold cyan]\n")

    results = load_recent_results(days=21)
    console.print(f"  Loaded {len(results)} recent evolution results")

    if len(results) < 2:
        console.print("  [yellow]Not enough history for analysis. Need ≥2 runs.[/yellow]")
        return

    # Build history summary
    summary = []
    for r in results:
        summary.append({
            "type": r.get("artifact_type", "unknown"),
            "improvement": r.get("improvement", 0),
            "constraints_passed": r.get("constraints_passed", True),
            "baseline_size": r.get("baseline_size", 0),
            "evolved_size": r.get("evolved_size", 0),
            "size_change_pct": round((r.get("evolved_size", 0) - r.get("baseline_size", 0)) / max(1, r.get("baseline_size", 1)) * 100, 1),
        })

    current_constraints = """
Current constraint rules:
- size_limit: Max 15000 chars per artifact
- growth_limit: Max +20% size increase from baseline
- non_empty: Artifact must not be empty
- skill_structure: Must have YAML frontmatter with name + description (skills only)
"""

    # Configure DSPy
    api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    lm_kwargs = {}
    if api_base:
        lm_kwargs["api_base"] = api_base
    if api_key:
        lm_kwargs["api_key"] = api_key
    lm = dspy.LM("openai/glm-5.1", **lm_kwargs)

    with dspy.context(lm=lm):
        gen = dspy.Predict(ConstraintTuningProposal)
        result = gen(
            evolution_history=json.dumps(summary, indent=2),
            current_constraints=current_constraints,
        )

    # Parse proposals
    import json_repair
    try:
        proposals = json.loads(json_repair.repair_json(result.proposals))
    except Exception:
        proposals = []

    if not proposals:
        console.print("  [yellow]No proposals generated.[/yellow]")
        return

    console.print(f"\n[bold]Generated {len(proposals)} constraint proposals:[/bold]\n")
    for i, p in enumerate(proposals, 1):
        name = p.get("constraint_name", "?")
        current = p.get("current_value", "?")
        proposed = p.get("proposed_value", "?")
        reasoning = p.get("reasoning", "")
        console.print(f"  {i}. [bold]{name}[/bold]")
        console.print(f"     Current: {current} → Proposed: {proposed}")
        console.print(f"     Reason: {reasoning}")
        console.print()

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("output") / "constraints" / timestamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "proposals.json").write_text(json.dumps(proposals, indent=2))
    console.print(f"  Saved to {out}/")


if __name__ == "__main__":
    analyze_constraints()
