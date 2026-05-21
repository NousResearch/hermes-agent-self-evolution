"""L4 Memory Reflection — Analyze recent sessions and evolution logs to suggest memory updates.

This is NOT traditional memory compaction. Instead, it:
1. Reads recent evolution logs and metrics
2. Identifies patterns in what works/doesn't work
3. Proposes memory entries (user preferences, system insights)
4. Outputs suggestions that can be manually reviewed or auto-applied

Usage:
    python -m evolution.artifacts.memory_reflector
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
LOG_DIR = Path("logs")


def load_recent_metrics(days: int = 7) -> List[Dict]:
    """Load metrics.json files from recent evolution runs."""
    metrics = []
    cutoff = datetime.now().timestamp() - (days * 86400)

    for artifact_dir in OUTPUT_DIR.glob("*/*"):
        if not artifact_dir.is_dir():
            continue
        mfile = artifact_dir / "metrics.json"
        if mfile.exists() and mfile.stat().st_mtime > cutoff:
            try:
                data = json.loads(mfile.read_text())
                data["_source"] = str(mfile)
                metrics.append(data)
            except Exception:
                pass

    return metrics


class MemoryInsightGenerator(dspy.Signature):
    """Analyze evolution metrics and generate memory insights.

    Given a history of self-evolution runs, identify patterns:
    - What consistently improves scores?
    - What causes failures or regressions?
    - What user preferences are emerging?
    - What system configuration works best?

    Output 3-5 concise memory entries that should be saved.
    """

    metrics_history = dspy.InputField(desc="JSON array of recent evolution metrics")
    insights = dspy.OutputField(desc="JSON array of memory insights, each with: category, content, confidence (high/medium/low)")


def reflect_and_propose_memory():
    """Main reflection function."""
    console.print("\n[bold cyan]🧠 L4 Memory Reflection[/bold cyan]\n")

    metrics = load_recent_metrics(days=14)
    console.print(f"  Loaded {len(metrics)} recent evolution metrics")

    if len(metrics) < 2:
        console.print("  [yellow]Not enough history for meaningful reflection. Need ≥2 runs.[/yellow]")
        return

    # Sort by timestamp
    metrics.sort(key=lambda x: x.get("timestamp", ""))

    # Summarize for the LLM
    summary = []
    for m in metrics:
        summary.append({
            "type": m.get("artifact_type", "unknown"),
            "timestamp": m.get("timestamp", "?"),
            "improvement": m.get("improvement", 0),
            "baseline": m.get("baseline_score", 0),
            "evolved": m.get("evolved_score", 0),
            "model": m.get("eval_model", "?"),
            "elapsed": m.get("elapsed_seconds", 0),
        })

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
        gen = dspy.Predict(MemoryInsightGenerator)
        result = gen(metrics_history=json.dumps(summary, indent=2))

    # Parse insights
    import json_repair
    try:
        insights = json.loads(json_repair.repair_json(result.insights))
    except Exception:
        insights = []

    if not insights:
        console.print("  [yellow]No insights generated.[/yellow]")
        return

    console.print(f"\n[bold]Generated {len(insights)} memory insights:[/bold]\n")
    for i, ins in enumerate(insights, 1):
        cat = ins.get("category", "general")
        content = ins.get("content", "")
        conf = ins.get("confidence", "medium")
        color = {"high": "green", "medium": "yellow", "low": "red"}.get(conf, "white")
        console.print(f"  {i}. [{color}]{conf.upper()}[/{color}] [{cat}] {content}")

    # Save suggestions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("output") / "memory" / timestamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "insights.json").write_text(json.dumps(insights, indent=2))
    console.print(f"\n  Saved to {out}/")


if __name__ == "__main__":
    reflect_and_propose_memory()
