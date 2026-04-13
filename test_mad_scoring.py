"""Test MAD confidence scoring on holdout evaluation — no optimization, just scoring."""

import sys
sys.path.insert(0, ".")

import dspy
import json
from pathlib import Path
from rich.console import Console

from evolution.core.config import EvolutionConfig
from evolution.core.fitness import LLMJudge, ConfidenceScoredFitness, compute_confidence

console = Console()

# Load dataset
ds_path = Path("datasets/skills/arxiv")
train = [json.loads(l) for l in (ds_path / "train.jsonl").read_text().strip().split("\n") if l.strip()]
holdout = [json.loads(l) for l in (ds_path / "holdout.jsonl").read_text().strip().split("\n") if l.strip()]
val = [json.loads(l) for l in (ds_path / "val.jsonl").read_text().strip().split("\n") if l.strip()]

console.print(f"Dataset: {len(train)} train / {len(val)} val / {len(holdout)} holdout")

# Load skill
skill_text = Path("/Users/nopenotagain/Orchestra Unified/third_party/hermes-agent/skills/research/arxiv/SKILL.md").read_text()

# Configure DSPy with Nous API
config = EvolutionConfig(judge_model="openai/xiaomi/mimo-v2-pro")
lm = dspy.LM(
    "openai/xiaomi/mimo-v2-pro",
    api_key=config.nous_api_key,
    api_base=config.nous_base_url,
)
dspy.configure(lm=lm)

# Create judge and MAD scorer
judge = LLMJudge(config)
mad_judge = ConfidenceScoredFitness(judge, n_trials=3)

console.print(f"\n[bold cyan]Running MAD confidence scoring on {len(holdout)} holdout examples...[/bold cyan]")
console.print(f"  3 trials per example, judge model: xiaomi/mimo-v2-pro\n")

results = []
for i, ex in enumerate(holdout):
    task = ex.get("task_input", "") or ""
    expected = ex.get("expected_behavior", "") or ""
    
    # Simulate "agent output" as the expected behavior (baseline = perfect)
    # This tells us what the judge's variance looks like on a known-good output
    console.print(f"  [{i+1}/{len(holdout)}] Scoring: {task[:80]}...")
    
    try:
        fitness, confidence = mad_judge.score_with_confidence(
            task_input=task,
            expected_behavior=expected,
            agent_output=expected,  # "perfect" output — baseline
            skill_text=skill_text,
        )
        
        results.append({
            "task": task[:100],
            "fitness_composite": fitness.composite,
            "correctness": fitness.correctness,
            "procedure_following": fitness.procedure_following,
            "conciseness": fitness.conciseness,
            "confidence_label": confidence.label,
            "confidence_score": confidence.confidence,
            "delta": confidence.delta,
            "mad": confidence.mad,
            "decision": confidence.decision,
            "feedback": fitness.feedback[:200],
        })
        
        color = {"likely real": "green", "marginal": "yellow", "within noise": "red"}.get(confidence.label, "white")
        console.print(f"    → composite={fitness.composite:.3f} | confidence=[{color}]{confidence.label} ({confidence.confidence:.2f}x)[/{color}] | decision={confidence.decision}")
        
    except Exception as e:
        console.print(f"    → [red]ERROR: {e}[/red]")
        results.append({"task": task[:100], "error": str(e)})

# Summary
console.print(f"\n[bold]=== MAD Confidence Summary ===[/bold]")
from collections import Counter
labels = Counter(r.get("confidence_label", "error") for r in results)
for label, count in sorted(labels.items()):
    color = {"likely real": "green", "marginal": "yellow", "within noise": "red"}.get(label, "white")
    console.print(f"  [{color}]{label}[/{color}]: {count}/{len(results)}")

avg_composite = sum(r.get("fitness_composite", 0) for r in results) / max(1, len(results))
avg_confidence = sum(r.get("confidence_score", 0) for r in results) / max(1, len(results))
console.print(f"\n  Avg composite: {avg_composite:.3f}")
console.print(f"  Avg confidence: {avg_confidence:.2f}x")

# Save
output = Path("output/mad_test_results.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, indent=2))
console.print(f"\n  Saved to {output}")
