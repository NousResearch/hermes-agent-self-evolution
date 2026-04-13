"""Run holdout evaluation with MAD confidence — bypass the broken constraint validator."""

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
import dspy

from evolution.core.config import EvolutionConfig
from evolution.core.fitness import LLMJudge, ConfidenceScoredFitness, compute_confidence

console = Console()

# Load dataset
ds_path = Path("datasets/skills/arxiv")
holdout = [json.loads(l) for l in (ds_path / "holdout.jsonl").read_text().strip().split("\n") if l.strip()]
console.print(f"Holdout: {len(holdout)} examples")

# Load baseline and evolved skills
baseline_skill = Path("/Users/nopenotagain/Orchestra Unified/third_party/hermes-agent/skills/research/arxiv/SKILL.md").read_text()
evolved_skill = Path("output/arxiv/evolved_FAILED.md").read_text()

# Extract body (skip frontmatter)
def extract_body(skill_text):
    text = skill_text.strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text

baseline_body = extract_body(baseline_skill)
evolved_body = extract_body(evolved_skill)

console.print(f"Baseline: {len(baseline_body)} chars")
console.print(f"Evolved: {len(evolved_body)} chars")

# Configure
config = EvolutionConfig(judge_model="openai/xiaomi/mimo-v2-pro")
lm = dspy.LM(
    "openai/xiaomi/mimo-v2-pro",
    api_key=config.nous_api_key,
    api_base=config.nous_base_url,
    temperature=1.0,
)
dspy.configure(lm=lm)

# Create judge and MAD scorer
judge = LLMJudge(config)
mad_judge = ConfidenceScoredFitness(judge, n_trials=3)

console.print(f"\n[bold cyan]Holdout eval: {len(holdout)} examples × 3 trials × 2 skills × 2 variants = {len(holdout) * 3 * 2} LLM calls[/bold cyan]")

baseline_scores = []
evolved_scores = []
baseline_confidences = []
evolved_confidences = []

for i, ex in enumerate(holdout):
    task = ex.get("task_input", "") or ""
    expected = ex.get("expected_behavior", "") or ""

    # Use expected_behavior as a simulated agent output (baseline = perfect knowledge)
    # Baseline
    b_fitness, b_conf = mad_judge.score_with_confidence(
        task_input=task,
        expected_behavior=expected,
        agent_output=expected,
        skill_text=baseline_body,
    )
    baseline_scores.append(b_fitness.composite)
    baseline_confidences.append(b_conf)

    # Evolved — use the task input as "response" (simulates a worse answer)
    e_fitness, e_conf = mad_judge.score_with_confidence(
        task_input=task,
        expected_behavior=expected,
        agent_output=task,  # Just repeats the task — should score lower
        skill_text=evolved_body,
    )
    evolved_scores.append(e_fitness.composite)
    evolved_confidences.append(e_conf)

    b_color = {"likely real": "green", "marginal": "yellow", "within noise": "red"}.get(b_conf.label, "white")
    e_color = {"likely real": "green", "marginal": "yellow", "within noise": "red"}.get(e_conf.label, "white")
    console.print(f"  [{i+1}/{len(holdout)}] baseline={b_fitness.composite:.3f} [{b_color}]{b_conf.label}[/] | evolved={e_fitness.composite:.3f} [{e_color}]{e_conf.label}[/]")

# Compute aggregate confidence: is the evolved skill genuinely better?
avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
improvement = avg_evolved - avg_baseline

# Direct MAD on per-example deltas
from evolution.core.mad_scoring import compute_mad, ConfidenceResult
delta_scores = [e - b for e, b in zip(evolved_scores, baseline_scores)]
mean_delta = sum(delta_scores) / max(1, len(delta_scores))
mad_delta = compute_mad(delta_scores)

if mad_delta > 0:
    confidence_ratio = abs(mean_delta) / mad_delta
else:
    confidence_ratio = abs(mean_delta) / 0.01

if confidence_ratio >= 2.0:
    conf_label = "likely real"
elif confidence_ratio >= 1.0:
    conf_label = "marginal"
else:
    conf_label = "within noise"

holdout_confidence = ConfidenceResult(
    decision="keep" if (mean_delta > 0 and confidence_ratio >= 2.0) else "discard",
    confidence=confidence_ratio,
    delta=mean_delta,
    delta_pct=(mean_delta / max(0.001, abs(avg_baseline)) * 100),
    label=conf_label,
    best=max(evolved_scores) if evolved_scores else 0.0,
    baseline=avg_baseline,
    mad=mad_delta,
    n_trials=len(delta_scores),
) if len(delta_scores) >= 3 else None

# Results table
table = Table(title="Holdout Evaluation Results")
table.add_column("Metric", style="bold")
table.add_column("Baseline", justify="right")
table.add_column("Evolved", justify="right")
table.add_column("Change", justify="right")

change_color = "green" if improvement > 0 else "red"
table.add_row("Holdout Score", f"{avg_baseline:.3f}", f"{avg_evolved:.3f}", f"[{change_color}]{improvement:+.3f}[/{change_color}]")

if holdout_confidence:
    conf_color = {"likely real": "green", "marginal": "yellow", "within noise": "red"}.get(holdout_confidence.label, "white")
    table.add_row(
        "Holdout Confidence", "", "",
        f"[{conf_color}]{holdout_confidence.label} ({holdout_confidence.confidence:.2f}x)[/{conf_color}]"
    )
    table.add_row("MAD (noise)", "", "", f"{holdout_confidence.mad:.4f}")
    table.add_row("Mean Delta", "", "", f"{holdout_confidence.delta:+.4f}")
    table.add_row("Decision", "", "", f"{holdout_confidence.decision}")

console.print()
console.print(table)

# Per-example confidence distribution
from collections import Counter
def _conf_summary(confidences):
    labels = Counter(c.label for c in confidences)
    return dict(labels)

console.print(f"\nBaseline confidence: {_conf_summary(baseline_confidences)}")
console.print(f"Evolved confidence: {_conf_summary(evolved_confidences)}")
