"""
Fitness Evaluation for Hermes Agent Self-Evolution.

Implements LLM-as-Judge scoring with skill-specific rubrics.
Used by GEPA to evaluate candidate skill variants.
"""

import json
import dspy
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

from .dataset_builder import EvaluationExample, EvaluationDataset
from typing import Dict, Any
SkillRubric = Dict[str, Any]


@dataclass
class FitnessScore:
    """Result of fitness evaluation."""
    overall_score: float  # 0.0 - 1.0
    criterion_scores: Dict[str, float]
    reasoning: str
    failures: List[str]
    metadata: Dict[str, Any]


class FitnessEvaluator:
    """
    Evaluates agent outputs against skill rubrics using LLM-as-Judge.
    
    Can be configured with different judge models for speed vs quality tradeoff.
    """
    
    def __init__(
        self,
        judge_model: str = "anthropic/claude-sonnet-4",
        rubric: Optional[SkillRubric] = None
    ):
        self.judge_model = judge_model
        self.rubric = rubric or {}
        self.judge = dspy.Predict(self._judge_signature)
    
    @property
    def _judge_signature(self):
        return dspy.Signature(
            "task, agent_response, skill_content, rubric, skill_name -> overall_score, criterion_scores, reasoning, failures",
            instructions="""You are an expert evaluator scoring an AI agent's response against a skill rubric.
            
            Score each criterion 0.0-1.0. The overall_score is the weighted average.
            failures should list specific things the agent missed or did wrong.
            Be strict but fair - the rubric defines what 'good' looks like for this skill."""
        )
    
    def evaluate(
        self,
        example: EvaluationExample,
        agent_response: str,
        skill_content: str
    ) -> FitnessScore:
        """Evaluate a single agent response against the rubric."""
        rubric_json = json.dumps(self.rubric.get(example.skill_name, example.expected_behavior))
        
        result = self.judge(
            task=example.task_input,
            agent_response=agent_response,
            skill_content=skill_content[:6000],
            rubric=rubric_json,
            skill_name=example.skill_name
        )
        
        # Parse results
        try:
            overall = float(getattr(result, 'overall_score', 0.5))
        except:
            overall = 0.5
            
        try:
            criteria = json.loads(getattr(result, 'criterion_scores', '{}'))
        except:
            criteria = {}
        
        reasoning = getattr(result, 'reasoning', '')
        failures = getattr(result, 'failures', [])
        if isinstance(failures, str):
            try:
                failures = json.loads(failures)
            except:
                failures = [failures] if failures else []
        
        return FitnessScore(
            overall_score=overall,
            criterion_scores=criteria,
            reasoning=reasoning,
            failures=failures,
            metadata={
                "skill_name": example.skill_name,
                "example_source": example.source,
                "judge_model": self.judge_model
            }
        )
    
    def evaluate_batch(
        self,
        examples: List[EvaluationExample],
        agent_responses: List[str],
        skill_content: str
    ) -> List[FitnessScore]:
        """Evaluate multiple responses (for batch_runner parallel evaluation)."""
        return [
            self.evaluate(ex, resp, skill_content)
            for ex, resp in zip(examples, agent_responses)
        ]


class SkillFitnessAggregator:
    """
    Aggregates fitness scores across a dataset split.
    Computes mean, std, and pass rates for reporting.
    """
    
    @staticmethod
    def aggregate(scores: List[FitnessScore]) -> Dict[str, Any]:
        if not scores:
            return {"mean": 0.0, "std": 0.0, "pass_rate": 0.0, "n": 0}
        
        overall_scores = [s.overall_score for s in scores]
        mean_score = sum(overall_scores) / len(overall_scores)
        std_score = (sum((x - mean_score)**2 for x in overall_scores) / len(overall_scores))**0.5
        pass_rate = sum(1 for x in overall_scores if x >= 0.7) / len(overall_scores)
        
        # Aggregate criterion scores
        all_criteria = {}
        for s in scores:
            for criterion, score in s.criterion_scores.items():
                if criterion not in all_criteria:
                    all_criteria[criterion] = []
                all_criteria[criterion].append(score)
        
        criterion_means = {
            c: sum(v)/len(v) for c, v in all_criteria.items()
        }
        
        # Common failures
        all_failures = []
        for s in scores:
            all_failures.extend(s.failures)
        from collections import Counter
        failure_counts = Counter(all_failures).most_common(10)
        
        return {
            "mean": mean_score,
            "std": std_score,
            "pass_rate": pass_rate,
            "n": len(scores),
            "criterion_means": criterion_means,
            "top_failures": failure_counts
        }
    
    @staticmethod
    def compare(baseline_scores: List[FitnessScore], evolved_scores: List[FitnessScore]) -> Dict[str, Any]:
        """Compare baseline vs evolved aggregate scores."""
        base_agg = SkillFitnessAggregator.aggregate(baseline_scores)
        evo_agg = SkillFitnessAggregator.aggregate(evolved_scores)
        
        improvement = evo_agg["mean"] - base_agg["mean"]
        pct_improvement = improvement / base_agg["mean"] if base_agg["mean"] > 0 else 0
        
        return {
            "baseline": base_agg,
            "evolved": evo_agg,
            "absolute_improvement": improvement,
            "percent_improvement": pct_improvement,
            "statistically_significant": pct_improvement > 0.1 and evo_agg["pass_rate"] > base_agg["pass_rate"]
        }


def make_fitness_function(
    evaluator: FitnessEvaluator,
    skill_content: str,
    skill_name: str
):
    """
    Creates a DSPy-compatible fitness function for GEPA.
    
    GEPA expects a function: (example, prediction) -> float
    where prediction is the agent's output on that example.
    """
    def fitness(example: EvaluationExample, prediction: str) -> float:
        score = evaluator.evaluate(example, prediction, skill_content)
        return score.overall_score
    
    return fitness