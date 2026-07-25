"""
Evaluation Dataset Builder for Hermes Agent Self-Evolution.

Builds train/val/holdout datasets from multiple sources:
- Synthetic generation (primary bootstrapping)
- SessionDB mining (real usage, LLM-as-judge scored)
- Hand-curated golden sets (high-value skills)
- Skill-specific auto-evaluation (where applicable)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import dspy


@dataclass
class EvaluationExample:
    """Single evaluation example with task input and expected behavior rubric."""
    task_input: str
    expected_behavior: Dict[str, Any]  # Rubric, not exact text
    metadata: Dict[str, Any]
    source: str  # "synthetic", "sessiondb", "golden", "autoeval"
    skill_name: str


@dataclass 
class EvaluationDataset:
    """Train/validation/holdout split for a skill."""
    skill_name: str
    train: List[EvaluationExample]
    val: List[EvaluationExample]
    holdout: List[EvaluationExample]
    rubric: Dict[str, Any]  # Skill-specific scoring rubric
    
    def to_jsonl(self, output_dir: Path) -> None:
        """Save dataset splits as JSONL files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, examples in [("train", self.train), ("val", self.val), ("holdout", self.holdout)]:
            path = output_dir / f"{self.skill_name}_{split_name}.jsonl"
            with open(path, 'w') as f:
                for ex in examples:
                    f.write(json.dumps(asdict(ex)) + "\n")
        # Save rubric
        rubric_path = output_dir / f"{self.skill_name}_rubric.json"
        with open(rubric_path, 'w') as f:
            json.dump(self.rubric, f, indent=2)
    
    @classmethod
    def from_jsonl(cls, skill_name: str, input_dir: Path) -> "EvaluationDataset":
        """Load dataset from JSONL files."""
        splits = {}
        for split_name in ["train", "val", "holdout"]:
            path = input_dir / f"{skill_name}_{split_name}.jsonl"
            examples = []
            if path.exists():
                with open(path) as f:
                    for line in f:
                        data = json.loads(line)
                        examples.append(EvaluationExample(**data))
            splits[split_name] = examples
        
        rubric_path = input_dir / f"{skill_name}_rubric.json"
        rubric = {}
        if rubric_path.exists():
            with open(rubric_path) as f:
                rubric = json.load(f)
        
        return cls(
            skill_name=skill_name,
            train=splits["train"],
            val=splits["val"], 
            holdout=splits["holdout"],
            rubric=rubric
        )


class DatasetSource(ABC):
    """Abstract base for dataset generation sources."""
    
    @abstractmethod
    def generate(self, skill_name: str, skill_content: str, n_examples: int) -> List[EvaluationExample]:
        """Generate evaluation examples for a skill."""
        pass


class SyntheticDatasetSource(DatasetSource):
    """
    Source A: Synthetic generation using a strong model.
    
    Reads the skill file → understands what it does → generates 15-30 
    realistic (task_input, expected_behavior) pairs.
    Expected_behavior is a rubric, not exact text.
    """
    
    def __init__(self, generator_model: str = "anthropic/claude-opus-4"):
        self.generator_model = generator_model
        self.generator = dspy.Predict(self._generate_signature)
    
    @property
    def _generate_signature(self):
        return dspy.Signature(
            "skill_name, skill_content -> task_input, expected_behavior_rubric",
            instructions="""Generate a realistic evaluation example for the given skill.
            The skill helps the agent perform a specific task. Create a task that would
            require this skill, and a rubric describing what correct behavior looks like.
            
            The rubric should be a JSON object with scoring criteria, not exact expected output.
            Example for github-code-review:
            {
                "criteria": [
                    "Identifies security issues (SQL injection, XSS, etc.)",
                    "Checks for proper error handling",
                    "Verifies tests exist for new functionality",
                    "Code style consistency",
                    "Performance considerations noted"
                ],
                "must_catch": ["SQL injection on line 42", "Missing null check on line 15"],
                "nice_to_have": ["Suggests using parameterized queries", "Notes duplicate code"]
            }"""
        )
    
    def generate(self, skill_name: str, skill_content: str, n_examples: int) -> List[EvaluationExample]:
        examples = []
        for i in range(n_examples):
            # Use DSPy to generate examples
            result = self.generator(
                skill_name=skill_name,
                skill_content=skill_content[:8000]  # Truncate if needed
            )
            try:
                expected = json.loads(result.expected_behavior_rubric)
            except:
                expected = {"criteria": ["Follows skill procedure"], "raw": result.expected_behavior_rubric}
            
            examples.append(EvaluationExample(
                task_input=result.task_input,
                expected_behavior=expected,
                metadata={"generation_index": i},
                source="synthetic",
                skill_name=skill_name
            ))
        return examples


class SessionDBDatasetSource(DatasetSource):
    """
    Source B: SessionDB mining — real usage with LLM-as-judge scoring.
    
    Queries SessionDB for sessions where the skill was loaded.
    Extracts task and agent response.
    Uses LLM-as-judge to score on rubric.
    High-scoring → "good" examples; low-scoring → failure cases for GEPA reflection.
    """
    
    def __init__(self, hermes_agent_repo: Path, judge_model: str = "anthropic/claude-sonnet-4"):
        self.hermes_agent_repo = Path(hermes_agent_repo)
        self.judge_model = judge_model
        self.judge = dspy.Predict(self._judge_signature)
    
    @property
    def _judge_signature(self):
        return dspy.Signature(
            "task, agent_response, skill_content, rubric -> score, reasoning, failures",
            instructions="""Score the agent's response on the skill rubric (0-1).
            Output: score (float), reasoning (str), failures (list of what was missed)."""
        )
    
    def _query_session_db(self, skill_name: str) -> List[Dict[str, Any]]:
        """Query the session database for skill usage."""
        # Import hermes session db
        import sys
        sys.path.insert(0, str(self.hermes_agent_repo))
        from hermes_state import SessionStore
        
        store = SessionStore(self.hermes_agent_repo / "sessions")
        sessions = store.list_sessions()
        
        skill_sessions = []
        for session in sessions:
            # Check if skill was loaded in this session
            for msg in session.messages:
                if skill_name.lower() in msg.content.lower():
                    skill_sessions.append({
                        "session_id": session.id,
                        "messages": session.messages
                    })
                    break
        return skill_sessions
    
    def _extract_task_response_pairs(self, sessions: List[Dict]) -> List[Tuple[str, str]]:
        """Extract (task, response) pairs from sessions."""
        pairs = []
        for sess in sessions:
            msgs = sess["messages"]
            for i, msg in enumerate(msgs):
                if msg.role == "user":
                    # Find next assistant response
                    for j in range(i+1, len(msgs)):
                        if msgs[j].role == "assistant":
                            pairs.append((msg.content, msgs[j].content))
                            break
        return pairs
    
    def generate(self, skill_name: str, skill_content: str, n_examples: int) -> List[EvaluationExample]:
        # Get sessions where skill was used
        sessions = self._query_session_db(skill_name)
        if not sessions:
            return []
        
        pairs = self._extract_task_response_pairs(sessions)
        if not pairs:
            return []
        
        # Use LLM-as-judge to score
        examples = []
        rubric = self._default_rubric(skill_name)
        
        for task, response in pairs[:n_examples * 2]:  # Get extra, filter by score
            result = self.judge(
                task=task,
                agent_response=response,
                skill_content=skill_content[:4000],
                rubric=json.dumps(rubric)
            )
            
            score = float(result.score) if hasattr(result, 'score') else 0.5
            expected = rubric.copy()
            expected["judge_score"] = score
            expected["judge_reasoning"] = getattr(result, 'reasoning', '')
            expected["judge_failures"] = getattr(result, 'failures', [])
            
            examples.append(EvaluationExample(
                task_input=task,
                expected_behavior=expected,
                metadata={"session_source": True, "judge_score": score},
                source="sessiondb",
                skill_name=skill_name
            ))
        
        # Sort by score, take top as "good" examples
        examples.sort(key=lambda x: x.metadata.get("judge_score", 0), reverse=True)
        return examples[:n_examples]
    
    def _default_rubric(self, skill_name: str) -> Dict[str, Any]:
        return {
            "criteria": [
                "Follows the skill's procedure",
                "Produces correct/useful output",
                "Stays within token budget",
                "Handles edge cases appropriately"
            ]
        }


class GoldenDatasetSource(DatasetSource):
    """
    Source C: Hand-curated golden sets (optional, high-value skills).
    
    Manually written test cases with expected outputs.
    Stored as JSONL in ~/.hermes/evolution/datasets/<skill-name>/golden.jsonl
    """
    
    def __init__(self, datasets_dir: Path):
        self.datasets_dir = Path(datasets_dir)
    
    def generate(self, skill_name: str, skill_content: str, n_examples: int) -> List[EvaluationExample]:
        golden_path = self.datasets_dir / skill_name / "golden.jsonl"
        if not golden_path.exists():
            return []
        
        examples = []
        with open(golden_path) as f:
            for line in f:
                data = json.loads(line)
                examples.append(EvaluationExample(
                    task_input=data["task"],
                    expected_behavior=data["expected"],
                    metadata={"golden": True},
                    source="golden",
                    skill_name=skill_name
                ))
                if len(examples) >= n_examples:
                    break
        return examples


class AutoEvalDatasetSource(DatasetSource):
    """
    Source D: Skill-specific auto-evaluation (where applicable).
    
    - systematic-debugging: Plant a bug, run skill, check if tests pass after
    - arxiv: Search for known papers, check if found
    - github-code-review: Create PR with planted issues, check if caught
    """
    
    def __init__(self, hermes_agent_repo: Path):
        self.hermes_agent_repo = Path(hermes_agent_repo)
    
    def generate(self, skill_name: str, skill_content: str, n_examples: int) -> List[EvaluationExample]:
        # Skill-specific generators
        if skill_name == "systematic-debugging":
            return self._generate_debugging_examples(n_examples)
        elif skill_name == "arxiv":
            return self._generate_arxiv_examples(n_examples)
        elif skill_name == "github-code-review":
            return self._generate_code_review_examples(n_examples)
        return []
    
    def _generate_debugging_examples(self, n: int) -> List[EvaluationExample]:
        # Would plant bugs in test projects, run skill, verify fix
        return []
    
    def _generate_arxiv_examples(self, n: int) -> List[EvaluationExample]:
        # Would search for known papers
        return []
    
    def _generate_code_review_examples(self, n: int) -> List[EvaluationExample]:
        # Would create PRs with planted issues
        return []


class EvaluationDatasetBuilder:
    """
    Main dataset builder orchestrating all sources.
    
    Creates train/val/holdout splits from multiple sources.
    Default split: 60% train / 20% val / 20% holdout
    """
    
    def __init__(
        self,
        hermes_agent_repo: Path,
        datasets_dir: Path,
        generator_model: str = "anthropic/claude-opus-4",
        judge_model: str = "anthropic/claude-sonnet-4"
    ):
        self.hermes_agent_repo = Path(hermes_agent_repo)
        self.datasets_dir = Path(datasets_dir)
        
        # Initialize sources
        self.sources = {
            "synthetic": SyntheticDatasetSource(generator_model),
            "sessiondb": SessionDBDatasetSource(hermes_agent_repo, judge_model),
            "golden": GoldenDatasetSource(datasets_dir),
            "autoeval": AutoEvalDatasetSource(hermes_agent_repo),
        }
    
    def build_dataset(
        self,
        skill_name: str,
        skill_content: str,
        n_total: int = 30,
        source_weights: Optional[Dict[str, float]] = None
    ) -> EvaluationDataset:
        """
        Build evaluation dataset for a skill from multiple sources.
        
        Args:
            skill_name: Name of the skill (e.g., "github-code-review")
            skill_content: Full text of the SKILL.md file
            n_total: Total examples desired
            source_weights: Relative weights for each source (default: synthetic heavy)
        
        Returns:
            EvaluationDataset with train/val/holdout splits
        """
        if source_weights is None:
            source_weights = {"synthetic": 0.6, "sessiondb": 0.3, "golden": 0.1, "autoeval": 0.0}
        
        # Normalize weights
        total_weight = sum(source_weights.values())
        source_weights = {k: v/total_weight for k, v in source_weights.items()}
        
        # Generate from each source
        all_examples = []
        for source_name, weight in source_weights.items():
            n_from_source = max(1, int(n_total * weight))
            source = self.sources[source_name]
            try:
                examples = source.generate(skill_name, skill_content, n_from_source)
                all_examples.extend(examples)
            except Exception as e:
                print(f"Warning: Source {source_name} failed: {e}")
        
        # Shuffle and split
        random.shuffle(all_examples)
        all_examples = all_examples[:n_total]
        
        n_train = int(0.6 * len(all_examples))
        n_val = int(0.2 * len(all_examples))
        
        train = all_examples[:n_train]
        val = all_examples[n_train:n_train + n_val]
        holdout = all_examples[n_train + n_val:]
        
        # Default rubric (can be overridden per-skill)
        rubric = self._get_skill_rubric(skill_name)
        
        return EvaluationDataset(
            skill_name=skill_name,
            train=train,
            val=val,
            holdout=holdout,
            rubric=rubric
        )
    
    def _get_skill_rubric(self, skill_name: str) -> Dict[str, Any]:
        """Get skill-specific scoring rubric."""
        rubrics = {
            "github-code-review": {
                "criteria": [
                    "Identifies security vulnerabilities (SQL injection, XSS, path traversal)",
                    "Checks for proper error handling and null checks",
                    "Verifies tests exist for new/changed functionality", 
                    "Code style and consistency",
                    "Performance considerations noted",
                    "Documentation/comments adequacy"
                ],
                "weight_per_criterion": 1.0,
                "must_catch_patterns": ["security", "error handling", "testing"],
                "bonus_criteria": ["Suggests improvements", "Notes duplicate code", "API design feedback"]
            },
            "systematic-debugging": {
                "criteria": [
                    "Forms correct hypothesis before acting",
                    "Uses minimal reproduction steps",
                    "Uses debugging tools effectively (logs, breakpoints, prints)",
                    "Identifies root cause, not just symptoms",
                    "Verifies fix works and doesn't regress",
                    "Documents the fix and reasoning"
                ],
                "weight_per_criterion": 1.0
            },
            "arxiv": {
                "criteria": [
                    "Finds relevant papers for the query",
                    "Extracts key information (method, results, limitations)",
                    "Synthesizes across multiple papers",
                    "Cites papers correctly with links",
                    "Identifies gaps/future work"
                ],
                "weight_per_criterion": 1.0
            }
        }
        return rubrics.get(skill_name, {
            "criteria": [
                "Follows the skill's procedure",
                "Produces correct/useful output",
                "Stays within token budget",
                "Handles edge cases"
            ],
            "weight_per_criterion": 1.0
        })