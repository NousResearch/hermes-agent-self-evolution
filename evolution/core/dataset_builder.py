"""Evaluation dataset generation for hermes-agent-self-evolution.

Sources:
A) Synthetic generation — LLM reads a skill/tool/prompt and generates test cases
B) SessionDB mining — extract real usage patterns and score with LLM-as-judge
C) Golden sets — hand-curated JSONL files
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import dspy

from evolution.core.config import EvolutionConfig


@dataclass
class EvalExample:
    """A single evaluation example."""
    task_input: str  # What the user asks
    expected_behavior: str  # Rubric — what a good response looks like
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "general"  # Category for stratified eval
    source: str = "synthetic"  # synthetic, sessiondb, golden

    def to_dict(self) -> dict:
        return {
            "task_input": self.task_input,
            "expected_behavior": self.expected_behavior,
            "difficulty": self.difficulty,
            "category": self.category,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalExample":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EvalDataset:
    """Train/val/holdout split of evaluation examples."""
    train: list[EvalExample] = field(default_factory=list)
    val: list[EvalExample] = field(default_factory=list)
    holdout: list[EvalExample] = field(default_factory=list)

    @property
    def all_examples(self) -> list[EvalExample]:
        return self.train + self.val + self.holdout

    def save(self, path: Path):
        """Save dataset splits to JSONL files."""
        path.mkdir(parents=True, exist_ok=True)
        for split_name, split_data in [("train", self.train), ("val", self.val), ("holdout", self.holdout)]:
            with open(path / f"{split_name}.jsonl", "w") as f:
                for ex in split_data:
                    f.write(json.dumps(ex.to_dict()) + "\n")

    @classmethod
    def load(cls, path: Path) -> "EvalDataset":
        """Load dataset splits from JSONL files."""
        dataset = cls()
        for split_name in ["train", "val", "holdout"]:
            split_file = path / f"{split_name}.jsonl"
            if split_file.exists():
                examples = []
                with open(split_file) as f:
                    for line in f:
                        if line.strip():
                            examples.append(EvalExample.from_dict(json.loads(line)))
                setattr(dataset, split_name, examples)
        return dataset

    def to_dspy_examples(self, split: str = "train") -> list[dspy.Example]:
        """Convert a split to DSPy Example objects."""
        data = getattr(self, split)
        return [
            dspy.Example(
                task_input=ex.task_input,
                expected_behavior=ex.expected_behavior,
            ).with_inputs("task_input")
            for ex in data
        ]


class SyntheticDatasetBuilder:
    """Generate evaluation datasets using a strong LLM.

    Reads the target artifact (skill file, tool description, etc.)
    and generates realistic (task_input, expected_behavior) pairs.
    """

    class GenerateTestCases(dspy.Signature):
        """Generate realistic evaluation test cases for an agent skill or tool.

        Given the full text of a skill/tool description, generate diverse test cases
        that would exercise different aspects of the skill. Each test case should include:
        - A realistic task_input (what a user would actually ask)
        - An expected_behavior rubric (what a good response should contain/do, NOT exact text)
        - A difficulty level (easy, medium, hard)
        - A category (what aspect of the skill this tests)

        CRITICAL: Return ONLY a valid JSON array. No markdown, no commentary.
        Format: [{"task_input": "...", "expected_behavior": "...", "difficulty": "easy|medium|hard", "category": "..."}]
        """
        artifact_text: str = dspy.InputField(desc="The full text of the skill/tool/prompt being tested")
        artifact_type: str = dspy.InputField(desc="Type: 'skill', 'tool_description', or 'prompt_section'")
        num_cases: int = dspy.InputField(desc="Number of test cases to generate")
        test_cases: str = dspy.OutputField(desc="JSON array ONLY — no markdown, no commentary")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.generator = dspy.Predict(self.GenerateTestCases)

    def _extract_json(self, text: str) -> list:
        """Robust JSON extraction from LLM output."""
        import re

        # Try direct parse first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON array in the text
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Try to extract individual JSON objects
        objects = []
        for m in re.finditer(r'\{[^{}]*\}', text, re.DOTALL):
            try:
                obj = json.loads(m.group())
                if 'task_input' in obj and 'expected_behavior' in obj:
                    objects.append(obj)
            except json.JSONDecodeError:
                continue
        if objects:
            return objects

        # Try line-by-line parsing (for "key: value" format)
        return self._parse_prose(text)

    def _parse_prose(self, text: str) -> list:
        """Fallback: parse prose test cases into structured format."""
        import re
        cases = []
        # Match patterns like "Test Case 1: Input: ... Expected: ..."
        pattern = r'(?:Test\s*Case\s*\d+:?\s*)?(?:Input|Task):\s*(.+?)(?:Expected(?:\s*Output|\s*Behavior)?|Rubric):\s*(.+?)(?=(?:Test\s*Case\s*\d+:?\s*)?(?:Input|Task):|$)'
        for m in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
            task = m.group(1).strip()
            expected = m.group(2).strip()
            if task and expected:
                cases.append({
                    "task_input": task,
                    "expected_behavior": expected,
                    "difficulty": "medium",
                    "category": "general",
                })
        return cases

    def generate(
        self,
        artifact_text: str,
        artifact_type: str = "skill",
        num_cases: Optional[int] = None,
    ) -> EvalDataset:
        """Generate a full eval dataset with train/val/holdout splits."""
        n = num_cases or self.config.eval_dataset_size

        # Configure DSPy to use the judge model for generation (with proxy support)
        import os as _os_db
        _proxy = _os_db.getenv("HERMES_PROXY_URL")
        _pkey = _os_db.getenv("HERMES_PROXY_KEY", "fake")
        if _proxy:
            lm = dspy.LM(self.config.judge_model, api_base=_proxy, api_key=_pkey, custom_llm_provider="openai")
        else:
            lm = dspy.LM(self.config.judge_model)

        cases_raw = []
        with dspy.context(lm=lm):
            # Try batch generation first
            result = self.generator(
                artifact_text=artifact_text,
                artifact_type=artifact_type,
                num_cases=n,
            )
            cases_raw = self._extract_json(result.test_cases)

        # If batch failed, try generating one at a time
        if not cases_raw:
            print(f"  Batch generation failed, trying single-case generation...")
            with dspy.context(lm=lm):
                for i in range(n):
                    try:
                        result = self.generator(
                            artifact_text=artifact_text,
                            artifact_type=artifact_type,
                            num_cases=1,
                        )
                        single = self._extract_json(result.test_cases)
                        if single:
                            cases_raw.extend(single)
                    except Exception:
                        continue
                    if len(cases_raw) >= n:
                        break

        if not cases_raw:
            raise ValueError(f"Could not generate any test cases from LLM output")

        examples = [
            EvalExample(
                task_input=c.get("task_input", ""),
                expected_behavior=c.get("expected_behavior", ""),
                difficulty=c.get("difficulty", "medium"),
                category=c.get("category", "general"),
                source="synthetic",
            )
            for c in cases_raw
            if c.get("task_input") and c.get("expected_behavior")
        ]

        # Shuffle and truncate to exactly eval_dataset_size
        random.shuffle(examples)
        examples = examples[:n]
        n_total = len(examples)
        n_train = max(1, int(n_total * self.config.train_ratio))
        n_val = max(1, int(n_total * self.config.val_ratio))

        return EvalDataset(
            train=examples[:n_train],
            val=examples[n_train:n_train + n_val],
            holdout=examples[n_train + n_val:],
        )


class GoldenDatasetLoader:
    """Load hand-curated evaluation datasets from JSONL files."""

    @staticmethod
    def load(path: Path) -> EvalDataset:
        """Load a golden dataset. If no splits exist, auto-split the single file."""
        if (path / "train.jsonl").exists():
            return EvalDataset.load(path)

        # Single file — auto-split
        golden_file = path if path.suffix == ".jsonl" else path / "golden.jsonl"
        if not golden_file.exists():
            raise FileNotFoundError(f"No golden dataset found at {golden_file}")

        examples = []
        with open(golden_file) as f:
            for line in f:
                if line.strip():
                    examples.append(EvalExample.from_dict(json.loads(line)))

        random.shuffle(examples)
        n = len(examples)
        n_train = max(1, int(n * 0.5))
        n_val = max(1, int(n * 0.25))

        return EvalDataset(
            train=examples[:n_train],
            val=examples[n_train:n_train + n_val],
            holdout=examples[n_train + n_val:],
        )
