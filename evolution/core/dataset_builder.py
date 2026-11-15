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


VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def _extract_cases_list(parsed: object) -> Optional[list]:
    """Unwrap a parsed JSON payload into a list of raw test-case entries."""
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        if "task_input" in parsed or "expected_behavior" in parsed:
            return [parsed]
        for key in ("test_cases", "cases", "examples", "items", "data"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
    return None


def _parse_test_cases(raw_text: str) -> list[dict]:
    """Extract test-case dicts from raw JSON, fenced blocks, or wrapper objects."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("Could not parse test cases from empty LLM output")

    try:
        cases = _extract_cases_list(json.loads(raw_text))
        if cases is not None:
            dict_cases = [c for c in cases if isinstance(c, dict)]
            if dict_cases or cases == []:
                return dict_cases
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    saw_empty_list = False
    for idx, ch in enumerate(raw_text):
        if ch not in "[{":
            continue
        try:
            candidate, _ = decoder.raw_decode(raw_text, idx)
        except json.JSONDecodeError:
            continue
        cases = _extract_cases_list(candidate)
        if cases is None:
            continue
        dict_cases = [c for c in cases if isinstance(c, dict)]
        if dict_cases:
            return dict_cases
        if cases == []:
            saw_empty_list = True

    if saw_empty_list:
        return []

    raise ValueError(f"Could not parse test cases from LLM output: {raw_text[:200]}")


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
        """
        artifact_text: str = dspy.InputField(desc="The full text of the skill/tool/prompt being tested")
        artifact_type: str = dspy.InputField(desc="Type: 'skill', 'tool_description', or 'prompt_section'")
        num_cases: int = dspy.InputField(desc="Number of test cases to generate")
        test_cases: str = dspy.OutputField(desc="JSON array of test cases, each with: task_input, expected_behavior, difficulty, category")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.generator = dspy.ChainOfThought(self.GenerateTestCases)

    def generate(
        self,
        artifact_text: str,
        artifact_type: str = "skill",
        num_cases: Optional[int] = None,
    ) -> EvalDataset:
        """Generate a full eval dataset with train/val/holdout splits."""

        n = num_cases or self.config.eval_dataset_size

        # Configure DSPy to use the judge model for generation
        lm = dspy.LM(self.config.judge_model)

        with dspy.context(lm=lm):
            result = self.generator(
                artifact_text=artifact_text,
                artifact_type=artifact_type,
                num_cases=n,
            )

        # Parse the generated test cases
        cases_raw = _parse_test_cases(result.test_cases)

        examples = []
        for c in cases_raw:
            task_input = c.get("task_input")
            expected_behavior = c.get("expected_behavior")
            if not isinstance(task_input, str) or not task_input.strip():
                continue
            if not isinstance(expected_behavior, str) or not expected_behavior.strip():
                continue

            difficulty = c.get("difficulty")
            difficulty_norm = difficulty.strip().lower() if isinstance(difficulty, str) and difficulty.strip() else "medium"
            if difficulty_norm not in VALID_DIFFICULTIES:
                difficulty_norm = "medium"

            category = c.get("category")
            category_norm = category.strip() if isinstance(category, str) and category.strip() else "general"

            examples.append(
                EvalExample(
                    task_input=task_input.strip(),
                    expected_behavior=expected_behavior.strip(),
                    difficulty=difficulty_norm,
                    category=category_norm,
                    source="synthetic",
                )
            )

        # Shuffle and split
        random.shuffle(examples)
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
