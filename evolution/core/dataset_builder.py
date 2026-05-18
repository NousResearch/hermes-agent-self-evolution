"""Evaluation dataset generation for hermes-agent-self-evolution.

Sources:
A) Synthetic generation — LLM reads a skill/tool/prompt and generates test cases
B) SessionDB mining — extract real usage patterns and score with LLM-as-judge
C) Golden sets — hand-curated JSONL files
"""

import json
import random
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import dspy

from evolution.core.config import EvolutionConfig


def _extract_json_objects(text: str) -> list[dict]:
    """Scan text for top-level {...} JSON objects, tolerant of surrounding noise
    or truncated arrays. Skips braces inside strings."""
    objects: list[dict] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escape = False
        start = i
        j = i
        while j < n:
            ch = text[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : j + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                objects.append(parsed)
                        except json.JSONDecodeError:
                            pass
                        break
            j += 1
        i = max(j + 1, start + 1)
    return objects


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
            with open(path / f"{split_name}.jsonl", "w", encoding="utf-8") as f:
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
                with open(split_file, encoding="utf-8") as f:
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

    @staticmethod
    def _normalize_case_keys(case: dict) -> dict:
        if not isinstance(case, dict):
            return {}
        input_aliases = (
            "task_input",
            "user_input",
            "input",
            "prompt",
            "query",
            "request",
            "scenario",
            "question",
        )
        expected_aliases = (
            "expected_behavior",
            "expected_output",
            "expected",
            "expected_response",
            "expected_result",
            "rubric",
            "answer",
            "ideal_output",
            "criteria",
        )

        def first_value(keys):
            for k in keys:
                if k in case and case[k] not in (None, ""):
                    return case[k]
            return None

        task_input = first_value(input_aliases)
        expected = first_value(expected_aliases)

        def stringify(value):
            if value is None:
                return ""
            if isinstance(value, str):
                return value.strip()
            try:
                return json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(value)

        normalized = dict(case)
        normalized["task_input"] = stringify(task_input)
        normalized["expected_behavior"] = stringify(expected)
        normalized.setdefault("difficulty", case.get("difficulty", "medium"))
        normalized.setdefault("category", case.get("category", "general"))
        return normalized

    @staticmethod
    def _parse_test_cases(raw_output: str) -> list[dict]:
        text = (raw_output or "").strip()
        if not text:
            return []

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        objects = _extract_json_objects(text)
        if objects:
            return objects

        blocks = re.split(r"(?m)^\s*\d+\.\s+", text)
        if len(blocks) <= 1:
            return []

        cases: list[dict] = []
        for block in blocks[1:]:
            chunk = block.strip()
            if not chunk:
                continue

            title_match = re.match(r"\*\*(.+?)\*\*", chunk)
            title = title_match.group(1).strip() if title_match else ""

            input_match = re.search(
                r"(?im)^\s*[-*]?\s*(?:\*\*)?(?:Input|Test Case|Task Input)(?:\*\*)?\s*:\s*(.+)$",
                chunk,
            )
            expected_match = re.search(
                r"(?im)^\s*[-*]?\s*(?:\*\*)?(?:Expected Output|Expected Behavior)(?:\*\*)?\s*:\s*(.+)$",
                chunk,
            )

            task_input = input_match.group(1) if input_match else title
            expected_behavior = expected_match.group(1) if expected_match else chunk[:400]
            task_input = re.sub(r"^\*\*\s*|\s*\*\*$", "", task_input).strip(" `\"*")
            expected_behavior = re.sub(r"^\*\*\s*|\s*\*\*$", "", expected_behavior).strip(" `\"*")

            if task_input and expected_behavior:
                cases.append(
                    {
                        "task_input": task_input,
                        "expected_behavior": expected_behavior,
                        "difficulty": "medium",
                        "category": "general",
                    }
                )
        return cases

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

        cases_raw = self._parse_test_cases(result.test_cases)
        if not cases_raw:
            raise ValueError(f"Could not parse test cases from LLM output: {result.test_cases[:200]}")

        cases_raw = [self._normalize_case_keys(c) for c in cases_raw]

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
        with open(golden_file, encoding="utf-8") as f:
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
