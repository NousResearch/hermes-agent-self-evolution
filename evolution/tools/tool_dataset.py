"""Synthetic dataset of (task -> correct_tool) triples for tool-selection eval."""

import ast
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import dspy

from evolution.core.config import EvolutionConfig


@dataclass
class ToolSelectionExample:
    task: str
    correct_tool: str
    alt_tools: list = field(default_factory=list)
    difficulty: str = "medium"
    kind: str = "clear"  # clear | confusable | no_tool

    def to_dict(self) -> dict:
        return {
            "task": self.task, "correct_tool": self.correct_tool,
            "alt_tools": self.alt_tools, "difficulty": self.difficulty, "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ToolSelectionExample":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ToolSelectionDataset:
    train: list = field(default_factory=list)
    val: list = field(default_factory=list)
    holdout: list = field(default_factory=list)

    @property
    def all_examples(self) -> list:
        return self.train + self.val + self.holdout

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for split, data in [("train", self.train), ("val", self.val), ("holdout", self.holdout)]:
            with open(path / f"{split}.jsonl", "w", encoding="utf-8") as f:
                for ex in data:
                    f.write(json.dumps(ex.to_dict()) + "\n")

    @classmethod
    def load(cls, path: Path) -> "ToolSelectionDataset":
        path = Path(path)
        ds = cls()
        for split in ["train", "val", "holdout"]:
            f = path / f"{split}.jsonl"
            if f.exists():
                items = [ToolSelectionExample.from_dict(json.loads(line))
                         for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
                setattr(ds, split, items)
        return ds

    def to_dspy_examples(self, split: str = "train") -> list:
        return [
            dspy.Example(task=ex.task, correct_tool=ex.correct_tool,
                         alt_tools=ex.alt_tools, kind=ex.kind).with_inputs("task")
            for ex in getattr(self, split)
        ]


def _parse_triples(raw_text: str) -> list:
    """Parse a JSON (or Python-literal) array of triples from LLM output."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse triples: {raw_text[:200]}")
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(match.group())
            except (ValueError, SyntaxError):
                continue
        raise ValueError(f"Could not parse triples: {raw_text[:200]}")


class ToolDatasetBuilder:
    """Generate tool-selection triples using a strong LLM."""

    class GenerateTriples(dspy.Signature):
        """Generate realistic tool-selection test cases for an agent.

        Given the available tools, produce diverse cases: clear-cut choices,
        confusable near-misses (two plausible tools, one clearly better), and
        no-tool cases (a direct answer needs no tool). For each case give:
        task, correct_tool (or "none"), alt_tools (acceptable alternatives),
        difficulty (easy/medium/hard), kind (clear/confusable/no_tool).
        """
        tools_overview: str = dspy.InputField(desc="The available tools and their descriptions")
        num_cases: int = dspy.InputField(desc="Number of cases to generate")
        triples: str = dspy.OutputField(desc="JSON array of cases")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.generator = dspy.ChainOfThought(self.GenerateTriples)

    def generate(self, all_tools: list, num_cases: Optional[int] = None) -> ToolSelectionDataset:
        n = num_cases or max(20, self.config.eval_dataset_size)
        overview = "\n".join(f"- {name}: {desc}" for name, desc in all_tools)

        lm = dspy.LM(self.config.judge_model)
        with dspy.context(lm=lm):
            result = self.generator(tools_overview=overview, num_cases=n)

        raw = _parse_triples(result.triples)
        examples = [
            ToolSelectionExample(
                task=c.get("task", ""),
                correct_tool=c.get("correct_tool", ""),
                alt_tools=c.get("alt_tools", []) or [],
                difficulty=c.get("difficulty", "medium"),
                kind=c.get("kind", "clear"),
            )
            for c in raw
            if c.get("task") and c.get("correct_tool")
        ]

        random.shuffle(examples)
        total = len(examples)
        n_train = max(1, int(total * self.config.train_ratio))
        n_val = max(1, int(total * self.config.val_ratio))
        return ToolSelectionDataset(
            train=examples[:n_train],
            val=examples[n_train:n_train + n_val],
            holdout=examples[n_train + n_val:],
        )
