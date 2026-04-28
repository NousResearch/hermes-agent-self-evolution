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
from pydantic import BaseModel, Field

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


class TestCaseSchema(BaseModel):
    task_input: str
    expected_behavior: str
    difficulty: str
    category: str

class SSoTOutputSchema(BaseModel):
    ssot_random_string: str = Field(..., max_length=16)
    ssot_ascii_math_cot: str
    test_cases: list[TestCaseSchema]

SSoTOutputSchema.model_rebuild()

class SyntheticDatasetBuilder:
    """Generate evaluation datasets using a strong LLM.

    Reads the target artifact (skill file, tool description, etc.)
    and generates realistic (task_input, expected_behavior) pairs.
    """

    class GenerateTestCases(dspy.Signature):
        """Generate realistic evaluation test cases for an agent skill or tool.

        The model MUST follow the SSoT protocol to ensure distribution-faithful diversity.
        CRITICAL: Do not use backslashes (\), quotation marks ("), or newline characters inside the ssot_ascii_math_cot block. Use plain alphanumeric text only to prevent JSON parsing errors.
        """
        artifact_text: str = dspy.InputField(desc="The full text of the skill/tool/prompt being tested")
        artifact_type: str = dspy.InputField(desc="Type: 'skill', 'tool_description', or 'prompt_section'")
        num_cases_per_batch: int = dspy.InputField(desc="Number of test cases to generate in this batch")
        random_seed: str = dspy.InputField(desc="Unique entropy seed. First, generate 16 random characters in ssot_random_string. Then, in ssot_ascii_math_cot, perform a polynomial rolling hash on those characters to map the seed to a specific structural constraint. Finally, output the payload.")
        output: SSoTOutputSchema = dspy.OutputField()

    def __init__(self, config: EvolutionConfig):
        self.config = config
        # Use Predict for Pydantic/Outlines guided decoding
        self.generator = dspy.Predict(self.GenerateTestCases)

    def generate(
        self,
        artifact_text: str,
        artifact_type: str = "skill",
        num_cases: Optional[int] = None,
    ) -> EvalDataset:
        """Generate a full eval dataset using Batched Stochastic Slot Mapping."""

        total_needed = num_cases or self.config.eval_dataset_size
        batch_size = 5  # Items per LLM call
        num_batches = (total_needed // batch_size) + 1
        
        # Ensure at least 100+ parallel requests for vLLM batching efficiency if needed,
        # but here we scale to the requested total_needed.
        # For 'Batched Stochastic Slot Mapping', we generate unique seeds for each call.
        
        lm = dspy.LM(self.config.judge_model, cache=False) # Disable cache for diversity
        
        import uuid
        import asyncio

        def _run_gen(seed: str):
            with dspy.context(lm=lm):
                return self.generator(
                    artifact_text=artifact_text,
                    artifact_type=artifact_type,
                    num_cases_per_batch=batch_size,
                    random_seed=seed
                )

        async def run_batch(seed: str):
            # DSPy Predict is sync, so use to_thread for parallelism with local context
            return await asyncio.to_thread(_run_gen, seed)

        # Generate unique seeds for each request to preserve prefix caching while forcing diversity
        seeds = [str(uuid.uuid4())[:8] for _ in range(num_batches)]
        
        # Flat array of concurrent requests
        import nest_asyncio
        nest_asyncio.apply()
        
        loop = asyncio.get_event_loop()
        tasks = [run_batch(seed) for seed in seeds]
        results = loop.run_until_complete(asyncio.gather(*tasks))

        examples = []
        for result in results:
            # result is a Prediction object with an 'output' attribute
            for c in result.output.test_cases:
                examples.append(
                    EvalExample(
                        task_input=c.task_input,
                        expected_behavior=c.expected_behavior,
                        difficulty=c.difficulty,
                        category=c.category,
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
