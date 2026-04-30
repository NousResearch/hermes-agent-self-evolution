"""Synthetic dataset builder for tool description evolution.

For tool descriptions, fitness is contrastive: the description should
attract tasks where this tool fits (positive examples) and repel tasks
where it does not (negative examples). The LLM generates both.

Positive example: a realistic user task where this tool IS the right pick.
Negative example: a realistic task where this tool is NOT the right pick
  (often because a different tool — file system vs. web fetch vs. shell —
  would be more appropriate).
"""

from __future__ import annotations

import json
import random
from typing import Optional

import dspy

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import (
    EvalDataset,
    EvalExample,
    _try_parse_json_list,
)


# Use the existing `category` field on EvalExample to encode polarity:
#   category="positive" → agent should invoke this tool
#   category="negative" → agent should NOT invoke this tool
_POSITIVE = "positive"
_NEGATIVE = "negative"


class GenerateToolEvalCases(dspy.Signature):
    """Generate contrastive evaluation cases for a tool description.

    Given the tool's name and description, produce realistic user tasks
    split between two classes:
      * positive — tasks where this tool is the appropriate choice
      * negative — tasks where this tool is NOT the right choice (a
        different tool or no tool at all would be better)

    Output a JSON array of objects, each with keys:
      task_input        — the user's task as natural-language text
      polarity          — "positive" or "negative"
      expected_behavior — one short sentence describing what a correct
                          tool-selection decision looks like (e.g. "should
                          pick this tool", "should pick a different tool")
      difficulty        — "easy", "medium", or "hard"
    """

    tool_name: str = dspy.InputField(desc="Name of the tool")
    tool_description: str = dspy.InputField(desc="Current description of the tool")
    num_cases: int = dspy.InputField(desc="Total number of cases (positive + negative)")
    test_cases: str = dspy.OutputField(
        desc="JSON array of cases with task_input, polarity, expected_behavior, difficulty"
    )


class ToolDatasetBuilder:
    """Generate a contrastive (task, polarity) eval dataset for a tool."""

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.generator = dspy.ChainOfThought(GenerateToolEvalCases)

    def generate(
        self,
        tool_name: str,
        description: str,
        num_cases: Optional[int] = None,
    ) -> EvalDataset:
        """Generate a balanced positive/negative eval dataset for a tool."""
        n = num_cases or self.config.eval_dataset_size

        lm = self.config.make_lm(self.config.judge_model)
        with dspy.context(lm=lm):
            result = self.generator(
                tool_name=tool_name,
                tool_description=description,
                num_cases=n,
            )

        cases_raw = _try_parse_json_list(result.test_cases)
        if cases_raw is None:
            raise ValueError(
                f"Could not parse tool eval cases from LLM output: "
                f"{result.test_cases[:500]}"
            )

        examples = []
        for c in cases_raw:
            task_input = c.get("task_input", "").strip()
            if not task_input:
                continue
            polarity = (c.get("polarity") or "").strip().lower()
            if polarity not in (_POSITIVE, _NEGATIVE):
                # Skip cases the LLM did not classify cleanly
                continue
            expected = c.get("expected_behavior", "").strip()
            if not expected:
                expected = (
                    "should pick this tool" if polarity == _POSITIVE
                    else "should NOT pick this tool"
                )
            examples.append(
                EvalExample(
                    task_input=task_input,
                    expected_behavior=expected,
                    difficulty=c.get("difficulty", "medium"),
                    category=polarity,
                    source="synthetic",
                )
            )

        # Reject datasets with extreme polarity skew — fitness is
        # uninformative if all examples are positives.
        positives = sum(1 for e in examples if e.category == _POSITIVE)
        negatives = sum(1 for e in examples if e.category == _NEGATIVE)
        if positives == 0 or negatives == 0:
            raise ValueError(
                f"Dataset must contain BOTH positive ({positives}) and negative "
                f"({negatives}) examples for contrastive evaluation"
            )

        random.shuffle(examples)
        n_total = len(examples)
        n_train = max(1, int(n_total * self.config.train_ratio))
        n_val = max(1, int(n_total * self.config.val_ratio))

        return EvalDataset(
            train=examples[:n_train],
            val=examples[n_train:n_train + n_val],
            holdout=examples[n_train + n_val:],
        )


def is_positive(example: EvalExample) -> bool:
    """Polarity helper — readable in fitness code."""
    return example.category == _POSITIVE


def to_dspy_examples_with_polarity(dataset: EvalDataset, split: str = "train") -> list[dspy.Example]:
    """Build DSPy examples that carry the polarity (category) field.

    The default ``EvalDataset.to_dspy_examples`` only forwards
    ``task_input`` and ``expected_behavior``. The tool fitness metric needs
    ``category`` to know whether the example is positive or negative, so
    this helper preserves it.
    """
    data = getattr(dataset, split)
    return [
        dspy.Example(
            task_input=ex.task_input,
            expected_behavior=ex.expected_behavior,
            category=ex.category,
        ).with_inputs("task_input")
        for ex in data
    ]
