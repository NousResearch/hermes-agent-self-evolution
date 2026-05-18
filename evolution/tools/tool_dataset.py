"""Synthetic dataset builder for tool-selection accuracy evaluation.

Generates pairs of (task, expected_decision) where expected_decision is
"yes" when the target tool is the right choice for that task and "no"
when a different tool (or no tool) would be a better fit. Both classes
are required so the optimizer cannot raise selection accuracy by simply
over-claiming for the target tool.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import dspy

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import (
    EvalDataset,
    EvalExample,
    SyntheticDatasetBuilder,
)


@dataclass
class ToolSelectionCase:
    task_input: str
    label: str  # "yes" or "no"
    rationale: str = ""
    competing_tool: str = ""

    def to_eval_example(self) -> EvalExample:
        return EvalExample(
            task_input=self.task_input,
            expected_behavior=self.label,
            difficulty="medium",
            category="positive" if self.label == "yes" else "negative",
            source="synthetic",
        )


class SyntheticToolSelectionBuilder:
    """Generate positive/negative tool-selection cases via the judge model."""

    class GeneratePositiveCases(dspy.Signature):
        """Generate realistic user requests where this tool is clearly the best choice.

        Output a JSON array of objects with keys: task_input, rationale.
        task_input must be a concrete user request (one sentence). rationale must
        explain in one sentence why this specific tool is the right pick.
        Cover varied difficulty and diverse phrasings; avoid duplicates.
        """

        tool_name: str = dspy.InputField(desc="The tool whose selection is being evaluated")
        tool_description: str = dspy.InputField(desc="The current tool description")
        other_tools_summary: str = dspy.InputField(desc="Brief summaries of the peer tools")
        num_cases: int = dspy.InputField(desc="How many examples to produce")
        test_cases: str = dspy.OutputField(desc="JSON array of cases")

    class GenerateNegativeCases(dspy.Signature):
        """Generate realistic user requests where this tool is NOT the right choice.

        Each generated request should superficially relate to this tool but a
        different tool (or no tool) is actually the right fit. Output a JSON
        array of objects with keys: task_input, rationale, competing_tool.
        competing_tool must be the name of the peer tool that should be used
        instead, taken from the other_tools_summary.
        """

        tool_name: str = dspy.InputField(desc="The tool whose selection is being evaluated")
        tool_description: str = dspy.InputField(desc="The current tool description")
        other_tools_summary: str = dspy.InputField(desc="Brief summaries of the peer tools")
        num_cases: int = dspy.InputField(desc="How many examples to produce")
        test_cases: str = dspy.OutputField(desc="JSON array of cases")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.positive_generator = dspy.ChainOfThought(self.GeneratePositiveCases)
        self.negative_generator = dspy.ChainOfThought(self.GenerateNegativeCases)

    def _generate(self, generator, **inputs) -> list[dict]:
        lm = dspy.LM(self.config.judge_model)
        with dspy.context(lm=lm):
            result = generator(**inputs)
        cases = SyntheticDatasetBuilder._parse_test_cases(result.test_cases)
        return [SyntheticDatasetBuilder._normalize_case_keys(c) for c in cases]

    def generate(
        self,
        *,
        tool_name: str,
        tool_description: str,
        other_tools_summary: str,
        num_positive: Optional[int] = None,
        num_negative: Optional[int] = None,
    ) -> EvalDataset:
        n_pos = num_positive if num_positive is not None else max(8, self.config.eval_dataset_size // 2)
        n_neg = num_negative if num_negative is not None else max(8, self.config.eval_dataset_size // 2)

        positive_raw = self._generate(
            self.positive_generator,
            tool_name=tool_name,
            tool_description=tool_description,
            other_tools_summary=other_tools_summary,
            num_cases=n_pos,
        )
        negative_raw = self._generate(
            self.negative_generator,
            tool_name=tool_name,
            tool_description=tool_description,
            other_tools_summary=other_tools_summary,
            num_cases=n_neg,
        )

        if not positive_raw and not negative_raw:
            raise ValueError("Tool-selection dataset generator returned no parseable cases")

        cases: list[ToolSelectionCase] = []
        for c in positive_raw:
            task = (c.get("task_input") or "").strip()
            if task:
                cases.append(ToolSelectionCase(task_input=task, label="yes",
                                               rationale=(c.get("expected_behavior") or "")))
        for c in negative_raw:
            task = (c.get("task_input") or "").strip()
            if task:
                cases.append(ToolSelectionCase(
                    task_input=task,
                    label="no",
                    rationale=(c.get("expected_behavior") or ""),
                    competing_tool=str(c.get("competing_tool", "") or ""),
                ))

        if not cases:
            raise ValueError("Tool-selection dataset generator produced no usable cases")

        examples = [c.to_eval_example() for c in cases]
        random.shuffle(examples)

        n_total = len(examples)
        n_train = max(1, int(n_total * self.config.train_ratio))
        n_val = max(1, int(n_total * self.config.val_ratio))
        return EvalDataset(
            train=examples[:n_train],
            val=examples[n_train:n_train + n_val],
            holdout=examples[n_train + n_val:],
        )
