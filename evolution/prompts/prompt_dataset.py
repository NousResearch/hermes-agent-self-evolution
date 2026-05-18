"""Behavioral test-scenario builder for system-prompt section evolution.

Each section type targets a specific behavior. The dataset builder asks
the judge model to generate scenarios that exercise that behavior, with a
rubric describing what a good response should contain (or refuse).
"""

from __future__ import annotations

import random
from typing import Optional

import dspy

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import (
    EvalDataset,
    EvalExample,
    SyntheticDatasetBuilder,
)


SECTION_BEHAVIORAL_BRIEFS: dict[str, str] = {
    "DEFAULT_AGENT_IDENTITY": (
        "Tests whether the agent is helpful, direct, admits uncertainty, and avoids "
        "unnecessary verbosity. Generate scenarios across different request types — "
        "factual questions, ambiguous asks, requests that exceed the agent's knowledge."
    ),
    "MEMORY_GUIDANCE": (
        "Tests whether the agent saves durable user preferences and environment facts "
        "to persistent memory, while NOT saving transient task progress, secrets, or "
        "imperative directives. Include scenarios where saving is warranted and others "
        "where it explicitly should be skipped."
    ),
    "SESSION_SEARCH_GUIDANCE": (
        "Tests whether the agent searches past sessions when the user references prior "
        "context (e.g. 'like last time', 'the script we wrote yesterday'). Include "
        "scenarios that warrant a search and scenarios where searching is unnecessary."
    ),
    "SKILLS_GUIDANCE": (
        "Tests whether the agent saves a new skill after solving a non-trivial workflow "
        "and patches an existing skill when it discovers the skill is outdated. Include "
        "scenarios that should trigger skill creation and others that should not."
    ),
}


def behavioral_brief_for(section_name: str) -> str:
    base = section_name.split(":")[0]
    if base in SECTION_BEHAVIORAL_BRIEFS:
        return SECTION_BEHAVIORAL_BRIEFS[base]
    if base == "PLATFORM_HINTS":
        platform = section_name.split(":", 1)[1] if ":" in section_name else "the target platform"
        return (
            f"Tests whether the agent uses formatting that is appropriate for {platform}. "
            "Include scenarios that require platform-specific behaviors (formatting, "
            "media handling, conciseness) and scenarios that should remain plain text."
        )
    return (
        "Tests whether the agent behaves correctly according to the prompt section. "
        "Generate diverse scenarios covering the intended behavior."
    )


class SyntheticPromptScenarioBuilder:
    class GenerateBehavioralScenarios(dspy.Signature):
        """Generate behavioral scenarios for a system-prompt section.

        Output a JSON array of objects each with: scenario (one-sentence user task),
        expected_behavior (rubric of phrases the ideal response should contain or
        refuse), category (one of: positive, negative).
        Roughly half the scenarios should be 'positive' (the section's behavior
        is appropriate) and half should be 'negative' (the section's behavior is
        explicitly NOT appropriate for that task). Avoid duplicates.
        """

        section_name: str = dspy.InputField(desc="Logical section name, e.g. MEMORY_GUIDANCE")
        section_text: str = dspy.InputField(desc="The current section text")
        behavioral_brief: str = dspy.InputField(desc="Plain-English description of what the section is meant to do")
        num_cases: int = dspy.InputField(desc="How many scenarios to generate")
        test_cases: str = dspy.OutputField(desc="JSON array of scenarios")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.generator = dspy.ChainOfThought(self.GenerateBehavioralScenarios)

    def generate(
        self,
        *,
        section_name: str,
        section_text: str,
        num_cases: Optional[int] = None,
    ) -> EvalDataset:
        n = num_cases if num_cases is not None else self.config.eval_dataset_size
        lm = dspy.LM(self.config.judge_model)
        with dspy.context(lm=lm):
            result = self.generator(
                section_name=section_name,
                section_text=section_text,
                behavioral_brief=behavioral_brief_for(section_name),
                num_cases=n,
            )
        cases_raw = SyntheticDatasetBuilder._parse_test_cases(result.test_cases)
        if not cases_raw:
            raise ValueError(
                f"Could not parse behavioral scenarios for {section_name}: "
                f"{result.test_cases[:200]}",
            )
        cases_raw = [SyntheticDatasetBuilder._normalize_case_keys(c) for c in cases_raw]

        examples: list[EvalExample] = []
        for c in cases_raw:
            task = (c.get("task_input") or "").strip()
            expected = (c.get("expected_behavior") or "").strip()
            if not task or not expected:
                continue
            examples.append(
                EvalExample(
                    task_input=task,
                    expected_behavior=expected,
                    difficulty=c.get("difficulty", "medium"),
                    category=c.get("category", "general"),
                    source="synthetic",
                ),
            )

        if not examples:
            raise ValueError(f"No usable behavioral scenarios produced for {section_name}")

        random.shuffle(examples)
        n_total = len(examples)
        n_train = max(1, int(n_total * self.config.train_ratio))
        n_val = max(1, int(n_total * self.config.val_ratio))
        return EvalDataset(
            train=examples[:n_train],
            val=examples[n_train:n_train + n_val],
            holdout=examples[n_train + n_val:],
        )
