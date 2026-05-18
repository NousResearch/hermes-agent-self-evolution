"""DSPy module wrapping a system-prompt section as the optimizable parameter.

The target section text is stored as the predictor's signature instructions,
so MIPROv2/GEPA mutates the section text directly. Forward asks the agent
a behavioral question and the metric compares the response against a rubric.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

import dspy

from evolution.tools.tool_module import clean_evolved_description


class PromptSectionModule(dspy.Module):
    """Wrap one system-prompt section as the predictor's signature instructions."""

    class _BehavioralSignature(dspy.Signature):
        """Placeholder; instructions are overridden with the section text."""

        scenario: str = dspy.InputField(
            desc="A user task that probes whether the prompt section steers behavior correctly.",
        )
        output: str = dspy.OutputField(
            desc="The agent's response to the scenario, exhibiting the behavior the section is meant to elicit.",
        )

    def __init__(self, section_name: str, text: str):
        super().__init__()
        self.section_name = section_name
        signature = self._BehavioralSignature.with_instructions(text or "")
        self.predictor = dspy.ChainOfThought(signature)

    def _inner_predict(self):
        return getattr(self.predictor, "predict", self.predictor)

    @property
    def section_text(self) -> str:
        sig = self._inner_predict().signature
        return getattr(sig, "instructions", "") or ""

    @section_text.setter
    def section_text(self, value: str) -> None:
        inner = self._inner_predict()
        inner.signature = inner.signature.with_instructions(value or "")

    def forward(self, task_input: str = None, scenario: str = None, **_: object) -> dspy.Prediction:
        text = scenario if scenario is not None else task_input
        result = self.predictor(scenario=text or "")
        return dspy.Prediction(output=result.output)


def clean_evolved_section(text: str, max_chars: int | None = None) -> str:
    """Reuse the tool-description cleaner to strip optimizer-inlined examples
    and enforce a max-char budget on the evolved section."""
    return clean_evolved_description(text, max_chars=max_chars)


def behavioral_fitness_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Keyword-overlap proxy mirroring skill_fitness_metric.

    Phase 3 evaluation uses LLM-as-judge under the hood for full runs, but a
    fast metric is needed inside the optimizer's inner loop. We score the
    fraction of rubric keywords that appear in the agent's output.
    """
    agent_output = (getattr(prediction, "output", "") or "").lower()
    expected = (getattr(example, "expected_behavior", "") or "").lower()
    if not agent_output.strip():
        return 0.0
    expected_words = {w for w in expected.split() if len(w) > 3}
    if not expected_words:
        return 0.5
    output_words = set(agent_output.split())
    overlap = len(expected_words & output_words) / len(expected_words)
    return 0.3 + 0.7 * overlap


class _BehavioralJudgeSignature(dspy.Signature):
    """Score how well an agent response exhibits the expected behavior.

    Return a single floating point number between 0.0 and 1.0 where:
      - 0.0 = output completely misses or contradicts the expected behavior
      - 0.5 = partial credit; touches on it but lacks key elements
      - 1.0 = output cleanly exhibits the expected behavior
    Do not write anything except the number.
    """

    scenario: str = dspy.InputField(desc="The user task the agent saw")
    expected_behavior: str = dspy.InputField(desc="Rubric of behavior the response should exhibit")
    agent_output: str = dspy.InputField(desc="What the agent produced")
    score: str = dspy.OutputField(desc="A single floating point number between 0.0 and 1.0")


def _parse_judge_score(raw: str) -> float:
    match = re.search(r"[01](?:\.\d+)?|\.\d+", raw or "")
    if not match:
        return 0.0
    try:
        v = float(match.group(0))
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, v))


def make_llm_judge_metric(
    judge_lm,
    *,
    fallback_weight: float = 0.0,
) -> Callable[[dspy.Example, dspy.Prediction, Optional[object]], float]:
    """Build an LLM-as-judge behavioral fitness metric.

    The judge model rates each (scenario, expected_behavior, output) on 0..1.
    When ``fallback_weight`` > 0 a weighted average with keyword overlap is
    returned so the metric is still useful when the judge errors out.
    """

    judge = dspy.Predict(_BehavioralJudgeSignature)

    def _metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
        output = (getattr(prediction, "output", "") or "").strip()
        if not output:
            return 0.0
        try:
            with dspy.context(lm=judge_lm):
                result = judge(
                    scenario=getattr(example, "task_input", "") or "",
                    expected_behavior=getattr(example, "expected_behavior", "") or "",
                    agent_output=output,
                )
            judge_score = _parse_judge_score(getattr(result, "score", ""))
        except Exception:
            judge_score = behavioral_fitness_metric(example, prediction)

        if fallback_weight <= 0:
            return judge_score
        overlap_score = behavioral_fitness_metric(example, prediction)
        return (1.0 - fallback_weight) * judge_score + fallback_weight * overlap_score

    return _metric
