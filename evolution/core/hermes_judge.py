"""Hermes-powered LLM judge using hermes chat as the scoring backend.

Instead of calling dspy.LM directly (which requires API key management),
this judge shells out to hermes chat with GPT-5.4 (or any configured model).
Hermes handles auth, routing, retries, and model selection natively.

Usage:
    from evolution.core.hermes_judge import HermesJudge
    judge = HermesJudge(model="gpt-5.4")
    score = judge.score(task_input, expected_behavior, agent_output, skill_text)
"""

import subprocess
import re
from dataclasses import dataclass
from typing import Optional, Literal

from evolution.core.config import EvolutionConfig
from evolution.core.fitness import FitnessScore, _parse_score


class HermesJudge:
    """LLM-as-judge scorer powered by hermes chat subprocess.

    Calls `hermes chat -q "..." -m MODEL -Q` for each scoring request.
    No API key management needed — hermes handles auth natively.
    """

    SCORING_PROMPT = """You are an expert evaluator. Score the agent's response on three dimensions (0.0 to 1.0 each).

TASK: {task_input}

EXPECTED BEHAVIOR: {expected_behavior}

AGENT'S RESPONSE: {agent_output}

SKILL INSTRUCTIONS (what the agent was following):
{skill_text}

Rate each dimension:
1. correctness (0.0-1.0): Did the response correctly address the task?
2. procedure_following (0.0-1.0): Did it follow the expected approach?
3. conciseness (0.0-1.0): Was it appropriately concise?

Also provide brief feedback on what could improve.

Reply EXACTLY in this format (no extra text):
correctness: 0.X
procedure_following: 0.X
conciseness: 0.X
feedback: <one sentence>"""

    def __init__(
        self,
        model: str = "gpt-5.4",
        hermes_bin: str = "hermes",
        timeout: int = 120,
    ):
        self.model = model
        self.hermes_bin = hermes_bin
        self.timeout = timeout

    def score(
        self,
        task_input: str,
        expected_behavior: str,
        agent_output: str,
        skill_text: str,
        artifact_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> FitnessScore:
        """Score an agent output using hermes chat."""

        prompt = self.SCORING_PROMPT.format(
            task_input=task_input,
            expected_behavior=expected_behavior,
            agent_output=agent_output,
            skill_text=skill_text[:2000],  # Truncate to avoid context overflow
        )

        try:
            result = subprocess.run(
                [self.hermes_bin, "chat", "-q", prompt, "-m", self.model, "-Q"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            output = result.stdout.strip()
        except subprocess.TimeoutExpired:
            return FitnessScore(
                correctness=0.0,
                procedure_following=0.0,
                conciseness=0.0,
                feedback="Judge timed out",
            )
        except Exception as e:
            return FitnessScore(
                correctness=0.0,
                procedure_following=0.0,
                conciseness=0.0,
                feedback=f"Judge error: {e}",
            )

        # Parse scores from output — try structured format first, then table format
        correctness = self._extract_score(output, "correctness")
        procedure = self._extract_score(output, "procedure_following")
        conciseness = self._extract_score(output, "conciseness")
        feedback = self._extract_feedback(output)

        # Fallback: parse from table format if structured parsing failed
        if correctness == 0.5 and procedure == 0.5:
            scores = re.findall(r"(\d\.\d+)", output)
            if len(scores) >= 3:
                correctness = _parse_score(scores[0])
                procedure = _parse_score(scores[1])
                conciseness = _parse_score(scores[2])

        # Length penalty
        length_penalty = 0.0
        if artifact_size is not None and max_size is not None:
            ratio = artifact_size / max_size
            if ratio > 0.9:
                length_penalty = min(0.3, (ratio - 0.9) * 3.0)

        return FitnessScore(
            correctness=correctness,
            procedure_following=procedure,
            conciseness=conciseness,
            length_penalty=length_penalty,
            feedback=feedback,
        )

    def _extract_score(self, text: str, field: str) -> float:
        """Extract a 0.0-1.0 score from structured output."""
        pattern = rf"{field}:\s*([0-9]*\.?[0-9]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _parse_score(match.group(1))
        return 0.5  # Default on parse failure

    def _extract_feedback(self, text: str) -> str:
        """Extract feedback text from structured output."""
        match = re.search(r"feedback:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()[:500]
        return text[:200]  # Fallback: first 200 chars of raw output
