"""Opt-in LLM generator for lesson probes.

Turns one lesson paragraph into a candidate probe (scenario + expected
phrases). The result is deliberately *untrusted*: ``run_lesson_probes``
subjects every generated probe to the lesion check, and a probe that cannot
tell the lesson from its absence is marked dead and never gates.
"""

from __future__ import annotations

import json
from typing import Optional

from evolution.core.lesson_probes import ProbeGenerator


def make_dspy_generator(model: str) -> ProbeGenerator:
    """Build a generator backed by ``dspy.LM(model)``. Imports dspy lazily."""
    import dspy

    class ProbeFromLesson(dspy.Signature):
        """A skill contains a lesson learned from a past failure.

        Write ONE concrete, self-contained task an agent could be given in
        which an agent that ignores this lesson would act wrongly, and list
        1-3 short lowercase phrases that a response honoring the lesson must
        contain. Do not restate the lesson; describe the situation.
        """

        lesson: str = dspy.InputField(desc="One paragraph from the skill that encodes a lesson")
        scenario: str = dspy.OutputField(desc="A concrete task input that exercises the lesson")
        must_include: str = dspy.OutputField(desc="JSON list of 1-3 lowercase phrases a correct response must contain")

    predictor = dspy.ChainOfThought(ProbeFromLesson)
    lm = dspy.LM(model)

    def generate(lesson_text: str) -> Optional[dict]:
        with dspy.context(lm=lm):
            result = predictor(lesson=lesson_text)
        raw = str(getattr(result, "must_include", "") or "")
        try:
            phrases = json.loads(raw)
            if not isinstance(phrases, list):
                phrases = [phrases]
        except json.JSONDecodeError:
            phrases = [p.strip().strip("\"'") for p in raw.split(",")]
        phrases = [str(p).strip() for p in phrases if str(p).strip()][:3]
        scenario = str(getattr(result, "scenario", "") or "").strip()
        if not scenario or not phrases:
            return None
        return {"scenario": scenario, "must_include": phrases}

    return generate
