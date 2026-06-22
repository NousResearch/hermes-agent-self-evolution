"""Guidance-quality judge as a DSPy module (Approach A).

The mutable guidance block texts are packed into the predictor's signature
instruction as a labeled block. GEPA mutates that instruction string; after
compile, parse_guidance() reads each evolved block back out.
"""

import re

import dspy

from evolution.prompts.targets import (
    TARGET_GUIDANCE, LABEL_OPEN, LABEL_CLOSE, BLOCK_START, BLOCK_END,
)

INSTRUCTION_HEADER = (
    "You are a behavioral judge. Given a task, the relevant system prompt "
    "guidance, and a model response, score how well the response follows "
    "the guidance. Respond with a score from 0.0 to 1.0."
)


def build_instruction(target_guidance: dict, all_guidance: list) -> str:
    """Render the judge instruction with the mutable labeled block.

    target_guidance: {constant_name: text} for the 6 evolvable blocks.
    all_guidance: [(name, text), ...] for ALL constants (frozen shown for context).
    """
    lines = [INSTRUCTION_HEADER, "", BLOCK_START]
    for name in TARGET_GUIDANCE:
        if name in target_guidance:
            lines.append(f"{LABEL_OPEN}{name}{LABEL_CLOSE} {target_guidance[name]}")
    lines.append(BLOCK_END)
    lines.append("")
    lines.append("Frozen guidance blocks (context only, not evolved):")
    target_set = set(TARGET_GUIDANCE)
    for name, text in all_guidance:
        if name not in target_set:
            lines.append(f"- {name}: {text}")
    return "\n".join(lines)


def parse_guidance(instruction: str, baseline: dict) -> dict:
    """Extract {constant_name: text} from a (possibly GEPA-mutated) instruction.

    Reads each `` [[name]] text `` segment up to the next `` [[`` or the block end.
    Any target whose label is missing or empty falls back to its baseline.
    """
    result = dict(baseline)
    block = instruction
    if BLOCK_START in instruction and BLOCK_END in instruction:
        block = instruction.split(BLOCK_START, 1)[1].split(BLOCK_END, 1)[0]

    pattern = re.compile(
        re.escape(LABEL_OPEN) + r"(\w+)" + re.escape(LABEL_CLOSE) + r"(.*?)(?=" +
        re.escape(LABEL_OPEN) + r"\w+" + re.escape(LABEL_CLOSE) + r"|$)",
        re.DOTALL,
    )
    for match in pattern.finditer(block):
        name = match.group(1)
        text = match.group(2).strip()
        if name in baseline and text:
            result[name] = text
    return result


class GuidanceJudgeModule(dspy.Module):
    """Wraps a behavioral judge for GEPA optimization.

    The mutable guidance blocks are embedded in the predictor's signature
    instruction. After GEPA mutates the instruction, `guidance` reads them
    back out (with baseline fallback for any missing label).
    """

    def __init__(self, target_guidance: dict, all_guidance: list):
        super().__init__()
        self._baseline = dict(target_guidance)
        instruction = build_instruction(target_guidance, all_guidance)
        signature = type("_GuidanceJudge", (dspy.Signature,), {
            "__doc__": instruction,
            "task": dspy.InputField(),
            "guidance_block": dspy.InputField(desc="The relevant guidance block text"),
            "model_response": dspy.InputField(desc="The model's output to evaluate"),
            "score": dspy.OutputField(desc="0.0 to 1.0 behavioral compliance score"),
        })
        self.judge = dspy.Predict(signature)

    def forward(self, task: str, guidance_block: str, model_response: str) -> dspy.Prediction:
        result = self.judge(task=task, guidance_block=guidance_block, model_response=model_response)
        return dspy.Prediction(score=result.score)

    @property
    def guidance(self) -> dict:
        """Current (possibly evolved) target guidance, with baseline fallback."""
        try:
            for _name, pred in self.named_predictors():
                return parse_guidance(pred.signature.instructions, self._baseline)
            return parse_guidance(self.judge.signature.instructions, self._baseline)
        except Exception:
            return dict(self._baseline)
