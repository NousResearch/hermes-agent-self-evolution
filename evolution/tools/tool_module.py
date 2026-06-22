"""Tool-selector classifier as a DSPy module (Approach A).

The mutable target tool descriptions are packed into the predictor's signature
instruction as a labeled block. GEPA mutates that instruction string; after
compile, parse_descriptions() reads each tool's evolved description back out.
"""

import re

import dspy

from evolution.tools.targets import (
    TARGET_TOOLS, LABEL_OPEN, LABEL_CLOSE, BLOCK_START, BLOCK_END,
)

INSTRUCTION_HEADER = (
    "You are a tool selector. Given a task, choose the single best tool by name, "
    'or "none" if no tool is needed. Respond with only the tool name.'
)


def build_instruction(target_descriptions: dict, all_tools: list) -> str:
    """Render the classifier instruction with the mutable labeled block.

    target_descriptions: {tool_name: description} for the 6 evolvable tools.
    all_tools: [(name, description), ...] for ALL tools (fixed context). Targets
               are skipped here since they appear in the mutable block above.
    """
    lines = [INSTRUCTION_HEADER, "", BLOCK_START]
    for name in TARGET_TOOLS:
        if name in target_descriptions:
            lines.append(f"{LABEL_OPEN}{name}{LABEL_CLOSE} {target_descriptions[name]}")
    lines.append(BLOCK_END)
    lines.append("")
    lines.append("Other available tools (fixed context):")
    target_set = set(TARGET_TOOLS)
    for name, desc in all_tools:
        if name not in target_set:
            lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def parse_descriptions(instruction: str, baseline: dict) -> dict:
    """Extract {tool_name: description} from a (possibly GEPA-mutated) instruction.

    Reads each `` [[name]] text `` segment up to the next `` [[`` or the block end.
    Any target tool whose label is missing or empty falls back to its baseline
    description — never returns an empty string for a target.
    """
    result = dict(baseline)  # start from baseline; overwrite what we find
    # Limit parsing to the mutable block if the markers are present.
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


class ToolSelectorModule(dspy.Module):
    """Wraps a tool-selection classifier for GEPA optimization.

    The mutable tool descriptions are embedded in the predictor's signature
    instruction. After GEPA mutates the instruction, `descriptions` reads them
    back out (with baseline fallback for any missing label).
    """

    def __init__(self, target_descriptions: dict, all_tools: list):
        super().__init__()
        self._baseline = dict(target_descriptions)
        instruction = build_instruction(target_descriptions, all_tools)
        signature = type("_ToolSelector", (dspy.Signature,), {
            "__doc__": instruction,
            "task": dspy.InputField(),
            "chosen_tool": dspy.OutputField(desc="The single best tool name, or 'none'"),
        })
        self.selector = dspy.Predict(signature)

    def forward(self, task: str) -> dspy.Prediction:
        result = self.selector(task=task)
        return dspy.Prediction(chosen_tool=result.chosen_tool)

    @property
    def descriptions(self) -> dict:
        """Current (possibly evolved) target descriptions, with baseline fallback."""
        try:
            for _name, pred in self.named_predictors():
                return parse_descriptions(pred.signature.instructions, self._baseline)
            return parse_descriptions(self.selector.signature.instructions, self._baseline)
        except Exception:
            return dict(self._baseline)


# ── Phase 4: parameter-aware instruction builder and parse-back ──────────────


def build_param_instruction(
    target_descriptions: dict,
    target_params: dict,
    all_tools: list,
) -> str:
    """Render the classifier instruction with both tool AND parameter description labels.

    target_descriptions: {tool_name: description} for the 6 evolvable tools.
    target_params: {tool_name: {param_name: description}} for evolvable params.
    all_tools: [(name, description), ...] for ALL tools.
    """
    lines = [INSTRUCTION_HEADER, "", BLOCK_START]
    for name in TARGET_TOOLS:
        if name in target_descriptions:
            lines.append(f"{LABEL_OPEN}{name}{LABEL_CLOSE} {target_descriptions[name]}")
            if name in target_params:
                for pname, pdesc in target_params[name].items():
                    lines.append(f"  {LABEL_OPEN}{name}.{pname}{LABEL_CLOSE} {pdesc}")
    lines.append(BLOCK_END)
    lines.append("")
    lines.append("Other available tools (fixed context):")
    target_set = set(TARGET_TOOLS)
    for name, desc in all_tools:
        if name not in target_set:
            lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def parse_param_descriptions(
    instruction: str,
    tool_baseline: dict,
    param_baseline: dict,
) -> tuple:
    """Extract (tool_descriptions, param_descriptions) from a GEPA-mutated instruction.

    Returns:
        tool_descriptions: {tool_name: text} -- same as parse_descriptions
        param_descriptions: {tool_name: {param_name: text}} -- param-level descriptions
    """
    tool_result = dict(tool_baseline)
    param_result = {name: {} for name in param_baseline}

    block = instruction
    if BLOCK_START in instruction and BLOCK_END in instruction:
        block = instruction.split(BLOCK_START, 1)[1].split(BLOCK_END, 1)[0]

    # Match both tool-level ([[name]]) and param-level ([[name.param]]) labels
    pattern = re.compile(
        re.escape(LABEL_OPEN) + r"(\w+(?:\.\w+)?)" + re.escape(LABEL_CLOSE) + r"(.*?)(?=" +
        re.escape(LABEL_OPEN) + r"\w+(?:\.\w+)?" + re.escape(LABEL_CLOSE) + r"|$)",
        re.DOTALL,
    )
    for match in pattern.finditer(block):
        full_name = match.group(1)
        text = match.group(2).strip()
        if "." in full_name:
            tool_name, param_name = full_name.split(".", 1)
            if tool_name in param_result and text:
                param_result[tool_name][param_name] = text
        else:
            if full_name in tool_result and text:
                tool_result[full_name] = text

    return tool_result, param_result
