"""Does the agent pick the right tool, and call it with the right arguments?

Phase 2's fitness signal. PLAN.md asks for three classes of example and this
module generates, scores and persists all three:

    clear        one tool is obviously right ("read lines 50-100 of config.py")
    confusable   two tools could work but one is better ("find all Python files
                 containing 'import os'" is search_files, not terminal(grep))
    no_tool      the right move is to answer directly and call nothing

The third class is the one that is easy to forget and expensive to lose. A
description rewritten to be maximally eager wins on the first two classes and
quietly turns the agent into something that reaches for a tool when it should
just answer, so "no tool" is scored as a first-class outcome with its own row in
the results, not filtered out of the dataset.

Two design notes:

* **The catalogue is rendered into the predictor's instructions**, not just
  passed as an input. That is what makes an instruction-mutating optimizer like
  GEPA actually rewrite description text rather than rewrite prose about it. The
  render format is strict and reversible, so evolved text can be parsed back out
  again; :func:`parse_tool_catalog` falls back to the baseline for any block it
  cannot read, so a reformatted candidate degrades to "no change" instead of to
  garbage in a source file.
* **Nothing here contacts a model unless it is asked to.** The dataset builder
  takes an injectable predictor and an optional LM, and every scoring function
  is pure.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import dspy

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import EvalDataset, EvalExample
from evolution.tools.tool_catalog import ToolCatalog, ToolDescriptions

__all__ = [
    "NO_TOOL",
    "CATEGORY_CLEAR",
    "CATEGORY_CONFUSABLE",
    "CATEGORY_NO_TOOL",
    "CATEGORIES",
    "TOOL_WEIGHT",
    "PARAM_WEIGHT",
    "ToolSelectionExample",
    "ToolSelectionDataset",
    "SelectionOutcome",
    "SelectionReport",
    "ToolSelectionDatasetBuilder",
    "GenerateSelectionCases",
    "SelectTool",
    "ToolSelector",
    "selector_predict_fn",
    "render_catalog_for",
    "catalog_signatures",
    "normalise_tool_name",
    "parse_params",
    "parameter_correctness",
    "score_selection",
    "evaluate_selection",
    "render_tool_catalog",
    "parse_tool_catalog",
    "extract_bundle",
    "split_examples",
    "tool_selection_metric",
    "gepa_selection_metric",
]

# The sentinel for "the agent should not call a tool at all".
NO_TOOL = "none"

CATEGORY_CLEAR = "clear"
CATEGORY_CONFUSABLE = "confusable"
CATEGORY_NO_TOOL = "no_tool"
CATEGORIES = (CATEGORY_CLEAR, CATEGORY_CONFUSABLE, CATEGORY_NO_TOOL)

# A wrong tool with perfect arguments is still a wrong tool, so selection
# carries most of the weight and parameters refine it.
TOOL_WEIGHT = 0.7
PARAM_WEIGHT = 0.3

_NO_TOOL_ALIASES = {
    "",
    "none",
    "null",
    "nil",
    "no_tool",
    "no-tool",
    "no tool",
    "notool",
    "n/a",
    "na",
    "nothing",
    "direct",
    "answer",
    "answer_directly",
    "respond",
    "respond_directly",
}


def normalise_tool_name(value: Any) -> str:
    """Reduce a model's answer to a bare tool name, or :data:`NO_TOOL`.

    Models wrap tool names in backticks, call them ``read_file()``, prefix them
    with "tool:", or answer "none". All of those mean the same thing and none of
    them should read as a wrong selection.
    """
    if value is None:
        return NO_TOOL
    text = str(value).strip().strip("`'\"").strip()
    text = re.sub(r"^(tool|function|call)\s*[:=]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(.*$", "", text, flags=re.DOTALL).strip()
    text = text.strip("`'\"").strip()
    if text.lower() in _NO_TOOL_ALIASES:
        return NO_TOOL
    return text


def parse_params(raw: Any) -> dict[str, Any]:
    """Coerce a predicted argument blob into a dict, never raising.

    Accepts a dict, a JSON object, or a JSON object buried in prose. Anything
    else scores as "no arguments given", which is the honest reading.
    """
    if isinstance(raw, dict):
        return dict(raw)
    if raw is None:
        return {}
    text = str(raw).strip()
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            loaded = json.loads(match.group())
        except json.JSONDecodeError:
            return {}
    return dict(loaded) if isinstance(loaded, dict) else {}


# ──────────────────────────────────────────────────────────────────────────
# Examples and datasets
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class ToolSelectionExample:
    """One (task, correct_tool, correct_params) triple from PLAN.md."""

    task: str
    correct_tool: str = NO_TOOL
    correct_params: dict[str, Any] = field(default_factory=dict)
    distractors: list[str] = field(default_factory=list)
    category: str = CATEGORY_CLEAR
    difficulty: str = "medium"
    source: str = "synthetic"

    def __post_init__(self) -> None:
        self.correct_tool = normalise_tool_name(self.correct_tool)
        if self.expects_no_tool:
            self.correct_params = {}

    @property
    def expects_no_tool(self) -> bool:
        """True when the right answer is to call nothing at all."""
        return self.correct_tool == NO_TOOL

    def to_dict(self) -> dict:
        """Serialise the example for the dataset file."""
        return {
            "task": self.task,
            "correct_tool": self.correct_tool,
            "correct_params": dict(self.correct_params),
            "distractors": list(self.distractors),
            "category": self.category,
            "difficulty": self.difficulty,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, blob: dict) -> "ToolSelectionExample":
        """Build from a stored dict, defaulting every absent field."""
        return cls(
            task=blob.get("task", ""),
            correct_tool=blob.get("correct_tool", NO_TOOL),
            correct_params=dict(blob.get("correct_params") or {}),
            distractors=list(blob.get("distractors") or []),
            category=blob.get("category", CATEGORY_CLEAR),
            difficulty=blob.get("difficulty", "medium"),
            source=blob.get("source", "synthetic"),
        )

    def to_eval_example(self) -> EvalExample:
        """Bridge into the Phase 1 dataset type for shared tooling."""
        expected = (
            "Call no tool; answer directly."
            if self.expects_no_tool
            else f"Call {self.correct_tool} with {json.dumps(self.correct_params, sort_keys=True)}"
        )
        return EvalExample(
            task_input=self.task,
            expected_behavior=expected,
            difficulty=self.difficulty,
            category=self.category,
            source=self.source,
        )

    def to_dspy_example(self) -> dspy.Example:
        """Convert to a ``dspy.Example`` with the task as its only input."""
        return dspy.Example(
            task=self.task,
            correct_tool=self.correct_tool,
            correct_params=dict(self.correct_params),
            category=self.category,
        ).with_inputs("task")


def _split_counts(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    """Split sizes that never starve a split that could have been filled.

    The ratios come from EvolutionConfig (0.5 / 0.25 / 0.25 by default). Once
    there are three or more examples every split gets at least one, because a
    cross-tool matrix with an empty holdout row cannot prove anything.
    """
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val > n - 1:
        n_train = max(1, n - 2)
        n_val = 1
    return n_train, n_val, n - n_train - n_val


def split_examples(
    examples: Sequence[ToolSelectionExample],
    config: Optional[EvolutionConfig] = None,
    seed: int = 0,
) -> "ToolSelectionDataset":
    """Stratify by correct tool, then split with the config's ratios.

    Stratifying matters here in a way it does not for skills: the cross-tool
    guard compares per-tool selection rates between splits, and a random split
    that happens to put every ``patch`` example in train makes that comparison
    meaningless.
    """
    train_ratio = config.train_ratio if config else 0.5
    val_ratio = config.val_ratio if config else 0.25

    groups: dict[str, list[ToolSelectionExample]] = {}
    for example in examples:
        groups.setdefault(example.correct_tool, []).append(example)

    rng = random.Random(seed)
    dataset = ToolSelectionDataset()
    for tool in sorted(groups):
        group = list(groups[tool])
        rng.shuffle(group)
        n_train, n_val, _ = _split_counts(len(group), train_ratio, val_ratio)
        dataset.train.extend(group[:n_train])
        dataset.val.extend(group[n_train:n_train + n_val])
        dataset.holdout.extend(group[n_train + n_val:])

    for split in (dataset.train, dataset.val, dataset.holdout):
        rng.shuffle(split)
    return dataset


@dataclass
class ToolSelectionDataset:
    """Train/val/holdout split of tool-selection triples, persisted as JSONL."""

    train: list[ToolSelectionExample] = field(default_factory=list)
    val: list[ToolSelectionExample] = field(default_factory=list)
    holdout: list[ToolSelectionExample] = field(default_factory=list)

    @property
    def all_examples(self) -> list[ToolSelectionExample]:
        """Every example across all three splits."""
        return self.train + self.val + self.holdout

    def __len__(self) -> int:
        return len(self.all_examples)

    def split(self, name: str) -> list[ToolSelectionExample]:
        """The named split, or raise ValueError for anything but the three."""
        if name not in ("train", "val", "holdout"):
            raise ValueError(f"unknown split {name!r}")
        return getattr(self, name)

    def category_counts(self) -> dict[str, int]:
        """Example count per category, zero-filled for categories with none."""
        counts = {category: 0 for category in CATEGORIES}
        for example in self.all_examples:
            counts[example.category] = counts.get(example.category, 0) + 1
        return counts

    def tool_counts(self) -> dict[str, int]:
        """Example count per correct tool."""
        counts: dict[str, int] = {}
        for example in self.all_examples:
            counts[example.correct_tool] = counts.get(example.correct_tool, 0) + 1
        return counts

    def save(self, path: Path) -> Path:
        """Write each split to its own JSONL file under *path*."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name in ("train", "val", "holdout"):
            with open(path / f"{name}.jsonl", "w", encoding="utf-8") as handle:
                for example in self.split(name):
                    handle.write(json.dumps(example.to_dict()) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> "ToolSelectionDataset":
        """Read a dataset back, skipping any split with no file on disk."""
        path = Path(path)
        dataset = cls()
        for name in ("train", "val", "holdout"):
            split_file = path / f"{name}.jsonl"
            if not split_file.exists():
                continue
            examples = []
            with open(split_file, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        examples.append(ToolSelectionExample.from_dict(json.loads(line)))
            setattr(dataset, name, examples)
        return dataset

    def to_dspy_examples(self, split: str = "train") -> list[dspy.Example]:
        """The named split as ``dspy.Example`` objects."""
        return [example.to_dspy_example() for example in self.split(split)]

    def to_eval_dataset(self) -> EvalDataset:
        """Same examples in the Phase 1 shape, for shared reporting code."""
        return EvalDataset(
            train=[e.to_eval_example() for e in self.train],
            val=[e.to_eval_example() for e in self.val],
            holdout=[e.to_eval_example() for e in self.holdout],
        )


# ──────────────────────────────────────────────────────────────────────────
# Rendering the catalogue the agent sees
# ──────────────────────────────────────────────────────────────────────────

_CATALOG_PREAMBLE = (
    "You are the tool-selection layer of an agent. Given a task, choose exactly "
    "one tool from the catalogue below, or 'none' when the task needs no tool "
    "and should be answered directly. Use the tool descriptions to decide.\n\n"
    "Tool catalogue:\n"
)

_DESC_INDENT = "  "
_PARAM_INDENT = "    "
_TOOL_HEADER = re.compile(r"^##\s+(\S+)\s*$")


def _indent_block(text: str, pad: str) -> list[str]:
    return [pad + line for line in text.split("\n")]


def render_tool_catalog(
    bundle: dict[str, ToolDescriptions],
    signatures: Optional[dict[str, str]] = None,
    preamble: bool = True,
) -> str:
    """Render descriptions into the exact text the selector reads.

    Reversible by :func:`parse_tool_catalog`: continuation lines carry a fixed
    indent, so descriptions containing blank lines and newlines - hermes-agent's
    ``patch`` description has both - survive a round trip byte for byte.
    """
    signatures = signatures or {}
    lines: list[str] = []
    for tool_name in sorted(bundle):
        descriptions = bundle[tool_name]
        lines.append(f"## {tool_name}")
        lines.append(f"signature: {signatures.get(tool_name, tool_name + '(...)')}")
        lines.append("description:")
        lines.extend(_indent_block(descriptions.description, _DESC_INDENT))
        if descriptions.params:
            lines.append("params:")
            for param in sorted(descriptions.params):
                lines.append(f"- {param}:")
                lines.extend(_indent_block(descriptions.params[param], _PARAM_INDENT))
        lines.append("")
    body = "\n".join(lines).rstrip("\n")
    return (_CATALOG_PREAMBLE + body) if preamble else body


def render_catalog_for(catalog: ToolCatalog, bundle: Optional[dict] = None) -> str:
    """Render a catalogue, showing each tool's frozen call signature."""
    bundle = bundle if bundle is not None else catalog.bundle()
    signatures = {e.tool_name: e.signature() for e in catalog}
    return render_tool_catalog(bundle, signatures=signatures)


def parse_tool_catalog(
    text: str,
    baseline: dict[str, ToolDescriptions],
) -> dict[str, ToolDescriptions]:
    """Read descriptions back out of rendered catalogue text.

    Only tools and parameters that already exist in *baseline* are accepted, and
    anything that fails to parse keeps its baseline text. An optimizer is free
    to reformat, add commentary, or drop a section: the worst case is that the
    affected description simply does not change.
    """
    result = {name: entry.copy() for name, entry in baseline.items()}
    if not text:
        return result

    current: Optional[str] = None
    mode: Optional[str] = None  # "description" or a parameter name
    buffer: list[str] = []
    collected: dict[str, dict[str, list[str]]] = {}

    def flush() -> None:
        """Commit the buffered lines to the tool and mode currently being read."""
        if current is None or mode is None:
            return
        collected.setdefault(current, {})[mode] = list(buffer)

    for line in text.split("\n"):
        header = _TOOL_HEADER.match(line)
        if header:
            flush()
            buffer = []
            mode = None
            current = header.group(1)
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped == "description:":
            flush()
            buffer = []
            mode = "description"
            continue
        if stripped == "params:":
            flush()
            buffer = []
            mode = None
            continue
        param_start = re.match(r"^-\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$", line)
        if param_start:
            flush()
            buffer = []
            mode = f"param:{param_start.group(1)}"
            continue
        if mode is None:
            continue  # covers the signature line, which is frozen and ignored
        pad = _DESC_INDENT if mode == "description" else _PARAM_INDENT
        buffer.append(line[len(pad):] if line.startswith(pad) else line.strip())

    flush()

    for tool_name, blocks in collected.items():
        target = result.get(tool_name)
        if target is None:
            continue  # a tool the optimizer invented; ignored on purpose
        for key, lines in blocks.items():
            value = "\n".join(lines).strip("\n")
            if not value.strip():
                continue
            if key == "description":
                target.description = value
            elif key.startswith("param:"):
                param = key.split(":", 1)[1]
                if param in target.params:
                    target.params[param] = value
    return result


# ──────────────────────────────────────────────────────────────────────────
# The selector
# ──────────────────────────────────────────────────────────────────────────


class SelectTool(dspy.Signature):
    """Choose the single best tool for a task, or none.

    Answer with the exact tool name from the catalogue, or 'none' if the task
    should be answered directly without calling anything. Then give the
    arguments you would pass, as a JSON object using only the parameter names in
    that tool's signature.
    """

    task: str = dspy.InputField(desc="What the user asked for")
    tool_catalog: str = dspy.InputField(desc="The tools available, with their descriptions")
    tool_name: str = dspy.OutputField(desc="Exact tool name, or 'none'")
    parameters: str = dspy.OutputField(desc="JSON object of arguments, {} when no tool is called")


class ToolSelector(dspy.Module):
    """Tool selection with the description bundle as the evolvable parameter.

    The rendered catalogue goes into the predictor's instructions *and* is
    passed as an input field. The instructions copy is what an
    instruction-mutating optimizer edits, which is how evolved description text
    finds its way back out through :func:`extract_bundle`.
    """

    def __init__(
        self,
        bundle: dict[str, ToolDescriptions],
        signatures: Optional[dict[str, str]] = None,
    ):
        super().__init__()
        self.bundle = {name: entry.copy() for name, entry in bundle.items()}
        self.signatures = dict(signatures or {})
        self.rendered = render_tool_catalog(self.bundle, self.signatures)
        self.predictor = dspy.ChainOfThought(SelectTool.with_instructions(self.rendered))

    def forward(self, task: str) -> dspy.Prediction:
        """Select a tool for *task* and parse the parameters it proposed."""
        result = self.predictor(task=task, tool_catalog=self.rendered)
        return dspy.Prediction(
            tool_name=normalise_tool_name(getattr(result, "tool_name", "")),
            parameters=parse_params(getattr(result, "parameters", "")),
            raw_tool_name=str(getattr(result, "tool_name", "")),
        )


def extract_bundle(
    module: Any,
    baseline: dict[str, ToolDescriptions],
) -> dict[str, ToolDescriptions]:
    """Recover the evolved description bundle from an optimized module.

    Tried in order: the module's own bundle attribute when it has moved, then
    the catalogue parsed back out of the predictor instructions the optimizer
    rewrote, then the baseline unchanged. Never raises - a run that produced no
    usable mutation reports no improvement rather than falling over.
    """
    candidate = getattr(module, "bundle", None)
    if isinstance(candidate, dict) and candidate:
        rendered_now = {k: v.to_dict() for k, v in candidate.items()}
        rendered_base = {k: v.to_dict() for k, v in baseline.items()}
        if rendered_now != rendered_base:
            return {name: entry.copy() for name, entry in candidate.items()}

    instructions = ""
    try:
        for _, predictor in module.named_predictors():
            text = getattr(getattr(predictor, "signature", None), "instructions", "")
            if text and "## " in text:
                instructions = text
                break
    except Exception:
        instructions = ""

    if instructions:
        return parse_tool_catalog(instructions, baseline)
    return {name: entry.copy() for name, entry in baseline.items()}


# ──────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────


def _comparable(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return json.dumps([_comparable(v) for v in value])
    if isinstance(value, dict):
        return json.dumps({k: _comparable(v) for k, v in sorted(value.items())})
    return str(value).strip().strip("`'\"").casefold()


def parameter_correctness(expected: dict[str, Any], predicted: dict[str, Any]) -> float:
    """Fraction of the expected arguments that were supplied correctly.

    Only the arguments the example pins down are scored. Extra arguments are not
    penalised: the schema is frozen and full of optional parameters with
    defaults, so passing ``limit=500`` explicitly is not an error.
    """
    if not expected:
        return 1.0
    predicted = predicted or {}
    matched = sum(
        1
        for key, value in expected.items()
        if key in predicted and _comparable(predicted[key]) == _comparable(value)
    )
    return matched / len(expected)


@dataclass
class SelectionOutcome:
    """One example, scored."""

    example: ToolSelectionExample
    predicted_tool: str
    predicted_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.predicted_tool = normalise_tool_name(self.predicted_tool)

    @property
    def expected_tool(self) -> str:
        """The tool this example should have selected."""
        return self.example.correct_tool

    @property
    def tool_correct(self) -> bool:
        """True when the predicted tool matches the expected one."""
        return self.predicted_tool == self.expected_tool

    @property
    def param_score(self) -> float:
        """Parameter correctness, 0.0 when the tool itself was wrong."""
        if not self.tool_correct:
            return 0.0
        return parameter_correctness(self.example.correct_params, self.predicted_params)

    @property
    def score(self) -> float:
        """Weighted tool and parameter score for this example."""
        return TOOL_WEIGHT * float(self.tool_correct) + PARAM_WEIGHT * self.param_score

    def feedback(self) -> str:
        """Why this example scored what it did, in words GEPA can reflect on."""
        if self.tool_correct and self.param_score == 1.0:
            return f"Correct: chose {self.predicted_tool} with the right arguments."
        if self.tool_correct:
            missing = sorted(
                key
                for key, value in self.example.correct_params.items()
                if key not in self.predicted_params
                or _comparable(self.predicted_params[key]) != _comparable(value)
            )
            return (
                f"Chose {self.predicted_tool} correctly but the arguments were wrong "
                f"for: {', '.join(missing)}. The parameter descriptions should make "
                f"the expected values unambiguous."
            )
        if self.expected_tool == NO_TOOL:
            return (
                f"Called {self.predicted_tool} for a task that needed no tool. Its "
                f"description is too eager about when it applies."
            )
        if self.predicted_tool == NO_TOOL:
            return (
                f"Called nothing when {self.expected_tool} was right. That tool's "
                f"description does not make its applicability obvious."
            )
        return (
            f"Chose {self.predicted_tool} when {self.expected_tool} was right. The "
            f"two descriptions overlap; {self.expected_tool} should state what it is "
            f"for and {self.predicted_tool} should say when to prefer the other."
        )

    def to_dict(self) -> dict:
        """Serialise one selection outcome."""
        return {
            "task": self.example.task,
            "category": self.example.category,
            "expected_tool": self.expected_tool,
            "predicted_tool": self.predicted_tool,
            "tool_correct": self.tool_correct,
            "param_score": round(self.param_score, 4),
            "score": round(self.score, 4),
        }


@dataclass
class SelectionReport:
    """Aggregate scores over a set of outcomes."""

    outcomes: list[SelectionOutcome] = field(default_factory=list)

    @property
    def n(self) -> int:
        """How many examples were scored."""
        return len(self.outcomes)

    @property
    def tool_accuracy(self) -> float:
        """Fraction of examples where the right tool was selected."""
        if not self.outcomes:
            return 0.0
        return sum(1 for o in self.outcomes if o.tool_correct) / len(self.outcomes)

    @property
    def param_accuracy(self) -> float:
        """Argument correctness among the tools that were chosen correctly.

        Kept separate from ``tool_accuracy`` on purpose. Mixing them hides which
        half of the problem a rewrite actually moved.
        """
        correct = [o for o in self.outcomes if o.tool_correct]
        if not correct:
            return 0.0
        return sum(o.param_score for o in correct) / len(correct)

    @property
    def score(self) -> float:
        """Mean weighted score across every outcome."""
        if not self.outcomes:
            return 0.0
        return sum(o.score for o in self.outcomes) / len(self.outcomes)

    def by_category(self) -> dict[str, float]:
        """Selection accuracy per category."""
        buckets: dict[str, list[SelectionOutcome]] = {}
        for outcome in self.outcomes:
            buckets.setdefault(outcome.example.category, []).append(outcome)
        return {
            category: sum(1 for o in group if o.tool_correct) / len(group)
            for category, group in sorted(buckets.items())
        }

    def failures(self) -> list[SelectionOutcome]:
        """The outcomes where the wrong tool was selected."""
        return [o for o in self.outcomes if not o.tool_correct]

    def to_dict(self) -> dict:
        """Serialise the selection report and its per-category breakdown."""
        return {
            "n": self.n,
            "tool_accuracy": round(self.tool_accuracy, 4),
            "param_accuracy": round(self.param_accuracy, 4),
            "score": round(self.score, 4),
            "by_category": {k: round(v, 4) for k, v in self.by_category().items()},
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


PredictFn = Callable[[ToolSelectionExample], Any]


def score_selection(
    example: ToolSelectionExample,
    predicted_tool: Any,
    predicted_params: Any = None,
) -> SelectionOutcome:
    """Score one prediction against one example."""
    return SelectionOutcome(
        example=example,
        predicted_tool=normalise_tool_name(predicted_tool),
        predicted_params=parse_params(predicted_params),
    )


def evaluate_selection(
    examples: Iterable[ToolSelectionExample],
    predict: PredictFn,
) -> SelectionReport:
    """Run *predict* over every example and score the results.

    *predict* takes an example and returns either a ``(tool, params)`` pair or
    anything with ``tool_name`` and ``parameters`` attributes, which is what a
    :class:`ToolSelector` forward pass returns. A prediction that raises is
    scored as a miss rather than aborting the sweep: one flaky call should not
    throw away an entire evaluation.
    """
    outcomes: list[SelectionOutcome] = []
    for example in examples:
        try:
            raw = predict(example)
        except Exception:
            raw = (NO_TOOL, {})
        if isinstance(raw, tuple):
            tool, params = (list(raw) + [None])[:2]
        else:
            tool = getattr(raw, "tool_name", raw)
            params = getattr(raw, "parameters", None)
        outcomes.append(score_selection(example, tool, params))
    return SelectionReport(outcomes=outcomes)


def selector_predict_fn(module: Any) -> PredictFn:
    """Adapt a :class:`ToolSelector` to the ``predict`` shape above."""

    def predict(example: ToolSelectionExample):
        """Run the module over one example's task."""
        return module(task=example.task)

    return predict


def tool_selection_metric(example, prediction, trace=None) -> float:
    """DSPy metric: how good was this selection, 0-1."""
    target = ToolSelectionExample(
        task=str(getattr(example, "task", "")),
        correct_tool=getattr(example, "correct_tool", NO_TOOL),
        correct_params=dict(getattr(example, "correct_params", {}) or {}),
        category=str(getattr(example, "category", CATEGORY_CLEAR)),
    )
    outcome = score_selection(
        target,
        getattr(prediction, "tool_name", prediction),
        getattr(prediction, "parameters", None),
    )
    return outcome.score


def gepa_selection_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA's five-argument feedback metric.

    Returns a plain score for ordinary evaluation and a score with written
    feedback when GEPA is reflecting on a component. The feedback names the tool
    that was picked instead, which is the signal PLAN.md wants GEPA reading:
    traces of wrong selections, explained.
    """
    target = ToolSelectionExample(
        task=str(getattr(gold, "task", "")),
        correct_tool=getattr(gold, "correct_tool", NO_TOOL),
        correct_params=dict(getattr(gold, "correct_params", {}) or {}),
        category=str(getattr(gold, "category", CATEGORY_CLEAR)),
    )
    outcome = score_selection(
        target,
        getattr(pred, "tool_name", pred),
        getattr(pred, "parameters", None),
    )
    if pred_name is None:
        return outcome.score
    return dspy.Prediction(score=outcome.score, feedback=outcome.feedback())


# ──────────────────────────────────────────────────────────────────────────
# Synthetic dataset generation
# ──────────────────────────────────────────────────────────────────────────


class GenerateSelectionCases(dspy.Signature):
    """Write realistic tool-selection test cases for an agent's tool catalogue.

    Produce one of three kinds of case, as instructed:
    - clear: the focus tool is unmistakably the right choice.
    - confusable: two tools could plausibly work and the focus tool is the
      better one. Name the plausible alternative as a distractor.
    - no_tool: the task should be answered directly with no tool call at all,
      for example a definition, an opinion, or arithmetic.

    Every task must read like something a real user would type. Use only
    parameter names that appear in the tool's signature. For no_tool cases set
    correct_tool to "none" and correct_params to an empty object.
    """

    tool_catalog: str = dspy.InputField(desc="Available tools, signatures and descriptions")
    focus_tool: str = dspy.InputField(desc="Tool the cases should target, or 'any'")
    kind: str = dspy.InputField(desc="One of: clear, confusable, no_tool")
    num_cases: int = dspy.InputField(desc="How many cases to write")
    cases: str = dspy.OutputField(
        desc=(
            "JSON array. Each item: {task, correct_tool, correct_params, "
            "distractors, difficulty}"
        )
    )


@dataclass
class ToolSelectionDatasetBuilder:
    """Generate a tool-selection dataset with a strong model.

    PLAN.md's recipe: 10-20 clear cases per tool, 10-20 confusable cases, and 10
    no-tool cases, for roughly 200-400 triples in total.

    Generated cases are checked against the real schemas before they are kept.
    A model that invents a tool name or an argument that does not exist would
    otherwise poison the fitness signal with examples no description could ever
    satisfy, so unknown tools are dropped and unknown arguments are stripped.
    """

    catalog: ToolCatalog
    config: Optional[EvolutionConfig] = None
    generator: Any = None
    lm: Any = None
    seed: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.generator is None:
            self.generator = dspy.ChainOfThought(GenerateSelectionCases)

    # ── generation ──────────────────────────────────────────────────────
    def _call(self, **kwargs) -> Any:
        if self.lm is None:
            return self.generator(**kwargs)
        with dspy.context(lm=self.lm):
            return self.generator(**kwargs)

    def _rendered_catalog(self) -> str:
        return render_catalog_for(self.catalog)

    def _batch(self, kind: str, focus_tool: str, n: int) -> list[ToolSelectionExample]:
        if n <= 0:
            return []
        result = self._call(
            tool_catalog=self._rendered_catalog(),
            focus_tool=focus_tool,
            kind=kind,
            num_cases=n,
        )
        return self._parse(getattr(result, "cases", ""), kind, focus_tool)

    def _parse(self, raw: Any, kind: str, focus_tool: str) -> list[ToolSelectionExample]:
        if isinstance(raw, list):
            items = raw
        else:
            text = str(raw or "").strip()
            try:
                items = json.loads(text)
            except json.JSONDecodeError:
                match = re.search(r"\[.*\]", text, re.DOTALL)
                if not match:
                    self.rejected.append((f"{kind}/{focus_tool}", "output was not JSON"))
                    return []
                try:
                    items = json.loads(match.group())
                except json.JSONDecodeError:
                    self.rejected.append((f"{kind}/{focus_tool}", "output was not JSON"))
                    return []
        if not isinstance(items, list):
            self.rejected.append((f"{kind}/{focus_tool}", "output was not a JSON array"))
            return []

        known = set(self.catalog.names)
        examples: list[ToolSelectionExample] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            task = str(item.get("task", "")).strip()
            if not task:
                continue
            tool = normalise_tool_name(item.get("correct_tool"))

            if kind == CATEGORY_NO_TOOL:
                tool = NO_TOOL
            elif tool == NO_TOOL:
                self.rejected.append((task[:60], f"no-tool case generated as kind {kind!r}"))
                continue
            elif tool not in known:
                self.rejected.append((task[:60], f"unknown tool {tool!r}"))
                continue

            params: dict[str, Any] = {}
            if tool != NO_TOOL:
                entry = self.catalog.require(tool)
                allowed = set(entry.param_names)
                for key, value in (item.get("correct_params") or {}).items():
                    if key in allowed:
                        params[key] = value
                    else:
                        self.rejected.append((task[:60], f"unknown argument {key!r} for {tool}"))

            distractors = [
                normalise_tool_name(d)
                for d in (item.get("distractors") or [])
                if normalise_tool_name(d) in known and normalise_tool_name(d) != tool
            ]

            examples.append(
                ToolSelectionExample(
                    task=task,
                    correct_tool=tool,
                    correct_params=params,
                    distractors=distractors,
                    category=kind,
                    difficulty=str(item.get("difficulty", "medium")),
                    source="synthetic",
                )
            )
        return examples

    def generate(
        self,
        per_tool: int = 12,
        num_confusable: int = 16,
        num_no_tool: int = 10,
    ) -> ToolSelectionDataset:
        """Generate all three example classes and split them."""
        collected: list[ToolSelectionExample] = []
        for entry in self.catalog:
            collected.extend(self._batch(CATEGORY_CLEAR, entry.tool_name, per_tool))
        collected.extend(self._batch(CATEGORY_CONFUSABLE, "any", num_confusable))
        collected.extend(self._batch(CATEGORY_NO_TOOL, "any", num_no_tool))

        deduped: list[ToolSelectionExample] = []
        seen: set[str] = set()
        for example in collected:
            key = " ".join(example.task.lower().split())
            if key in seen:
                self.rejected.append((example.task[:60], "duplicate task"))
                continue
            seen.add(key)
            deduped.append(example)

        return split_examples(deduped, self.config, seed=self.seed)


def catalog_signatures(catalog: ToolCatalog) -> dict[str, str]:
    """``{tool_name: frozen call signature}`` for rendering."""
    return {entry.tool_name: entry.signature() for entry in catalog}
