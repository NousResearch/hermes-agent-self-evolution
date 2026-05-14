"""Evaluation dataset generation for agent-self-evolution.

Sources:
A) Synthetic generation — LLM reads a skill/tool/prompt and generates test cases
B) SessionDB mining — extract real usage patterns and score with LLM-as-judge
C) Golden sets — hand-curated JSONL files
"""

import json
import logging
import random
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import dspy

from evolution.core.config import EvolutionConfig

if TYPE_CHECKING:
    from evolution.tools.tool_source import ToolManifest

logger = logging.getLogger(__name__)


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


def split_examples(
    examples: list[EvalExample],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    holdout_ratio: float,
) -> EvalDataset:
    """Shuffle ``examples`` and split into train/val/holdout.

    Ratios are normalized internally — they don't need to sum to 1. Empty
    input returns an empty dataset. All three ratios are required (no
    defaults) so callers can't drift from EvolutionConfig defaults silently.
    """
    if not examples:
        return EvalDataset()

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    n_total = len(shuffled)
    ratio_sum = train_ratio + val_ratio + holdout_ratio
    n_train = max(1, int(n_total * train_ratio / ratio_sum))
    n_val = max(1, int(n_total * val_ratio / ratio_sum))

    return EvalDataset(
        train=shuffled[:n_train],
        val=shuffled[n_train:n_train + n_val],
        holdout=shuffled[n_train + n_val:],
    )


class SyntheticDatasetBuilder:
    """Generate evaluation datasets using a strong LLM.

    Reads the target artifact (skill file, tool description, etc.)
    and generates realistic (task_input, expected_behavior) pairs.
    """

    class GenerateTestCases(dspy.Signature):
        """Generate realistic evaluation test cases for an agent skill or tool.

        Given the full text of a skill/tool description, generate diverse test cases
        that would exercise different aspects of the skill. Each test case should include:
        - A realistic task_input (what a user would actually ask)
        - An expected_behavior rubric (what a good response should contain/do, NOT exact text)
        - A difficulty level (easy, medium, hard)
        - A category (what aspect of the skill this tests)
        """
        artifact_text: str = dspy.InputField(desc="The full text of the skill/tool/prompt being tested")
        artifact_type: str = dspy.InputField(desc="Type: 'skill', 'tool_description', or 'prompt_section'")
        num_cases: int = dspy.InputField(desc="Number of test cases to generate")
        test_cases: str = dspy.OutputField(desc="JSON array of test cases, each with: task_input, expected_behavior, difficulty, category")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.generator = dspy.ChainOfThought(self.GenerateTestCases)

    def generate(
        self,
        artifact_text: str,
        artifact_type: str = "skill",
        num_cases: Optional[int] = None,
    ) -> EvalDataset:
        """Generate a full eval dataset with train/val/holdout splits."""

        n = num_cases or self.config.eval_dataset_size

        # max_tokens=16000 because at eval_dataset_size=60 the JSON output
        # truncates mid-string at 4000, producing JSONDecodeError → process exit.
        _lm = self.config.get_lm("judge")
        lm = dspy.LM(_lm.model, **_lm.lm_kwargs, temperature=0.7, max_tokens=16000, request_timeout=120, num_retries=5)

        with dspy.context(lm=lm):
            result = self.generator(
                artifact_text=artifact_text,
                artifact_type=artifact_type,
                num_cases=n,
            )

        try:
            cases_raw = json.loads(result.test_cases)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\[.*\]', result.test_cases, re.DOTALL)
            if match:
                cases_raw = json.loads(match.group())
            else:
                raise ValueError(f"Could not parse test cases from LLM output: {result.test_cases[:200]}")

        examples = [
            EvalExample(
                task_input=c.get("task_input", ""),
                expected_behavior=c.get("expected_behavior", ""),
                difficulty=c.get("difficulty", "medium"),
                category=c.get("category", "general"),
                source="synthetic",
            )
            for c in cases_raw
            if c.get("task_input") and c.get("expected_behavior")
        ]

        return split_examples(
            examples,
            seed=self.config.seed,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            holdout_ratio=self.config.holdout_ratio,
        )

    def generate_tool_selection(
        self,
        manifest: "ToolManifest",
        target_tool: str,
        num_cases: int,
    ) -> list[EvalExample]:
        """Generate a tool-selection eval dataset via three per-bucket LM calls.

        Buckets (when ``config.enable_confusable_bucket`` is True):
        - 50% target-correct: target tool is the unambiguous best choice.
        - 30% confusable-neighbor: target competes with the manifest's declared
          near-neighbor (read from manifest.confusable_neighbor_for(target)).
        - 20% regression-detection: some other tool is correct.

        Default (flag off): the confusable bucket's share rolls into
        target-correct, producing an 80/0/20 split. The bucket is opt-in;
        empirical evidence has not justified its share of the eval budget.

        Safety net with the flag on: if the manifest declares no confusable
        neighbor for the target, the bucket would interpolate "None" into the
        LM directive; we reallocate the same way and log a WARNING.

        Degenerate manifest (only the target tool): the other two buckets are
        skipped with a WARNING; all num_cases come from the target-correct bucket.

        Anti-trivial filter: tasks whose normalized lowercase text contains
        any tool's name are dropped. If a bucket loses >30% to the filter, it
        retries once with a "do not name tools" reminder.

        Raises RuntimeError if the combined result is empty.
        """
        other_tools = [t.name for t in manifest.tools if t.name != target_tool]
        if not other_tools:
            logger.warning(
                "generate_tool_selection: manifest contains only the target tool %r; "
                "skipping confusable and regression buckets",
                target_tool,
            )
            return self._call_bucket_with_filter(
                manifest, target_tool,
                bucket="target_correct", count=num_cases,
                previously_generated=[],
            )

        n_target = round(0.50 * num_cases)
        n_confusable = round(0.30 * num_cases)
        n_regression = num_cases - n_target - n_confusable

        if not self.config.enable_confusable_bucket and n_confusable > 0:
            logger.info(
                "confusable bucket disabled; reallocating %d examples to "
                "the target_correct bucket",
                n_confusable,
            )
            n_target += n_confusable
            n_confusable = 0
        elif manifest.confusable_neighbor_for(target_tool) is None and n_confusable > 0:
            # Flag on but the manifest declares no neighbor: the bucket would
            # interpolate "None" into the LM directive, producing garbage cases.
            logger.warning(
                "no confusable neighbor declared for target tool %r; "
                "reallocating %d examples to the target_correct bucket",
                target_tool,
                n_confusable,
            )
            n_target += n_confusable
            n_confusable = 0

        examples_target = self._call_bucket_with_filter(
            manifest, target_tool,
            bucket="target_correct", count=n_target,
            previously_generated=[],
        )
        examples_confusable = self._call_bucket_with_filter(
            manifest, target_tool,
            bucket="confusable_neighbor", count=n_confusable,
            previously_generated=[e.task_input for e in examples_target],
        )
        examples_regression = self._call_bucket_with_filter(
            manifest, target_tool,
            bucket="regression_detection", count=n_regression,
            previously_generated=[
                e.task_input for e in examples_target + examples_confusable
            ],
        )

        all_examples = examples_target + examples_confusable + examples_regression
        if not all_examples:
            raise RuntimeError(
                "synthetic dataset generator produced 0 examples; the manifest "
                "may have tool names that dominate the generated text"
            )
        return all_examples

    def _call_bucket_with_filter(
        self,
        manifest: "ToolManifest",
        target_tool: str,
        bucket: str,
        count: int,
        previously_generated: list[str],
    ) -> list[EvalExample]:
        """Call the LM for one bucket, filter trivial tasks, retry if too many drop."""
        if count <= 0:
            return []
        response = self._call_lm_for_bucket(
            manifest=manifest, target_tool=target_tool, bucket=bucket, count=count,
            previously_generated=previously_generated, with_reminder=False,
        )
        raw_tasks = response.get("tasks", [])
        filtered = self._filter_trivial_tasks(raw_tasks, manifest)
        if raw_tasks and len(filtered) < 0.7 * len(raw_tasks):
            # >30% dropped — retry with a reminder.
            response = self._call_lm_for_bucket(
                manifest=manifest, target_tool=target_tool, bucket=bucket, count=count,
                previously_generated=previously_generated, with_reminder=True,
            )
            filtered = self._filter_trivial_tasks(response.get("tasks", []), manifest)

        return [
            EvalExample(
                task_input=t["task"],
                expected_behavior=t["correct_tool"],
                category=bucket,
                source="synthetic",
            )
            for t in filtered
        ]

    def _call_lm_for_bucket(
        self,
        manifest: "ToolManifest",
        target_tool: str,
        bucket: str,
        count: int,
        previously_generated: list[str],
        with_reminder: bool,
    ) -> dict:
        """One per-bucket LM call.

        Tests mock this method directly to avoid LM dependence.
        """
        bucket_directives = {
            "target_correct": (
                f"All {count} tasks should have {target_tool!r} as the unambiguous correct choice."
            ),
            "confusable_neighbor": (
                f"All {count} tasks should have {target_tool!r} as the correct choice "
                f"but {manifest.confusable_neighbor_for(target_tool)!r} as a plausible-looking alternative."
            ),
            "regression_detection": (
                f"All {count} tasks should have a tool OTHER than {target_tool!r} as the "
                f"correct choice. Pick from: {[t.name for t in manifest.tools if t.name != target_tool]}"
            ),
        }
        reminder = (
            "\n\nIMPORTANT: do not name any tool by name in the task text. "
            "Tasks reference the action ('find files', 'read contents') not the tool name."
            if with_reminder else ""
        )
        anti_dup = (
            f"\n\nDo not produce tasks semantically similar to these: {previously_generated[:10]}"
            if previously_generated else ""
        )

        prompt = self._render_bucket_prompt(
            manifest=manifest, target_tool=target_tool,
            directive=bucket_directives[bucket],
            anti_dup_context=anti_dup, reminder=reminder, count=count,
        )

        _lm = self.config.get_lm("judge")
        lm = dspy.LM(
            _lm.model,
            **_lm.lm_kwargs,
            temperature=0.7,
            max_tokens=16000,
            request_timeout=120,
            num_retries=5,
        )
        with dspy.context(lm=lm):
            raw = lm(prompt=prompt)

        text = raw[0] if isinstance(raw, list) else raw
        return self._parse_bucket_json(text)

    @staticmethod
    def _parse_bucket_json(text: str) -> dict:
        """Parse a JSON object from a bucket LM response, tolerating prose around it."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(
                f"Could not parse bucket JSON from LLM output: {text[:200]}"
            )

    @staticmethod
    def _filter_trivial_tasks(
        tasks: list[dict],
        manifest: "ToolManifest",
    ) -> list[dict]:
        """Drop tasks whose text names any tool from the manifest.

        Both sides are normalized symmetrically before comparison: lowercased and
        with ``-`` replaced by ``_``. So ``read_file`` in the manifest matches a
        task that says ``read-file`` and vice versa.

        Matching uses word boundaries so the plural form (``read_files``) and the
        compound form (``read_filemap``) are NOT flagged as containing the tool
        name — only standalone occurrences are.
        """
        from evolution.tools.tool_source import _normalize_tool_name

        forbidden_names: set[str] = set()
        for t in manifest.tools:
            forbidden_names.add(_normalize_tool_name(t.name).lower())
        forbidden_names.discard("")

        # Pre-compile word-boundary patterns once per call.
        patterns = [re.compile(rf"\b{re.escape(name)}\b") for name in forbidden_names]

        kept = []
        for task in tasks:
            raw_text = task.get("task", "")
            normalized_text = raw_text.lower().replace("-", "_")
            if any(p.search(normalized_text) for p in patterns):
                continue
            kept.append(task)
        return kept

    @staticmethod
    def _render_bucket_prompt(
        manifest: "ToolManifest",
        target_tool: str,
        directive: str,
        anti_dup_context: str,
        reminder: str,
        count: int,
    ) -> str:
        """Render the per-bucket synthetic-generation prompt."""
        tool_list = "\n".join(f"- {t.name}: {t.description}" for t in manifest.tools)
        return (
            f"Generate {count} tool-selection tasks.\n\n"
            f"Available tools:\n{tool_list}\n\n"
            f"Bucket directive: {directive}"
            f"{anti_dup_context}{reminder}\n\n"
            f"Output JSON: {{\"tasks\": [{{\"task\": \"...\", \"correct_tool\": \"...\"}}, ...]}}"
        )


class GoldenDatasetLoader:
    """Load hand-curated evaluation datasets from JSONL files."""

    @staticmethod
    def load(path: Path, seed: int = 42) -> EvalDataset:
        """Load a golden dataset. If no splits exist, auto-split the single file."""
        if (path / "train.jsonl").exists():
            return EvalDataset.load(path)

        golden_file = path if path.suffix == ".jsonl" else path / "golden.jsonl"
        if not golden_file.exists():
            raise FileNotFoundError(f"No golden dataset found at {golden_file}")

        examples = []
        with open(golden_file) as f:
            for line in f:
                if line.strip():
                    examples.append(EvalExample.from_dict(json.loads(line)))

        # Golden ratios are not config-driven yet; preserve historical
        # 50/25/25 until anyone needs to tune them.
        return split_examples(
            examples,
            seed=seed,
            train_ratio=0.5,
            val_ratio=0.25,
            holdout_ratio=0.25,
        )
