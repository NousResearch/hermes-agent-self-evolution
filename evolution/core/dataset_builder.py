"""Evaluation dataset generation for hermes-agent-self-evolution.

Sources:
A) Synthetic generation — LLM reads a skill/tool/prompt and generates test cases
B) SessionDB mining — extract real usage patterns and score with LLM-as-judge
C) Golden sets — hand-curated JSONL files
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import dspy

from evolution.core.config import EvolutionConfig
from evolution.core.hermes_lm import make_lm


@dataclass
class EvalExample:
    """A single evaluation example."""
    task_input: str  # What the user asks
    expected_behavior: str  # Rubric — what a good response looks like
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "general"  # Category for stratified eval
    source: str = "synthetic"  # synthetic, sessiondb, golden
    id: str = ""  # Stable case id for reports and rubric debugging
    rubric_checks: list[dict] = field(default_factory=list)  # Deterministic strict-rubric checks

    def to_dict(self) -> dict:
        payload: dict[str, object] = {
            "task_input": self.task_input,
            "expected_behavior": self.expected_behavior,
            "difficulty": self.difficulty,
            "category": self.category,
            "source": self.source,
        }
        if self.id:
            payload["id"] = self.id
        if self.rubric_checks:
            payload["rubric_checks"] = self.rubric_checks
        return payload

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
                id=ex.id,
                task_input=ex.task_input,
                expected_behavior=ex.expected_behavior,
                difficulty=ex.difficulty,
                category=ex.category,
                source=ex.source,
                rubric_checks=ex.rubric_checks,
            ).with_inputs("task_input")
            for ex in data
        ]


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

        # Configure DSPy to use the judge model for generation
        lm = make_lm(self.config.judge_model, hermes_repo=str(self.config.hermes_agent_path))

        with dspy.context(lm=lm):
            result = self.generator(
                artifact_text=artifact_text,
                artifact_type=artifact_type,
                num_cases=n,
            )

        # Parse the generated test cases
        try:
            cases_raw = json.loads(result.test_cases)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
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

        # Shuffle and split
        random.shuffle(examples)
        n_total = len(examples)
        n_train = max(1, int(n_total * self.config.train_ratio))
        n_val = max(1, int(n_total * self.config.val_ratio))

        return EvalDataset(
            train=examples[:n_train],
            val=examples[n_train:n_train + n_val],
            holdout=examples[n_train + n_val:],
        )


class GoldenDatasetLoader:
    """Load hand-curated evaluation datasets from JSONL files."""

    _MANIFEST_FILENAME = "skill-fixtures.json"

    @staticmethod
    def default_golden_root() -> Path:
        """Return the repository's bundled golden dataset directory."""
        return Path(__file__).resolve().parents[2] / "datasets" / "golden"

    @classmethod
    def available_regression_fixtures(cls, root: Optional[Path] = None) -> dict[str, Path]:
        """Return skill-name to dataset-path mappings from the fixture manifest."""
        base = Path(root) if root is not None else cls.default_golden_root()
        manifest_path = base / cls._MANIFEST_FILENAME
        if not manifest_path.exists():
            return {}

        with open(manifest_path) as f:
            manifest = json.load(f)

        if not isinstance(manifest, dict):
            raise ValueError(f"Golden fixture manifest must be a JSON object: {manifest_path}")

        fixtures: dict[str, Path] = {}
        for skill_name, relative_path in manifest.items():
            if not isinstance(skill_name, str) or not isinstance(relative_path, str):
                raise ValueError(f"Golden fixture manifest values must be strings: {manifest_path}")
            if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
                raise ValueError(f"Golden fixture path must stay under {base}: {relative_path}")
            fixtures[skill_name] = base / relative_path
        return fixtures

    @classmethod
    def find_regression_fixture(cls, skill_name: str, root: Optional[Path] = None) -> Path:
        """Resolve a promoted regression fixture path for a skill name."""
        fixtures = cls.available_regression_fixtures(root)
        try:
            path = fixtures[skill_name]
        except KeyError as exc:
            available = ", ".join(sorted(fixtures)) or "none"
            raise FileNotFoundError(f"No promoted regression fixture for {skill_name!r}; available: {available}") from exc
        if not path.exists():
            raise FileNotFoundError(f"Promoted regression fixture missing: {path}")
        return path

    @classmethod
    def load_regression_fixture(cls, skill_name: str, root: Optional[Path] = None) -> EvalDataset:
        """Load the promoted regression fixture for a skill name."""
        return cls.load(cls.find_regression_fixture(skill_name, root))

    @staticmethod
    def load(path: Path) -> EvalDataset:
        """Load a golden dataset. If no splits exist, auto-split the single file."""
        if any((path / f"{split}.jsonl").exists() for split in ["train", "val", "holdout"]):
            return EvalDataset.load(path)

        # Single file — auto-split
        golden_file = path if path.suffix == ".jsonl" else path / "golden.jsonl"
        if not golden_file.exists():
            raise FileNotFoundError(f"No golden dataset found at {golden_file}")

        examples = []
        with open(golden_file) as f:
            for line in f:
                if line.strip():
                    examples.append(EvalExample.from_dict(json.loads(line)))

        random.shuffle(examples)
        n = len(examples)
        n_train = max(1, int(n * 0.5))
        n_val = max(1, int(n * 0.25))

        return EvalDataset(
            train=examples[:n_train],
            val=examples[n_train:n_train + n_val],
            holdout=examples[n_train + n_val:],
        )


def expand_objective_examples(dataset: EvalDataset) -> tuple[EvalDataset, dict]:
    """Expand train/val objective pressure from existing strict rubric checks.

    The expansion intentionally uses only train and validation rows. Holdout rows
    are copied unchanged so the strict final gate remains an unseen evaluation
    slice. Each generated row focuses on one deterministic rubric check while
    preserving the original row context.
    """

    expanded_train, added_train = _expand_split_objective_examples(dataset.train)
    expanded_val, added_val = _expand_split_objective_examples(dataset.val)
    expanded = EvalDataset(
        train=expanded_train,
        val=expanded_val,
        holdout=list(dataset.holdout),
    )
    metadata = {
        "enabled": True,
        "source_splits": ["train", "val"],
        "original_train_examples": len(dataset.train),
        "original_val_examples": len(dataset.val),
        "original_holdout_examples": len(dataset.holdout),
        "added_train_examples": added_train,
        "added_val_examples": added_val,
        "holdout_unchanged": expanded.holdout == dataset.holdout,
    }
    return expanded, metadata


def _expand_split_objective_examples(examples: list[EvalExample]) -> tuple[list[EvalExample], int]:
    expanded = list(examples)
    added = 0
    for example in examples:
        for check in example.rubric_checks:
            if not isinstance(check, dict):
                continue
            check_id = _safe_objective_check_id(check, added)
            description = str(check.get("description") or check.get("id") or "strict rubric check")
            cues = _objective_cues_from_check(check)
            expanded.append(
                EvalExample(
                    id=f"{example.id or 'example'}::objective::{check_id}",
                    task_input=(
                        f"{example.task_input}\n\n"
                        f"Objective focus: satisfy the strict rubric check `{check_id}` ({description}) "
                        "without weakening the rest of the review workflow."
                    ),
                    expected_behavior=(
                        f"{example.expected_behavior}\n\n"
                        f"Rubric focus: {description}. Preserve this requirement explicitly."
                        + (f" Useful literal cues: {', '.join(cues)}." if cues else "")
                    ),
                    difficulty=example.difficulty,
                    category=f"{example.category}:objective",
                    source="objective_expansion",
                    rubric_checks=[check],
                )
            )
            added += 1
    return expanded, added


def _safe_objective_check_id(check: dict, fallback_index: int) -> str:
    raw = str(check.get("id") or check.get("description") or f"check-{fallback_index}").strip().lower()
    safe = "".join(ch if ch.isalnum() or ch == "_" else "-" for ch in raw).strip("-")
    return safe or f"check-{fallback_index}"


def _objective_cues_from_check(check: dict) -> list[str]:
    cues: list[str] = []
    for key in ("pattern_all", "pattern_any"):
        patterns = check.get(key)
        if not isinstance(patterns, list | tuple):
            continue
        for pattern in patterns:
            if isinstance(pattern, str) and pattern:
                cues.append(_simplify_objective_pattern(pattern))
    return cues[:4]


def _simplify_objective_pattern(pattern: str) -> str:
    value = pattern.strip().strip("^").strip("$")
    for old, new in {
        r"\$": "$",
        r"\.": ".",
        r"\?": "?",
        r"\"": '"',
        r"\\": "",
        ".?": "",
        ".*": " ... ",
    }.items():
        value = value.replace(old, new)
    return " ".join(value.split())
