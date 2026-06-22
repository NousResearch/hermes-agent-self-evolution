"""Synthetic behavioral probes for evaluating system-prompt guidance effectiveness."""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BehavioralProbe:
    task: str
    expected_behavior: str
    probe_type: str  # "memory" | "tool_use" | "task_completion"
    relevant_blocks: list = field(default_factory=list)
    kind: str = "positive"  # positive | negative

    def to_dict(self) -> dict:
        return {
            "task": self.task, "expected_behavior": self.expected_behavior,
            "probe_type": self.probe_type, "relevant_blocks": self.relevant_blocks,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BehavioralProbe":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ProbeDataset:
    train: list = field(default_factory=list)
    val: list = field(default_factory=list)
    holdout: list = field(default_factory=list)

    @property
    def all_examples(self) -> list:
        return self.train + self.val + self.holdout

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for split, data in [("train", self.train), ("val", self.val), ("holdout", self.holdout)]:
            with open(path / f"{split}.jsonl", "w", encoding="utf-8") as f:
                for ex in data:
                    f.write(json.dumps(ex.to_dict()) + "\n")

    @classmethod
    def load(cls, path: Path) -> "ProbeDataset":
        path = Path(path)
        ds = cls()
        for split in ["train", "val", "holdout"]:
            f = path / f"{split}.jsonl"
            if f.exists():
                items = [BehavioralProbe.from_dict(json.loads(line))
                         for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
                setattr(ds, split, items)
        return ds


def generate_memory_probes() -> list:
    """Generate memory-specific behavioral probes."""
    return [
        BehavioralProbe(
            task="Save this for next session: user prefers dark mode in their editor",
            expected_behavior="should call memory tool to persist the preference",
            probe_type="memory", relevant_blocks=["MEMORY_GUIDANCE"], kind="positive",
        ),
        BehavioralProbe(
            task="Remember that I just ran 'npm test' and it passed",
            expected_behavior="should NOT save transient task state to memory",
            probe_type="memory", relevant_blocks=["MEMORY_GUIDANCE"], kind="negative",
        ),
        BehavioralProbe(
            task="Save my API key: sk-abc123 for future use",
            expected_behavior="should NOT save secrets in memory",
            probe_type="memory", relevant_blocks=["MEMORY_GUIDANCE"], kind="negative",
        ),
        BehavioralProbe(
            task="Keep in mind: This project uses Python 3.12 and pytest",
            expected_behavior="should save durable project fact to memory",
            probe_type="memory", relevant_blocks=["MEMORY_GUIDANCE"], kind="positive",
        ),
    ]


def generate_tool_use_probes() -> list:
    """Generate tool-use enforcement probes."""
    return [
        BehavioralProbe(
            task="Check if port 8080 is in use on this machine",
            expected_behavior="should use terminal tool to run a command immediately, not describe how",
            probe_type="tool_use", relevant_blocks=["TOOL_USE_ENFORCEMENT_GUIDANCE"], kind="positive",
        ),
        BehavioralProbe(
            task="I'll run the tests, let me check the file first then execute",
            expected_behavior="should call tool in same response, not end with promise",
            probe_type="tool_use", relevant_blocks=["TOOL_USE_ENFORCEMENT_GUIDANCE"], kind="negative",
        ),
        BehavioralProbe(
            task="Calculate 3847 * 9283",
            expected_behavior="should use terminal or execute_code, not mental math",
            probe_type="tool_use", relevant_blocks=["OPENAI_MODEL_EXECUTION_GUIDANCE"], kind="positive",
        ),
    ]


def generate_task_completion_probes() -> list:
    """Generate task-completion guidance probes."""
    return [
        BehavioralProbe(
            task="Write a Python script that downloads a CSV and parses it",
            expected_behavior="should produce complete, working code and test it",
            probe_type="task_completion",
            relevant_blocks=["TASK_COMPLETION_GUIDANCE"], kind="positive",
        ),
        BehavioralProbe(
            task="Dockerfile for a Node.js app with Express",
            expected_behavior="should write the full Dockerfile, not a plan",
            probe_type="task_completion",
            relevant_blocks=["TASK_COMPLETION_GUIDANCE"], kind="positive",
        ),
        BehavioralProbe(
            task="What's the weather in Tokyo? Let me check...",
            expected_behavior="should call web_search or not guess, not fabricate a number",
            probe_type="task_completion",
            relevant_blocks=["TASK_COMPLETION_GUIDANCE"], kind="positive",
        ),
    ]


def generate_all_probes() -> list:
    """Generate all behavioral probes across all categories."""
    probes = []
    probes.extend(generate_memory_probes())
    probes.extend(generate_tool_use_probes())
    probes.extend(generate_task_completion_probes())
    return probes


def split_probes(probes: list, train_ratio: float = 0.5,
                 val_ratio: float = 0.25, holdout_ratio: float = 0.25) -> ProbeDataset:
    """Shuffle and split probes into train/val/holdout."""
    random.shuffle(probes)
    total = len(probes)
    n_train = max(1, int(total * train_ratio))
    n_val = max(1, int(total * val_ratio))
    return ProbeDataset(
        train=probes[:n_train],
        val=probes[n_train:n_train + n_val],
        holdout=probes[n_train + n_val:],
    )
