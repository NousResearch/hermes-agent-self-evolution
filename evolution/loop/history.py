"""Evolution run history — JSONL-based append-only log."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RunRecord:
    run_id: str
    phases: dict
    timestamp: str = ""
    git_sha: str = ""
    elapsed_seconds: int = 0

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id, "timestamp": self.timestamp,
            "phases": self.phases, "git_sha": self.git_sha,
            "elapsed_seconds": self.elapsed_seconds,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EvolutionHistory:
    runs: list = field(default_factory=list)

    def latest_per_phase(self) -> dict:
        """Return {phase_name: RunRecord} for the most recent run of each phase."""
        result = {}
        for run in self.runs:
            for phase in run.phases:
                result[phase] = run  # Later runs overwrite earlier ones
        return result


def load_history(history_path: Path) -> EvolutionHistory:
    """Load run history from a JSONL file. Returns empty history if file doesn't exist."""
    path = Path(history_path)
    if not path.exists():
        return EvolutionHistory()
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            records.append(RunRecord.from_dict(json.loads(stripped)))
    return EvolutionHistory(runs=records)


def append_run(history_path: Path, record: RunRecord) -> None:
    """Append a single run record to the JSONL file."""
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict()) + "\n")
