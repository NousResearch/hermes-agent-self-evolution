"""Routing evaluation primitives for competing Hermes skill descriptions.

Routing is evaluated against a catalog snapshot rather than an individual
skill.  This keeps discovery metrics reproducible and makes description
competition visible to the optimizer.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from evolution.skills.skill_module import load_skill


@dataclass
class RoutingExample:
    """A labelled user request for skill discovery evaluation."""

    query: str
    expected_skill: Optional[str]
    source: str = "golden"
    confidence: str = "reviewed"
    execution_success: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RoutingExample":
        return cls(
            query=data["query"],
            expected_skill=data.get("expected_skill"),
            source=data.get("source", "golden"),
            confidence=data.get("confidence", "reviewed"),
            execution_success=data.get("execution_success"),
            notes=data.get("notes", ""),
        )


@dataclass
class RoutingDataset:
    """Routing examples with explicit no-skill labels represented by ``None``."""

    examples: list[RoutingExample] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(example.to_dict()) for example in self.examples)
            + ("\n" if self.examples else "")
        )

    @classmethod
    def load(cls, path: Path) -> "RoutingDataset":
        examples = []
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    examples.append(RoutingExample.from_dict(json.loads(line)))
        return cls(examples)


def routing_examples_from_events(
    events: Iterable[dict],
    source: str = "sessiondb",
) -> list[RoutingExample]:
    """Convert normalized session signals into reviewable routing examples.

    Events without an explicit correction or no-skill signal are skipped rather
    than guessed. Callers can add ``expected_skill`` after human review.
    """
    examples = []
    for event in events:
        query = event.get("query") or event.get("task_input") or event.get("user_message")
        if not query:
            continue
        expected = event.get("expected_skill") or event.get("manual_skill")
        manual_command = event.get("manual_command") or ""
        if not expected and manual_command.startswith("/"):
            expected = manual_command[1:].split()[0]
        if expected:
            notes = "manual correction" if event.get("manual_skill") or manual_command else ""
            examples.append(
                RoutingExample(
                    query=query,
                    expected_skill=expected,
                    source=source,
                    confidence="inferred",
                    execution_success=event.get("execution_success"),
                    notes=notes,
                )
            )
        elif event.get("should_load_skill") is False:
            examples.append(
                RoutingExample(
                    query=query,
                    expected_skill=None,
                    source=source,
                    confidence="inferred",
                    notes="explicit no-skill signal",
                )
            )
    return examples


@dataclass(frozen=True)
class SkillCatalogEntry:
    """The immutable skill metadata supplied to a routing selector."""

    name: str
    description: str
    category: str = ""
    path: str = ""


@dataclass(frozen=True)
class SkillCatalog:
    """A reproducible snapshot of all descriptions visible to the selector."""

    entries: tuple[SkillCatalogEntry, ...]

    @property
    def names(self) -> set[str]:
        return {entry.name for entry in self.entries}

    @property
    def token_cost(self) -> int:
        # Hermes sends text, not tokenizer IDs. Four chars/token is a stable,
        # conservative estimate for comparing catalog variants.
        return sum(max(1, (len(entry.name) + len(entry.description) + 3) // 4) for entry in self.entries)

    def for_family(self, names: Iterable[str]) -> "SkillCatalog":
        selected = set(names)
        return SkillCatalog(tuple(entry for entry in self.entries if entry.name in selected))

    def with_descriptions(self, descriptions: dict[str, str]) -> "SkillCatalog":
        """Return a candidate snapshot with only descriptions changed."""
        return SkillCatalog(
            tuple(
                SkillCatalogEntry(
                    name=entry.name,
                    description=descriptions.get(entry.name, entry.description),
                    category=entry.category,
                    path=entry.path,
                )
                for entry in self.entries
            )
        )

    @classmethod
    def from_skills_dir(cls, skills_dir: Path) -> "SkillCatalog":
        entries = []
        for skill_path in sorted(skills_dir.rglob("SKILL.md")):
            skill = load_skill(skill_path)
            if not skill["name"]:
                continue
            entries.append(
                SkillCatalogEntry(
                    name=skill["name"],
                    description=skill["description"],
                    category=skill_path.parent.parent.name,
                    path=str(skill_path),
                )
            )
        return cls(tuple(entries))


@dataclass
class RoutingMetrics:
    """Metric vector used for Pareto comparison."""

    accuracy_at_1: float
    neighboring_false_positive_rate: float
    rejection_rate: float
    catalog_token_cost: int
    evaluated_examples: int
    per_skill_accuracy: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


Selector = Callable[[str, SkillCatalog], Optional[str]]


def evaluate_routing(
    dataset: RoutingDataset,
    catalog: SkillCatalog,
    selector: Selector,
    neighboring_skills: Optional[dict[str, set[str]]] = None,
) -> RoutingMetrics:
    """Evaluate a selector against a full catalog snapshot."""

    if not dataset.examples:
        return RoutingMetrics(0.0, 0.0, 0.0, catalog.token_cost, 0)

    correct = 0
    neighbor_false_positives = 0
    no_skill_count = 0
    no_skill_correct = 0
    skill_totals: dict[str, int] = defaultdict(int)
    skill_correct: dict[str, int] = defaultdict(int)

    for example in dataset.examples:
        selected = selector(example.query, catalog)
        expected = example.expected_skill
        if expected is not None:
            skill_totals[expected] += 1
            if selected == expected:
                correct += 1
                skill_correct[expected] += 1
            elif neighboring_skills and selected in neighboring_skills.get(expected, set()):
                neighbor_false_positives += 1
        else:
            no_skill_count += 1
            if selected is None:
                no_skill_correct += 1
                correct += 1

    total = len(dataset.examples)
    return RoutingMetrics(
        accuracy_at_1=correct / total,
        neighboring_false_positive_rate=neighbor_false_positives / max(1, total - no_skill_count),
        rejection_rate=no_skill_correct / max(1, no_skill_count),
        catalog_token_cost=catalog.token_cost,
        evaluated_examples=total,
        per_skill_accuracy={
            name: skill_correct[name] / count for name, count in skill_totals.items()
        },
    )


def group_skill_families(catalog: SkillCatalog) -> dict[str, SkillCatalog]:
    """Bootstrap families from the skill category directories."""

    grouped: dict[str, list[SkillCatalogEntry]] = defaultdict(list)
    for entry in catalog.entries:
        grouped[entry.category or "uncategorized"].append(entry)
    return {category: SkillCatalog(tuple(entries)) for category, entries in grouped.items()}


@dataclass(frozen=True)
class RoutingAcceptance:
    """Acceptance decision for a candidate catalog."""

    accepted: bool
    baseline_product: float
    candidate_product: float
    reason: str


def acceptance_gate(
    baseline: RoutingMetrics,
    candidate: RoutingMetrics,
    baseline_execution_success: float = 1.0,
    candidate_execution_success: float = 1.0,
    max_cost_increase: Optional[int] = None,
) -> RoutingAcceptance:
    """Require discovery × execution not to regress.

    Cost is an explicit Pareto objective: a candidate may only increase it when
    the caller supplies a permitted budget and the quality product improves.
    """

    baseline_product = baseline.accuracy_at_1 * baseline_execution_success
    candidate_product = candidate.accuracy_at_1 * candidate_execution_success
    quality_improved = candidate_product >= baseline_product
    cost_ok = (
        max_cost_increase is None
        or candidate.catalog_token_cost <= baseline.catalog_token_cost + max_cost_increase
    )
    accepted = quality_improved and cost_ok
    if not quality_improved:
        reason = "discovery-times-execution regressed"
    elif not cost_ok:
        reason = "catalog token cost exceeded budget"
    else:
        reason = "candidate is non-regressing on quality and cost"
    return RoutingAcceptance(accepted, baseline_product, candidate_product, reason)


def build_neighbor_map(catalog: SkillCatalog) -> dict[str, set[str]]:
    """Treat skills in the same bootstrap family as potential confusions."""
    families = group_skill_families(catalog)
    neighbors = {entry.name: set() for entry in catalog.entries}
    for family in families.values():
        names = {entry.name for entry in family.entries}
        for name in names:
            neighbors[name] = names - {name}
    return neighbors


def validate_descriptions(
    catalog: SkillCatalog,
    max_length: int = 500,
) -> list[str]:
    """Return human-readable validation failures for a candidate catalog."""
    failures = []
    seen_names: set[str] = set()
    for entry in catalog.entries:
        if not entry.name:
            failures.append("skill has an empty name")
        if entry.name in seen_names:
            failures.append(f"duplicate skill name: {entry.name}")
        seen_names.add(entry.name)
        if not entry.description.strip():
            failures.append(f"{entry.name}: description is empty")
        if len(entry.description) > max_length:
            failures.append(
                f"{entry.name}: description is {len(entry.description)} chars "
                f"(max {max_length})"
            )
    return failures


class PairwiseDescriptionMutator:
    """Build pairwise mutation prompts for a family-level optimizer."""

    def build_prompt(
        self,
        left: SkillCatalogEntry,
        right: SkillCatalogEntry,
        left_queries: Iterable[str] = (),
        right_queries: Iterable[str] = (),
    ) -> str:
        return (
            "Rewrite both skill descriptions to sharpen their boundary. Preserve "
            "what each skill actually does, keep each description concise, and add "
            "explicit anti-trigger guidance when useful. Do not move capabilities "
            "between skills.\n\n"
            f"SKILL A ({left.name}):\n{left.description}\n"
            f"Observed A queries:\n- " + "\n- ".join(left_queries) + "\n\n"
            f"SKILL B ({right.name}):\n{right.description}\n"
            f"Observed B queries:\n- " + "\n- ".join(right_queries)
        )

