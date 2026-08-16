"""Objective verifier for the arxiv skill.

Grades outputs against verifiable bibliographic facts: arXiv IDs, exact
titles, first authors, and submission years for a fixed set of landmark
papers. Grading is pure Python (regex + normalization), so a fitness
evaluation costs nothing and two runs of the same output always agree.

The embedded facts are historical constants (published papers do not
change), and every one of them can be re-checked against the live arXiv
API at any time:

    python -m evolution.verifiers.arxiv_verifier --validate

Other useful entry points:

    python -m evolution.verifiers.arxiv_verifier --demo
    python -m evolution.verifiers.arxiv_verifier --build --num-cases 24

Task design notes:

  - ID and title lookups test factual recall the skill should route
    through the arXiv API rather than memory.
  - Year lookups are solvable from the arXiv ID alone (the YYMM prefix),
    so they directly measure whether the skill teaches the ID format.
    A skill edit can flip these from wrong to right, which is exactly
    the signal evolution needs.
  - Wrong answers and missing answers get different feedback, steering
    GEPA toward "verify with the API, do not guess".
"""

import random
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from evolution.core.dataset_builder import EvalDataset, EvalExample
from evolution.core.fitness import FitnessScore
from evolution.core.verifier import Verifier, register_verifier

console = Console()

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

# Ground-truth kinds (also used as EvalExample.category).
KIND_ID = "arxiv-id"
KIND_TITLE = "title"
KIND_AUTHOR = "first-author"
KIND_YEAR = "year"


@dataclass(frozen=True)
class Paper:
    """One seed paper with its verifiable facts."""

    arxiv_id: str
    title: str
    first_author: str

    @property
    def year(self) -> int:
        """Submission year, derived from the ID's YYMM prefix."""
        return 2000 + int(self.arxiv_id[:2])


# Landmark ML papers. Facts are stable (published papers do not change);
# run this module with --validate to cross-check them against the live
# arXiv API before trusting a new environment.
SEED_PAPERS: tuple[Paper, ...] = (
    Paper("1301.3781", "Efficient Estimation of Word Representations in Vector Space", "Tomas Mikolov"),
    Paper("1406.2661", "Generative Adversarial Networks", "Ian J. Goodfellow"),
    Paper("1412.6980", "Adam: A Method for Stochastic Optimization", "Diederik P. Kingma"),
    Paper("1503.02531", "Distilling the Knowledge in a Neural Network", "Geoffrey Hinton"),
    Paper("1512.03385", "Deep Residual Learning for Image Recognition", "Kaiming He"),
    Paper("1706.03762", "Attention Is All You Need", "Ashish Vaswani"),
    Paper("1810.04805", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "Jacob Devlin"),
    Paper("2005.14165", "Language Models are Few-Shot Learners", "Tom B. Brown"),
    Paper("2006.11239", "Denoising Diffusion Probabilistic Models", "Jonathan Ho"),
    Paper("2010.11929", "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", "Alexey Dosovitskiy"),
    Paper("2103.00020", "Learning Transferable Visual Models From Natural Language Supervision", "Alec Radford"),
    Paper("2106.09685", "LoRA: Low-Rank Adaptation of Large Language Models", "Edward J. Hu"),
    Paper("2201.11903", "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", "Jason Wei"),
    Paper("2203.02155", "Training language models to follow instructions with human feedback", "Long Ouyang"),
    Paper("2310.03714", "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines", "Omar Khattab"),
    Paper("2402.03300", "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", "Zhihong Shao"),
    Paper("2507.19457", "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning", "Lakshya A Agrawal"),
)

PAPER_BY_ID: dict[str, Paper] = {p.arxiv_id: p for p in SEED_PAPERS}


@dataclass(frozen=True)
class GroundTruth:
    """The authoritative answer for one task."""

    kind: str
    value: str
    paper_id: str


def task_input_for(paper: Paper, kind: str) -> str:
    """Build the exact task text for a paper and ground-truth kind.

    Module-level so tests (and other tooling) can reconstruct any task
    the verifier knows how to grade.
    """
    if kind == KIND_ID:
        return f'Find the arXiv ID of the paper titled "{paper.title}".'
    if kind == KIND_TITLE:
        return f"What is the exact title of arXiv paper {paper.arxiv_id}?"
    if kind == KIND_AUTHOR:
        return f'Who is the first author of the paper "{paper.title}" (arXiv {paper.arxiv_id})?'
    if kind == KIND_YEAR:
        return f"In which year was arXiv paper {paper.arxiv_id} first submitted?"
    raise ValueError(f"Unknown task kind: {kind}")


def _rubric_for(paper: Paper, kind: str) -> str:
    """Human-readable expected_behavior, kept compatible with judge and
    keyword scoring modes."""
    if kind == KIND_ID:
        return f"The response states the correct arXiv ID: {paper.arxiv_id}."
    if kind == KIND_TITLE:
        return f"The response states the paper's exact title: {paper.title}."
    if kind == KIND_AUTHOR:
        return f"The response names the first author: {paper.first_author}."
    return (
        f"The response states the submission year {paper.year}. The year is "
        f"derivable from the arXiv ID prefix {paper.arxiv_id[:4]} (YYMM format)."
    )


_DIFFICULTY = {
    KIND_ID: "hard",
    KIND_TITLE: "medium",
    KIND_AUTHOR: "medium",
    KIND_YEAR: "easy",
}


# ── Graders ───────────────────────────────────────────────────────────────
# Each returns (correctness, well_formed, feedback). correctness is the
# truth of the answer; well_formed is whether an answer of the expected
# shape was present at all.


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())


def grade_arxiv_id(expected: str, output: str) -> tuple[float, float, str]:
    unique = sorted(set(ARXIV_ID_RE.findall(output)))
    if not unique:
        return 0.0, 0.0, (
            f"No arXiv ID found in the response; expected {expected}. The skill "
            "should make the agent state the ID explicitly and confirm it with an "
            "arXiv API query (export.arxiv.org/api/query) instead of guessing."
        )
    if expected in unique:
        if len(unique) == 1:
            return 1.0, 1.0, f"Correct arXiv ID {expected}."
        # Half credit: hedging across several IDs is not a usable answer,
        # and full credit here would reward shotgunning candidates.
        return 0.5, 1.0, (
            f"The correct ID {expected} appears, but alongside other IDs "
            f"{unique}. The skill should make the agent commit to a single "
            "ID verified against the arXiv API."
        )
    return 0.0, 1.0, (
        f"Wrong arXiv ID: the response contained {unique} but the correct ID is "
        f"{expected}. The skill should instruct verifying candidate IDs against "
        "the arXiv API (export.arxiv.org/api/query) before answering."
    )


def grade_title(expected: str, output: str) -> tuple[float, float, str]:
    norm_expected = _normalize(expected)
    norm_output = _normalize(output)
    # The spaceless comparison accepts fused spelling variants such as
    # "Pretraining" for "Pre-training" while still requiring the exact
    # letter sequence of the full title.
    contains = norm_expected in norm_output or (
        norm_expected.replace(" ", "") in norm_output.replace(" ", "")
    )
    if norm_expected and contains:
        return 1.0, 1.0, f"Correct title: {expected!r}."
    tokens = [t for t in norm_expected.split() if len(t) > 2]
    if tokens:
        output_words = set(norm_output.split())
        recall = sum(t in output_words for t in tokens) / len(tokens)
        if recall >= 0.5:
            return round(recall, 3), 1.0, (
                f"Partial title match (token recall {recall:.0%}); the exact title "
                f"is {expected!r}. The skill should have the agent quote the title "
                "verbatim from the arXiv API's metadata rather than paraphrasing."
            )
    return 0.0, 0.0, (
        f"Title not found in the response; expected {expected!r}. The skill should "
        "route title lookups through the arXiv API (id_list query) and quote the "
        "returned title verbatim."
    )


def grade_first_author(expected: str, output: str) -> tuple[float, float, str]:
    last_name = expected.split()[-1]
    if re.search(rf"\b{re.escape(last_name)}\b", output, re.IGNORECASE):
        return 1.0, 1.0, f"Correct first author: {expected}."
    return 0.0, 0.0, (
        f"First author not named; expected {expected}. The skill should have the "
        "agent read the author list from the paper's arXiv metadata, where the "
        "first listed author is the first author."
    )


def grade_year(expected: str, output: str) -> tuple[float, float, str]:
    # Remove arXiv-ID-shaped substrings first so an ID like 2005.14165
    # does not have its YYMM prefix misread as the year 2005.
    cleaned = ARXIV_ID_RE.sub(" ", output)
    unique = sorted(set(YEAR_RE.findall(cleaned)))
    if not unique:
        return 0.0, 0.0, (
            f"No year found in the response; expected {expected}. arXiv IDs encode "
            "the submission year and month in their first four digits (YYMM), so "
            "the skill should teach reading the year straight from the ID."
        )
    if expected in unique:
        if len(unique) == 1:
            return 1.0, 1.0, f"Correct year {expected}."
        # Half credit for hedging across several years, same as for IDs.
        return 0.5, 1.0, (
            f"The correct year {expected} appears, but alongside other years "
            f"{unique}. The skill should make the agent commit to one year, "
            "which the arXiv ID's YYMM prefix determines exactly."
        )
    return 0.0, 1.0, (
        f"Wrong year: the response said {unique} but the correct year is "
        f"{expected}. The first four digits of an arXiv ID are YYMM, so the ID "
        "alone determines the submission year."
    )


_GRADERS = {
    KIND_ID: grade_arxiv_id,
    KIND_TITLE: grade_title,
    KIND_AUTHOR: grade_first_author,
    KIND_YEAR: grade_year,
}


def _conciseness(output: str) -> float:
    """1.0 up to 800 chars, linear decay to 0.0 at 4000 chars."""
    n = len(output)
    if n <= 800:
        return 1.0
    if n >= 4000:
        return 0.0
    return 1.0 - (n - 800) / 3200


# ── Verifier ──────────────────────────────────────────────────────────────


@register_verifier
class ArxivVerifier(Verifier):
    """Objective verifier for the arxiv skill."""

    skill_name = "arxiv"

    def __init__(self):
        self._answers: dict[str, GroundTruth] = {}
        self._tasks: list[EvalExample] = []
        self._tasks_by_paper: dict[str, list[EvalExample]] = {}
        self._generate_tasks()

    def _generate_tasks(self) -> None:
        """Register every (task, ground truth) pair for the seed papers.

        All tasks are registered regardless of dataset sampling, so any
        subset (including one reloaded from disk) can be graded.
        """
        for paper in SEED_PAPERS:
            for kind in (KIND_ID, KIND_TITLE, KIND_AUTHOR, KIND_YEAR):
                task = task_input_for(paper, kind)
                if kind == KIND_ID:
                    value = paper.arxiv_id
                elif kind == KIND_TITLE:
                    value = paper.title
                elif kind == KIND_AUTHOR:
                    value = paper.first_author
                else:
                    value = str(paper.year)
                self._answers[task] = GroundTruth(
                    kind=kind,
                    value=value,
                    paper_id=paper.arxiv_id,
                )
                example = EvalExample(
                    task_input=task,
                    expected_behavior=_rubric_for(paper, kind),
                    difficulty=_DIFFICULTY[kind],
                    category=kind,
                    source="verifier",
                )
                self._tasks.append(example)
                self._tasks_by_paper.setdefault(paper.arxiv_id, []).append(example)

    def ground_truth_for(self, task_input: str) -> GroundTruth | None:
        """The authoritative answer for a task, or None if unknown."""
        return self._answers.get(task_input.strip())

    def build_dataset(self, num_cases: int = 24, seed: int = 13) -> EvalDataset:
        """Build a deterministic 50/25/25 split without paper leakage.

        Each paper is assigned to exactly one split before task kinds are
        sampled. Otherwise an ID lookup for a paper can train the optimizer
        while a title or author lookup for the same paper appears to be an
        independent validation or holdout example.
        """
        if num_cases < 3:
            raise ValueError("num_cases must be at least 3 for train/val/holdout")

        rng = random.Random(seed)
        paper_ids = list(self._tasks_by_paper)
        rng.shuffle(paper_ids)

        n_papers = len(paper_ids)
        n_train_papers = max(1, int(n_papers * 0.5))
        n_val_papers = max(1, int(n_papers * 0.25))
        grouped_ids = (
            paper_ids[:n_train_papers],
            paper_ids[n_train_papers:n_train_papers + n_val_papers],
            paper_ids[n_train_papers + n_val_papers:],
        )

        pools: list[list[EvalExample]] = []
        for ids in grouped_ids:
            pool = [task for paper_id in ids for task in self._tasks_by_paper[paper_id]]
            rng.shuffle(pool)
            pools.append(pool)

        total = min(num_cases, len(self._tasks))
        target = [
            max(1, int(total * 0.5)),
            max(1, int(total * 0.25)),
        ]
        target.append(total - sum(target))
        counts = [min(wanted, len(pool)) for wanted, pool in zip(target, pools)]

        remaining = total - sum(counts)
        while remaining:
            progressed = False
            for index, pool in enumerate(pools):
                if counts[index] < len(pool):
                    counts[index] += 1
                    remaining -= 1
                    progressed = True
                    if not remaining:
                        break
            if not progressed:
                break

        dataset = EvalDataset(
            train=pools[0][:counts[0]],
            val=pools[1][:counts[1]],
            holdout=pools[2][:counts[2]],
        )
        self._assert_disjoint_papers(dataset)
        return dataset

    def _assert_disjoint_papers(self, dataset: EvalDataset) -> None:
        """Fail loudly if any paper reached more than one split.

        The grouping above makes leakage structurally impossible, which is
        exactly why this check is cheap and worth keeping: a later change to
        the sampling would otherwise reintroduce it silently, and a leaked
        holdout reports a score the run has not earned.
        """
        groups = {
            name: {
                self._answers[example.task_input].paper_id
                for example in getattr(dataset, name)
            }
            for name in ("train", "val", "holdout")
        }
        for left, right in (("train", "val"), ("train", "holdout"), ("val", "holdout")):
            shared = groups[left] & groups[right]
            if shared:
                raise AssertionError(
                    f"paper leakage between {left} and {right}: {sorted(shared)}"
                )

    def score(self, task_input: str, output: str) -> FitnessScore:
        truth = self.ground_truth_for(task_input)
        if truth is None:
            return FitnessScore(feedback=(
                "No ground truth registered for this task. ArxivVerifier can only "
                "grade tasks from its own dataset (see build_dataset)."
            ))
        if not output.strip():
            return FitnessScore(feedback="Empty response.")

        correctness, well_formed, feedback = _GRADERS[truth.kind](truth.value, output)
        # Conciseness only counts when an answer is actually present;
        # a brief evasion must not outscore a verbose correct answer.
        answered = correctness > 0 or well_formed > 0
        return FitnessScore(
            correctness=correctness,
            procedure_following=well_formed,
            conciseness=_conciseness(output) if answered else 0.0,
            feedback=feedback,
        )


# ── Live cross-check against the arXiv API ────────────────────────────────


def fetch_arxiv_metadata(arxiv_ids: list[str], timeout: int = 30) -> dict[str, dict]:
    """Fetch title, first author, and submission year for a batch of IDs."""
    url = f"{ARXIV_API}?id_list={','.join(arxiv_ids)}&max_results={len(arxiv_ids)}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        root = ET.fromstring(resp.read())

    metadata: dict[str, dict] = {}
    for entry in root.findall("atom:entry", ATOM_NS):
        raw_id = entry.findtext("atom:id", "", ATOM_NS)
        match = ARXIV_ID_RE.search(raw_id)
        if not match:
            continue
        authors = [
            a.findtext("atom:name", "", ATOM_NS)
            for a in entry.findall("atom:author", ATOM_NS)
        ]
        published = entry.findtext("atom:published", "", ATOM_NS)
        metadata[match.group(1)] = {
            "title": " ".join(entry.findtext("atom:title", "", ATOM_NS).split()),
            "first_author": authors[0] if authors else "",
            "year": int(published[:4]) if published[:4].isdigit() else 0,
        }
    return metadata


def validate_seed_papers() -> bool:
    """Cross-check every embedded fact against the live arXiv API."""
    ids = [p.arxiv_id for p in SEED_PAPERS]
    console.print(f"Fetching metadata for {len(ids)} papers from export.arxiv.org ...")
    live = fetch_arxiv_metadata(ids)

    table = Table(title="Embedded facts vs live arXiv API")
    table.add_column("arXiv ID")
    table.add_column("Title")
    table.add_column("First author")
    table.add_column("Year")

    all_ok = True
    for paper in SEED_PAPERS:
        entry = live.get(paper.arxiv_id)
        if entry is None:
            table.add_row(paper.arxiv_id, "[red]NOT FOUND[/red]", "", "")
            all_ok = False
            continue
        title_ok = _normalize(entry["title"]) == _normalize(paper.title)
        author_ok = _normalize(entry["first_author"]) == _normalize(paper.first_author)
        year_ok = entry["year"] == paper.year
        ok = lambda flag, live_val: "[green]✓[/green]" if flag else f"[red]✗ live: {live_val}[/red]"
        table.add_row(
            paper.arxiv_id,
            ok(title_ok, entry["title"]),
            ok(author_ok, entry["first_author"]),
            ok(year_ok, entry["year"]),
        )
        all_ok = all_ok and title_ok and author_ok and year_ok

    console.print(table)
    if all_ok:
        console.print(f"[bold green]All {len(SEED_PAPERS)} papers verified against the live arXiv API.[/bold green]")
    else:
        console.print("[bold red]Mismatches found. Fix SEED_PAPERS before trusting this verifier.[/bold red]")
    return all_ok


def run_demo() -> None:
    """Grade three canned outputs so the scoring behavior is visible."""
    verifier = ArxivVerifier()
    paper = PAPER_BY_ID["1706.03762"]
    task = task_input_for(paper, KIND_ID)
    console.print(f"\nTask: {task}\n")

    samples = [
        ("Correct and concise", "The paper is arXiv 1706.03762."),
        ("Confidently wrong", "That paper is arXiv 1706.03799."),
        ("No answer given", "That is a very famous transformer paper from Google Brain."),
    ]
    table = Table(title="ArxivVerifier demo")
    table.add_column("Output")
    table.add_column("Composite", justify="right")
    table.add_column("Feedback")
    for label, output in samples:
        fs = verifier.score(task, output)
        table.add_row(label, f"{fs.composite:.3f}", fs.feedback[:90] + "...")
    console.print(table)


@click.command()
@click.option("--validate", "do_validate", is_flag=True,
              help="Cross-check embedded facts against the live arXiv API")
@click.option("--demo", "do_demo", is_flag=True,
              help="Grade sample outputs to show the scoring behavior")
@click.option("--build", "do_build", is_flag=True,
              help="Build and save a verified eval dataset")
@click.option("--num-cases", default=24, help="Dataset size for --build")
@click.option("--output", default=None, help="Output directory for --build")
def main(do_validate, do_demo, do_build, num_cases, output):
    """Inspect, validate, or build datasets for the arxiv verifier."""
    if not any([do_validate, do_demo, do_build]):
        console.print("Nothing to do. Pass --validate, --demo, or --build.")
        return

    if do_validate and not validate_seed_papers():
        raise SystemExit(1)

    if do_demo:
        run_demo()

    if do_build:
        verifier = ArxivVerifier()
        dataset = verifier.build_dataset(num_cases=num_cases)
        out_dir = Path(output) if output else Path("datasets") / "skills" / "arxiv-verifier"
        dataset.save(out_dir)
        console.print(
            f"Saved {len(dataset.all_examples)} verified examples to {out_dir}/ "
            f"(train {len(dataset.train)} / val {len(dataset.val)} / holdout {len(dataset.holdout)})"
        )


if __name__ == "__main__":
    main()
