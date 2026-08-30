"""Behavioral test suite for system prompt sections.

A prompt section cannot be scored by reading it. ``MEMORY_GUIDANCE`` is good if
the agent saves the durable fact and leaves the PR number alone; it is bad if
it does the opposite. So the unit of evaluation here is a *scenario*: a user
message, the section it is aimed at, and a rubric for what the agent should
have done. Fitness is measured by running the agent and judging the behaviour,
never by judging the prose.

Two execution paths, in preference order:

**batch_runner (real).** hermes-agent's ``batch_runner.py`` takes
``--ephemeral_system_prompt``: a system prompt used for the run but not saved
to trajectories. That is exactly the shape Phase 3 needs - a candidate prompt
can be evaluated end to end, with real tools, without ever being written to
disk. The harness builds the JSONL dataset, invokes the runner as a
subprocess (never an import - it calls ``fire.Fire`` at module scope and owns
its own multiprocessing), and reads the trajectories back out of
``data/<run_name>/batch_*.jsonl``.

**Direct dspy (degraded).** With no hermes-agent checkout there is no agent to
run, so the section is exercised as the instruction block of a dspy module and
the model states which tools it would call. This measures whether the guidance
*reads* correctly, not whether the agent *behaves* correctly, and it is
labelled as such in the report so nobody mistakes one for the other.

Judging is likewise two-tier. :func:`heuristic_outcome` is deterministic,
offline, and cheap enough for the inner optimization loop: it checks the tools
that were and were not called, keyword coverage of the rubric, verbosity, and
platform formatting. :class:`BehavioralJudge` adds an LLM rubric grader for the
holdout, where a few dozen careful judgements are worth paying for.

The seed bank below is hand written for a reason. PLAN.md asks for 60-80
scenarios across five categories, and generated scenarios drift toward
restating the section they came from, which makes a prompt look good for
agreeing with itself. The seeds encode behaviour the section is supposed to
produce, including the negative cases - the ones where the correct move is to
do nothing.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import tempfile
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import dspy

__all__ = [
    "BehavioralScenario",
    "BehavioralOutcome",
    "BehavioralReport",
    "BehavioralSuite",
    "BehavioralJudge",
    "ScenarioGenerator",
    "AgentTranscript",
    "BatchRunnerHarness",
    "DirectPromptHarness",
    "SectionBehaviorModule",
    "BEHAVIORAL_CATEGORIES",
    "CATEGORY_SECTIONS",
    "SECTION_CATEGORIES",
    "SEED_SCENARIOS",
    "PASS_THRESHOLD",
    "build_scenarios",
    "heuristic_outcome",
    "make_behavioral_metric",
    "scenarios_to_dspy_examples",
    "select_harness",
    "build_section_instructions",
    "extract_section_text",
    "SECTION_MARKER_OPEN",
    "SECTION_MARKER_CLOSE",
]


PASS_THRESHOLD = 0.6

CATEGORY_SECTIONS: dict[str, str] = {
    "memory_guidance": "MEMORY_GUIDANCE",
    "session_search": "SESSION_SEARCH_GUIDANCE",
    "skills_guidance": "SKILLS_GUIDANCE",
    "identity_tone": "DEFAULT_AGENT_IDENTITY",
    "platform_formatting": "PLATFORM_HINTS",
}

BEHAVIORAL_CATEGORIES: tuple[str, ...] = tuple(CATEGORY_SECTIONS)

SECTION_CATEGORIES: dict[str, str] = {v: k for k, v in CATEGORY_SECTIONS.items()}


# ──────────────────────────────────────────────────────────────────────────
# Scenarios
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class BehavioralScenario:
    """One behavioural test: a prompt, the section it targets, and a rubric.

    ``expected_tools`` and ``forbidden_tools`` are what make the negative cases
    scoreable offline. "Does the agent save this?" is a rubric; "did it call
    ``memory``?" is a fact, and half the scenarios here are about *not* calling
    something.
    """

    prompt: str
    section_under_test: str
    expected_behavior: str
    category: str
    scenario_id: str = ""
    difficulty: str = "medium"
    platform: Optional[str] = None
    expected_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    source: str = "seed"

    def __post_init__(self) -> None:
        self.expected_tools = tuple(self.expected_tools)
        self.forbidden_tools = tuple(self.forbidden_tools)
        if not self.scenario_id:
            # A stable digest, not hash(): ids have to survive a save/load
            # round trip and match across processes.
            seed = f"{self.category}|{self.prompt}".encode("utf-8")
            digest = hashlib.blake2s(seed, digest_size=4).hexdigest()
            self.scenario_id = f"{self.category}-{digest}"

    def to_dict(self) -> dict:
        """Serialise the scenario, tool lists included."""
        data = asdict(self)
        data["expected_tools"] = list(self.expected_tools)
        data["forbidden_tools"] = list(self.forbidden_tools)
        return data

    @classmethod
    def from_dict(cls, data: Mapping) -> "BehavioralScenario":
        """Build from a stored mapping, ignoring keys this dataclass does not declare."""
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in fields})


def _s(
    category: str,
    prompt: str,
    expected: str,
    *,
    tools: Sequence[str] = (),
    forbidden: Sequence[str] = (),
    platform: Optional[str] = None,
    difficulty: str = "medium",
) -> BehavioralScenario:
    return BehavioralScenario(
        prompt=prompt,
        section_under_test=CATEGORY_SECTIONS[category],
        expected_behavior=expected,
        category=category,
        difficulty=difficulty,
        platform=platform,
        expected_tools=tuple(tools),
        forbidden_tools=tuple(forbidden),
    )


# 60 seeds: 12 per category, each category carrying at least two negative
# cases where the graded behaviour is restraint.
SEED_SCENARIOS: tuple[BehavioralScenario, ...] = (
    # ── memory guidance ──────────────────────────────────────────────────
    _s(
        "memory_guidance",
        "I always want Python formatted with black, line length 100.",
        "Saves the formatting preference to memory as a durable declarative fact.",
        tools=["memory"],
        difficulty="easy",
    ),
    _s(
        "memory_guidance",
        "The staging database only accepts connections over the VPN, never directly.",
        "Saves the environment detail to memory; it will still be true next month.",
        tools=["memory"],
    ),
    _s(
        "memory_guidance",
        "We merged PR 412 this morning and closed issue 88.",
        "Does not write PR or issue numbers to memory; they are stale within a week.",
        forbidden=["memory"],
    ),
    _s(
        "memory_guidance",
        "Call me Alex, not Alexandra.",
        "Saves the name preference to memory, phrased as a fact about the user.",
        tools=["memory"],
        difficulty="easy",
    ),
    _s(
        "memory_guidance",
        "Our test suite is pytest with xdist and it needs four workers to finish in time.",
        "Saves the project convention to memory as a declarative fact.",
        tools=["memory"],
    ),
    _s(
        "memory_guidance",
        "Save this: always run the linter before committing in this repo.",
        "Records the convention as a declarative fact rather than an imperative order "
        "to itself, or saves the procedure as a skill instead.",
        tools=["memory"],
    ),
    _s(
        "memory_guidance",
        "I am on an M2 Mac with 8 GB of RAM, so keep builds small.",
        "Saves the hardware constraint to memory; it shapes every future suggestion.",
        tools=["memory"],
    ),
    _s(
        "memory_guidance",
        "Today I closed three tickets and updated the changelog.",
        "Treats this as session chatter and saves nothing; completed-work logs do not "
        "belong in memory.",
        forbidden=["memory"],
    ),
    _s(
        "memory_guidance",
        "Stop using emoji in commit messages on this project. I have asked twice.",
        "Saves the repeated correction to memory; a recurring correction is the highest "
        "value thing memory can hold.",
        tools=["memory"],
    ),
    _s(
        "memory_guidance",
        "I fixed the flaky parser test at commit 9ab21f.",
        "Does not save the commit SHA to memory.",
        forbidden=["memory"],
    ),
    _s(
        "memory_guidance",
        "For anything customer facing, we use British English spelling.",
        "Saves the writing convention to memory.",
        tools=["memory"],
        difficulty="easy",
    ),
    _s(
        "memory_guidance",
        "Remember everything about this conversation so we can pick it up later.",
        "Explains that transcripts are recalled with session search rather than dumped "
        "into memory, and saves at most the durable facts.",
        difficulty="hard",
    ),
    # ── session search ───────────────────────────────────────────────────
    _s(
        "session_search",
        "Set the deploy up the same way we did last time.",
        "Searches past sessions for the earlier deploy before asking the user to repeat "
        "themselves.",
        tools=["session_search"],
        difficulty="easy",
    ),
    _s(
        "session_search",
        "What was that workaround we found for the certificate error?",
        "Searches session history for the certificate workaround.",
        tools=["session_search"],
        difficulty="easy",
    ),
    _s(
        "session_search",
        "Continue where we left off yesterday.",
        "Searches past sessions to recover the state before doing anything else.",
        tools=["session_search"],
    ),
    _s(
        "session_search",
        "Remind me what we decided about the retry policy.",
        "Searches session history rather than inventing a decision.",
        tools=["session_search"],
    ),
    _s(
        "session_search",
        "What is 17 percent of 340?",
        "Answers directly. There is no cross-session context to recover.",
        forbidden=["session_search"],
        difficulty="easy",
    ),
    _s(
        "session_search",
        "Use the same folder layout as the previous project.",
        "Searches for the previous project's layout before guessing.",
        tools=["session_search"],
    ),
    _s(
        "session_search",
        "You recommended a tool for diffing PDFs a while back. Which was it?",
        "Searches session history for the recommendation.",
        tools=["session_search"],
    ),
    _s(
        "session_search",
        "Write me a haiku about rain.",
        "Just writes the haiku; no history lookup is warranted.",
        forbidden=["session_search"],
        difficulty="easy",
    ),
    _s(
        "session_search",
        "Same as before, but point it at staging.",
        "Recovers what before was via session search, then adapts it to staging.",
        tools=["session_search"],
    ),
    _s(
        "session_search",
        "Did we ever finish the CSV importer?",
        "Searches sessions to answer, instead of saying it cannot know.",
        tools=["session_search"],
    ),
    _s(
        "session_search",
        "Pick up the parser refactor we started.",
        "Searches for the refactor's state before editing anything.",
        tools=["session_search"],
    ),
    _s(
        "session_search",
        "I know we hit this exact error before but I cannot find it.",
        "Searches session history for the error rather than re-debugging from scratch.",
        tools=["session_search"],
        difficulty="hard",
    ),
    # ── skills guidance ──────────────────────────────────────────────────
    _s(
        "skills_guidance",
        "Run the monthly invoice export.",
        "Checks for a matching skill and loads it before starting the work.",
        tools=["skill_view"],
    ),
    _s(
        "skills_guidance",
        "That took eleven tool calls but the CI cache is fixed now.",
        "Saves the workflow as a skill so the next occurrence is cheap.",
        tools=["skill_manage"],
    ),
    _s(
        "skills_guidance",
        "The skill you loaded says npm, but this repo uses pnpm.",
        "Patches the skill immediately rather than working around it silently.",
        tools=["skill_manage"],
    ),
    _s(
        "skills_guidance",
        "Do the release checklist.",
        "Loads the release skill before running any step of it.",
        tools=["skill_view"],
        difficulty="easy",
    ),
    _s(
        "skills_guidance",
        "The deploy skill shows [SKILL_PRUNED]. Ship the release.",
        "Re-reads the skill with skill_view before acting on it; a pruned placeholder is "
        "not usable content.",
        tools=["skill_view"],
        difficulty="hard",
    ),
    _s(
        "skills_guidance",
        "What is the capital of Peru?",
        "Answers the question. No skill machinery is involved.",
        forbidden=["skill_view", "skill_manage"],
        difficulty="easy",
    ),
    _s(
        "skills_guidance",
        "Set up a new FastAPI service with the scaffolding we normally use.",
        "Loads the scaffolding skill before generating anything.",
        tools=["skill_view"],
    ),
    _s(
        "skills_guidance",
        "The code review skill never mentions draft PRs, and we skip those.",
        "Patches the skill with the draft PR rule instead of only remembering it here.",
        tools=["skill_manage"],
    ),
    _s(
        "skills_guidance",
        "I keep having to explain how to run our integration tests.",
        "Saves the procedure as a skill so the explanation is not needed a third time.",
        tools=["skill_manage"],
    ),
    _s(
        "skills_guidance",
        "Reload it, then continue. Ignore the old pruned markers further up.",
        "Reloads once and then proceeds, treating leftover [SKILL_PRUNED] markers from "
        "earlier compaction as historical.",
        difficulty="hard",
    ),
    _s(
        "skills_guidance",
        "Just print the current date.",
        "Answers directly; a one-step task does not justify loading or saving a skill.",
        forbidden=["skill_manage"],
        difficulty="easy",
    ),
    _s(
        "skills_guidance",
        "We solved that certificate pinning problem with a five step workaround.",
        "Saves the workaround as a skill, since it was non-trivial and will recur.",
        tools=["skill_manage"],
    ),
    # ── identity and tone ────────────────────────────────────────────────
    _s(
        "identity_tone",
        "Is this SQL safe? SELECT * FROM users WHERE id = '1' OR 1=1",
        "Says plainly that it is not safe, names the injection, and shows the "
        "parameterised form. No preamble.",
        difficulty="easy",
    ),
    _s(
        "identity_tone",
        "Will this stock go up tomorrow?",
        "Admits it cannot know and does not invent a forecast.",
    ),
    _s(
        "identity_tone",
        "Give me a one line summary of what a mutex is.",
        "Returns one line. No headings, no follow-up offers.",
        difficulty="easy",
    ),
    _s(
        "identity_tone",
        "How many people used our app last month?",
        "States that it has no access to that data and asks for the source, rather than "
        "guessing a number.",
    ),
    _s(
        "identity_tone",
        "Which is better for this API, REST or gRPC?",
        "Picks one, gives the reason in a sentence or two, and names the condition that "
        "would change the answer.",
    ),
    _s(
        "identity_tone",
        "Thanks, that worked.",
        "Brief acknowledgement. No summary of what was just done, no new suggestions.",
        difficulty="easy",
    ),
    _s(
        "identity_tone",
        "Are you sure that is the right API version?",
        "Rechecks or says what it is unsure about, instead of either caving or doubling "
        "down.",
        difficulty="hard",
    ),
    _s(
        "identity_tone",
        "Explain quantum tunnelling to a ten year old.",
        "Plain language, concrete image, no jargon dump.",
    ),
    _s(
        "identity_tone",
        "Write a three item checklist for reviewing a pull request.",
        "Exactly three items and nothing else.",
        difficulty="easy",
    ),
    _s(
        "identity_tone",
        "Do you have access to my camera?",
        "Says no, plainly, and stops.",
        difficulty="easy",
    ),
    _s(
        "identity_tone",
        "This code is garbage. Fix it.",
        "Stays useful and direct: fixes what is wrong without apologising at length or "
        "matching the tone.",
    ),
    _s(
        "identity_tone",
        "Summarise this 400 line log file and tell me what broke.",
        "Leads with the cause, keeps the summary short, and flags anything it is "
        "inferring rather than reading.",
        difficulty="hard",
    ),
    # ── platform formatting ──────────────────────────────────────────────
    _s(
        "platform_formatting",
        "List the three largest files in this directory.",
        "Plain terminal text, no markdown table and no bold; the CLI does not render "
        "markdown.",
        platform="cli",
        difficulty="easy",
    ),
    _s(
        "platform_formatting",
        "Summarise what the error means.",
        "Plain text suitable for a terminal, no markdown headers.",
        platform="cli",
    ),
    _s(
        "platform_formatting",
        "Give me the config values as a table.",
        "Uses aligned plain text or labelled key: value lines instead of a markdown "
        "table.",
        platform="cli",
    ),
    _s(
        "platform_formatting",
        "Save the chart and send it to me.",
        "States the absolute path in plain text and does not emit a MEDIA: tag; the CLI "
        "has no attachment channel.",
        platform="cli",
        difficulty="hard",
    ),
    _s(
        "platform_formatting",
        "Schedule that job to run nightly and ping me with the result.",
        "Says that cron output is local only here and that delivery has to target a "
        "messaging platform.",
        platform="cli",
        difficulty="hard",
    ),
    _s(
        "platform_formatting",
        "Give me the deployment steps.",
        "Uses markdown formatting and bullet lists, which Telegram renders natively.",
        platform="telegram",
    ),
    _s(
        "platform_formatting",
        "Send me the generated logo.",
        "Includes a MEDIA:/absolute/path tag so the file arrives as a native photo.",
        platform="telegram",
    ),
    _s(
        "platform_formatting",
        "Compare the three plans side by side.",
        "Prefers bullet lists or labelled key: value pairs over a table, since tables do "
        "not render here.",
        platform="telegram",
    ),
    _s(
        "platform_formatting",
        "What does this stack trace mean?",
        "Uses an inline code block for the trace, which Telegram supports.",
        platform="telegram",
        difficulty="easy",
    ),
    _s(
        "platform_formatting",
        "Show me the pricing breakdown.",
        "Bullet list or key: value pairs, not a markdown table; WhatsApp does not render "
        "tables.",
        platform="whatsapp",
    ),
    _s(
        "platform_formatting",
        "Send over the invoice PDF.",
        "Includes a MEDIA:/absolute/path tag so the PDF arrives as a document.",
        platform="whatsapp",
    ),
    _s(
        "platform_formatting",
        "Write up the summary with headings.",
        "Uses markdown that converts cleanly to WhatsApp's native syntax, and avoids "
        "ANSI escape codes entirely.",
        platform="whatsapp",
        difficulty="hard",
    ),
)


def build_scenarios(
    sections: Sequence[str] = (),
    categories: Sequence[str] = (),
    per_category: Optional[int] = None,
    include_platform: bool = True,
) -> list[BehavioralScenario]:
    """Select seed scenarios for the sections or categories under test.

    With no filters this returns the whole bank. Filters are strict and
    intersecting: ask for a section and you get that section's scenarios.
    ``include_platform`` drops the platform formatting scenarios, which are
    scored against ``PLATFORM_HINTS`` - a section this phase cannot rewrite.
    They are worth keeping as a regression signal (a rewritten identity section
    must not start emitting markdown into a terminal), so a caller that wants
    them alongside another section asks for both by name.
    """
    wanted_sections = set(sections)
    wanted_categories = set(categories)

    selected: list[BehavioralScenario] = []
    for scenario in SEED_SCENARIOS:
        if scenario.category == "platform_formatting" and not include_platform:
            continue
        if wanted_categories and scenario.category not in wanted_categories:
            continue
        if wanted_sections and scenario.section_under_test not in wanted_sections:
            continue
        selected.append(scenario)

    if per_category is not None:
        capped: list[BehavioralScenario] = []
        counts: dict[str, int] = {}
        for scenario in selected:
            n = counts.get(scenario.category, 0)
            if n >= per_category:
                continue
            counts[scenario.category] = n + 1
            capped.append(scenario)
        selected = capped

    return selected


# ──────────────────────────────────────────────────────────────────────────
# LLM scenario generation
# ──────────────────────────────────────────────────────────────────────────


def _parse_json_array(raw: str) -> list[dict]:
    """Pull a JSON array out of an LLM response, tolerating prose around it."""
    text = (raw or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return []
    if isinstance(parsed, dict):
        parsed = parsed.get("scenarios", [])
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


class GenerateScenarios(dspy.Signature):
    """Generate behavioural test scenarios for one system prompt section.

    Each scenario is a realistic user message plus a rubric describing what the
    agent should do. Cover the negative cases too: messages where the correct
    behaviour is to NOT use the machinery this section describes. Return a JSON
    array where each object has: prompt, expected_behavior, difficulty
    (easy/medium/hard), expected_tools (array of tool names the agent should
    call, may be empty), forbidden_tools (array of tool names it must not call,
    may be empty).
    """

    section_name: str = dspy.InputField(desc="Name of the prompt section under test")
    section_text: str = dspy.InputField(desc="Current text of that section")
    category: str = dspy.InputField(desc="Behavioural category being tested")
    num_scenarios: int = dspy.InputField(desc="How many scenarios to produce")
    scenarios: str = dspy.OutputField(desc="JSON array of scenario objects")


class ScenarioGenerator:
    """Top the seed bank up with model-written scenarios.

    The predictor is injectable so this can be exercised without a language
    model. It is also constructed lazily, so building a generator never
    requires a configured LM.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        predictor=None,
    ) -> None:
        self.model = model
        self._predictor = predictor

    @property
    def predictor(self):
        """The scenario generator, built on first use so import stays cheap."""
        if self._predictor is None:
            self._predictor = dspy.ChainOfThought(GenerateScenarios)
        return self._predictor

    def generate(
        self,
        section_name: str,
        section_text: str,
        category: str,
        count: int = 6,
    ) -> list[BehavioralScenario]:
        """Ask the model for *count* scenarios; return the ones that parse."""
        kwargs = dict(
            section_name=section_name,
            section_text=section_text,
            category=category,
            num_scenarios=count,
        )
        if self.model:
            with dspy.context(lm=dspy.LM(self.model, temperature=0.0)):
                result = self.predictor(**kwargs)
        else:
            result = self.predictor(**kwargs)

        out: list[BehavioralScenario] = []
        for item in _parse_json_array(getattr(result, "scenarios", "")):
            prompt = str(item.get("prompt", "")).strip()
            expected = str(item.get("expected_behavior", "")).strip()
            if not prompt or not expected:
                continue
            out.append(
                BehavioralScenario(
                    prompt=prompt,
                    section_under_test=section_name,
                    expected_behavior=expected,
                    category=category,
                    difficulty=str(item.get("difficulty", "medium")),
                    expected_tools=tuple(item.get("expected_tools", ()) or ()),
                    forbidden_tools=tuple(item.get("forbidden_tools", ()) or ()),
                    source="generated",
                )
            )
        return out[:count]


# ──────────────────────────────────────────────────────────────────────────
# Transcripts
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class AgentTranscript:
    """What the agent did for one scenario, normalised across both harnesses."""

    prompt: str
    response: str
    tool_calls: tuple[str, ...] = ()
    scenario_id: str = ""
    completed: bool = True
    source: str = "batch_runner"
    raw: dict = field(default_factory=dict)

    def called(self, tool: str) -> bool:
        """True when *tool* appears in any recorded call, case-insensitively."""
        needle = tool.lower()
        return any(needle in call.lower() for call in self.tool_calls)

    def to_dict(self) -> dict:
        """Serialise the transcript for the run artifacts."""
        return {
            "prompt": self.prompt,
            "response": self.response,
            "tool_calls": list(self.tool_calls),
            "scenario_id": self.scenario_id,
            "completed": self.completed,
            "source": self.source,
        }


# ──────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class BehavioralOutcome:
    """The score for one scenario under one system prompt."""

    scenario_id: str
    category: str
    section: str
    score: float
    passed: bool
    feedback: str = ""
    # False when the harness produced nothing for this scenario. A run that
    # never happened is unmeasured, not a behavioural failure: scoring it 0.0
    # on one side of a paired comparison and a real score on the other
    # manufactures a difference out of a timeout. Six baseline scenarios timing
    # out while the candidate completes reads as a significant +32% lift.
    measured: bool = True
    judge: str = "heuristic"

    def to_dict(self) -> dict:
        """Serialise one scored outcome."""
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "section": self.section,
            "score": round(self.score, 4),
            "passed": self.passed,
            # Without this the artifact cannot tell a scenario that timed out
            # apart from one that ran and failed, and anything rebuilding an
            # outcome from the file gets the True default. That is the exact
            # confusion the field exists to prevent: a 0.0 that was never
            # measured, sitting on one side of a paired comparison, invents a
            # difference out of a timeout.
            "measured": self.measured,
            "feedback": self.feedback,
            "judge": self.judge,
        }


_MARKDOWN_MARKERS = (
    re.compile(r"\*\*[^*\n]+\*\*"),
    re.compile(r"^#{1,6}\s", re.MULTILINE),
    re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE),
    re.compile(r"^\s*\|?[-: ]*-{3,}[-: |]*\|?\s*$", re.MULTILINE),
)

_UNCERTAINTY_MARKERS = (
    "i do not know",
    "i don't know",
    "cannot know",
    "can't know",
    "not sure",
    "uncertain",
    "no access",
    "i am unable",
    "i'm unable",
    "would need",
)

_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the to
    with without not no does do it's this these those than then there their when which
    who whom what while will would should could may might must can rather instead into
    over under about after before again very just only also""".split()
)


def _keyword_coverage(expected: str, response: str) -> float:
    """Fraction of the rubric's content words that show up in the response."""
    words = {
        w
        for w in re.findall(r"[a-z_]{3,}", expected.lower())
        if w not in _STOPWORDS
    }
    if not words:
        return 0.5
    seen = set(re.findall(r"[a-z_]{3,}", response.lower()))
    return len(words & seen) / len(words)


def heuristic_outcome(
    scenario: BehavioralScenario, transcript: AgentTranscript
) -> BehavioralOutcome:
    """Score a scenario deterministically, with no model in the loop.

    This is the inner-loop metric. It is blunt on purpose: tool discipline is
    checked exactly, everything else is a proxy. The point is a signal that is
    free, reproducible, and impossible to game by writing a longer prompt.
    """
    notes: list[str] = []
    response = transcript.response or ""

    if not response.strip() and not transcript.tool_calls:
        return BehavioralOutcome(
            scenario_id=scenario.scenario_id,
            category=scenario.category,
            section=scenario.section_under_test,
            score=0.0,
            passed=False,
            feedback="empty transcript: the agent produced neither text nor tool calls",
        )

    # Tool discipline dominates when the scenario declares any.
    tool_score: Optional[float] = None
    if scenario.expected_tools or scenario.forbidden_tools:
        hits = sum(1 for t in scenario.expected_tools if transcript.called(t))
        misses = [t for t in scenario.expected_tools if not transcript.called(t)]
        violations = [t for t in scenario.forbidden_tools if transcript.called(t)]

        if scenario.expected_tools:
            tool_score = hits / len(scenario.expected_tools)
            if misses:
                notes.append(f"did not call {', '.join(misses)}")
        else:
            tool_score = 1.0
        if violations:
            tool_score = min(tool_score, 0.15)
            notes.append(f"called forbidden {', '.join(violations)}")

    coverage = _keyword_coverage(scenario.expected_behavior, response)
    quality = 0.35 + 0.65 * coverage

    # Identity scenarios are partly about restraint, so length is a real signal.
    if scenario.category == "identity_tone":
        if len(response) > 2500:
            quality *= 0.6
            notes.append("response is long for a direct answer")
        rubric = scenario.expected_behavior.lower()
        wants_uncertainty = any(
            k in rubric for k in ("admit", "cannot know", "unsure", "no access", "guess")
        )
        if wants_uncertainty:
            if any(m in response.lower() for m in _UNCERTAINTY_MARKERS):
                quality = min(1.0, quality + 0.25)
            else:
                quality *= 0.5
                notes.append("did not acknowledge what it cannot know")

    # Platform formatting: markdown in a terminal is a concrete failure.
    if scenario.category == "platform_formatting":
        rubric = scenario.expected_behavior.lower()
        plain_required = "no markdown" in rubric or "plain" in rubric
        if plain_required:
            if any(p.search(response) for p in _MARKDOWN_MARKERS):
                quality *= 0.4
                notes.append("emitted markdown where plain text was required")

        forbids_media = "not emit a media" in rubric or "no media" in rubric
        if forbids_media:
            if "MEDIA:" in response:
                quality *= 0.4
                notes.append("emitted a MEDIA: tag on a platform that cannot use one")
        elif "media:" in rubric:
            if "MEDIA:" in response:
                quality = min(1.0, quality + 0.2)
            else:
                quality *= 0.6
                notes.append("no MEDIA: tag, so nothing is delivered")

    score = quality if tool_score is None else (0.6 * tool_score + 0.4 * quality)
    score = max(0.0, min(1.0, score))

    return BehavioralOutcome(
        scenario_id=scenario.scenario_id,
        category=scenario.category,
        section=scenario.section_under_test,
        score=score,
        passed=score >= PASS_THRESHOLD,
        feedback="; ".join(notes) if notes else "behaved as the rubric describes",
    )


class JudgeBehavior(dspy.Signature):
    """Judge whether an agent behaved as a rubric requires.

    Score only the behaviour: which tools were used, what the response did, and
    whether it matches the rubric. Do not reward eloquence, length, or restating
    the instructions. Return a score from 0.0 to 1.0 and one sentence of
    specific, actionable feedback naming what to change.
    """

    user_message: str = dspy.InputField(desc="What the user asked")
    expected_behavior: str = dspy.InputField(desc="Rubric for correct behaviour")
    category: str = dspy.InputField(desc="Behavioural category under test")
    agent_response: str = dspy.InputField(desc="The agent's final response")
    tool_calls: str = dspy.InputField(desc="Comma separated tools the agent called")
    score: float = dspy.OutputField(desc="0.0 to 1.0, how well the behaviour matched")
    feedback: str = dspy.OutputField(desc="One sentence naming what to change")


class BehavioralJudge:
    """Rubric grader with an LLM tier and a deterministic tier.

    The LLM tier is for the holdout, where the extra judgement is worth the
    cost. Any failure falls back to the heuristic and says so in the outcome,
    so a rate limit degrades the signal instead of destroying the run.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        predictor=None,
        use_llm: bool = True,
    ) -> None:
        self.model = model
        self.use_llm = use_llm
        self._predictor = predictor

    @property
    def predictor(self):
        """The judge, built on first use so import stays cheap."""
        if self._predictor is None:
            self._predictor = dspy.ChainOfThought(JudgeBehavior)
        return self._predictor

    def score(
        self, scenario: BehavioralScenario, transcript: AgentTranscript
    ) -> BehavioralOutcome:
        """Score one transcript against its scenario, by judge or by heuristic."""
        if not self.use_llm:
            return heuristic_outcome(scenario, transcript)

        kwargs = dict(
            user_message=scenario.prompt,
            expected_behavior=scenario.expected_behavior,
            category=scenario.category,
            agent_response=transcript.response,
            tool_calls=", ".join(transcript.tool_calls) or "none",
        )
        try:
            if self.model:
                with dspy.context(lm=dspy.LM(self.model, temperature=0.0)):
                    result = self.predictor(**kwargs)
            else:
                result = self.predictor(**kwargs)
        except Exception as exc:  # noqa: BLE001 - degrade, never abort a run
            fallback = heuristic_outcome(scenario, transcript)
            fallback.feedback = f"{fallback.feedback} (llm judge unavailable: {exc})"
            return fallback

        score = _parse_float(getattr(result, "score", None))
        return BehavioralOutcome(
            scenario_id=scenario.scenario_id,
            category=scenario.category,
            section=scenario.section_under_test,
            score=score,
            passed=score >= PASS_THRESHOLD,
            feedback=str(getattr(result, "feedback", "")),
            judge="llm",
        )

    def score_all(
        self,
        scenarios: Sequence[BehavioralScenario],
        transcripts: Mapping[str, AgentTranscript],
    ) -> "BehavioralReport":
        """Score every scenario, recording a miss where no transcript arrived."""
        outcomes = []
        for scenario in scenarios:
            transcript = transcripts.get(scenario.scenario_id)
            if transcript is None:
                outcomes.append(
                    BehavioralOutcome(
                        scenario_id=scenario.scenario_id,
                        category=scenario.category,
                        section=scenario.section_under_test,
                        score=0.0,
                        passed=False,
                        feedback="no transcript produced for this scenario",
                        measured=False,
                    )
                )
                continue
            outcomes.append(self.score(scenario, transcript))
        return BehavioralReport(outcomes=outcomes)


def _parse_float(value) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    try:
        return max(0.0, min(1.0, float(str(value).strip())))
    except (TypeError, ValueError):
        return 0.5


@dataclass
class BehavioralReport:
    """Aggregated behavioural scores for one system prompt."""

    outcomes: list[BehavioralOutcome] = field(default_factory=list)
    harness: str = "heuristic"

    def __len__(self) -> int:
        return len(self.outcomes)

    @property
    def mean_score(self) -> float:
        """Mean score across every outcome, 0.0 when there are none."""
        if not self.outcomes:
            return 0.0
        return sum(o.score for o in self.outcomes) / len(self.outcomes)

    @property
    def pass_rate(self) -> float:
        """Fraction of outcomes that passed, 0.0 when there are none."""
        if not self.outcomes:
            return 0.0
        return sum(1 for o in self.outcomes if o.passed) / len(self.outcomes)

    def by_category(self) -> dict[str, float]:
        """Mean score per scenario category."""
        buckets: dict[str, list[float]] = {}
        for outcome in self.outcomes:
            buckets.setdefault(outcome.category, []).append(outcome.score)
        return {k: sum(v) / len(v) for k, v in sorted(buckets.items())}

    def by_section(self) -> dict[str, float]:
        """Mean score per prompt section under test."""
        buckets: dict[str, list[float]] = {}
        for outcome in self.outcomes:
            buckets.setdefault(outcome.section, []).append(outcome.score)
        return {k: sum(v) / len(v) for k, v in sorted(buckets.items())}

    def worst(self, n: int = 5) -> list[BehavioralOutcome]:
        """The *n* lowest scoring outcomes, worst first."""
        return sorted(self.outcomes, key=lambda o: o.score)[:n]

    def to_dict(self) -> dict:
        """Serialise the report with its per-category and per-section breakdowns."""
        return {
            "harness": self.harness,
            "n": len(self.outcomes),
            "mean_score": round(self.mean_score, 4),
            "pass_rate": round(self.pass_rate, 4),
            "by_category": {k: round(v, 4) for k, v in self.by_category().items()},
            "by_section": {k: round(v, 4) for k, v in self.by_section().items()},
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


# ──────────────────────────────────────────────────────────────────────────
# The dspy module under optimization
# ──────────────────────────────────────────────────────────────────────────

SECTION_MARKER_OPEN = "<<<SECTION>>>"
SECTION_MARKER_CLOSE = "<<<END SECTION>>>"


def build_section_instructions(section_name: str, section_text: str) -> str:
    """Wrap a section as the instruction block of a dspy signature.

    The section text *is* the optimizable parameter, so it has to live where
    the optimizer writes: the signature instructions. The markers make the
    section recoverable afterwards, which is the difference between an
    optimizer run that produces a deployable section and one that produces a
    module nobody can extract anything from.
    """
    return (
        f"You are Hermes Agent. The system prompt section under test is "
        f"{section_name}. Follow it exactly.\n"
        f"{SECTION_MARKER_OPEN}\n{section_text}\n{SECTION_MARKER_CLOSE}\n"
        "Respond to the user message as the agent would. In tool_calls, list the "
        "names of the tools you would call, comma separated, or 'none'."
    )


def extract_section_text(instructions: str, section_name: str = "") -> str:
    """Recover the section text from an instruction block.

    Falls back to the whole instruction string when the markers are gone, which
    is what happens if the optimizer rewrites the scaffolding as well as the
    section. Losing the markers is not a reason to lose the candidate; the
    growth and identity constraints still get to judge it.
    """
    start = instructions.find(SECTION_MARKER_OPEN)
    end = instructions.find(SECTION_MARKER_CLOSE)
    if start != -1 and end != -1 and end > start:
        return instructions[start + len(SECTION_MARKER_OPEN) : end].strip("\n")
    return instructions.strip()


class SectionBehavior(dspy.Signature):
    """Answer a user message under a system prompt section."""

    user_message: str = dspy.InputField(desc="The user's message")
    response: str = dspy.OutputField(desc="The agent's reply")
    tool_calls: str = dspy.OutputField(
        desc="Comma separated tool names you would call, or 'none'"
    )


class SectionBehaviorModule(dspy.Module):
    """A prompt section as an optimizable dspy module.

    Unlike the Phase 1 skill module, the text is not an input field: it is the
    signature's instructions. That is deliberate. GEPA mutates instructions, so
    putting the section there means the optimizer is editing the artifact we
    intend to ship, and :attr:`section_text` reads the evolved version back out.
    """

    def __init__(self, section_text: str, section_name: str, predictor=None) -> None:
        super().__init__()
        self.section_name = section_name
        self._baseline_text = section_text
        signature = SectionBehavior.with_instructions(
            build_section_instructions(section_name, section_text)
        )
        self.predictor = predictor or dspy.ChainOfThought(signature)

    @property
    def section_text(self) -> str:
        """The current section text, read back out of the live instructions."""
        for _, predictor in self.named_predictors():
            signature = getattr(predictor, "signature", None)
            instructions = getattr(signature, "instructions", None)
            if instructions:
                return extract_section_text(instructions, self.section_name)
        return self._baseline_text

    def forward(self, user_message: str) -> dspy.Prediction:
        """Run the agent for one user message, returning its reply and tool calls."""
        result = self.predictor(user_message=user_message)
        return dspy.Prediction(
            response=getattr(result, "response", ""),
            tool_calls=getattr(result, "tool_calls", ""),
        )


def _split_tool_calls(raw: str) -> tuple[str, ...]:
    if not raw:
        return ()
    cleaned = raw.strip()
    if cleaned.lower() in {"none", "n/a", "-", "[]"}:
        return ()
    parts = re.split(r"[,\n;]+", cleaned)
    return tuple(p.strip().strip("`'\"()") for p in parts if p.strip())


def scenarios_to_dspy_examples(
    scenarios: Sequence[BehavioralScenario],
) -> list[dspy.Example]:
    """Turn scenarios into dspy examples that keep their rubric and tool lists."""
    return [
        dspy.Example(
            user_message=s.prompt,
            expected_behavior=s.expected_behavior,
            category=s.category,
            section_under_test=s.section_under_test,
            scenario_id=s.scenario_id,
            expected_tools=list(s.expected_tools),
            forbidden_tools=list(s.forbidden_tools),
            platform=s.platform or "",
        ).with_inputs("user_message")
        for s in scenarios
    ]


def _example_to_scenario(example) -> BehavioralScenario:
    return BehavioralScenario(
        prompt=getattr(example, "user_message", ""),
        section_under_test=getattr(example, "section_under_test", ""),
        expected_behavior=getattr(example, "expected_behavior", ""),
        category=getattr(example, "category", "general"),
        scenario_id=getattr(example, "scenario_id", ""),
        platform=getattr(example, "platform", "") or None,
        expected_tools=tuple(getattr(example, "expected_tools", ()) or ()),
        forbidden_tools=tuple(getattr(example, "forbidden_tools", ()) or ()),
    )


def make_behavioral_metric(judge: Optional[BehavioralJudge] = None):
    """Build the metric GEPA optimizes against.

    Defaults to the heuristic scorer. An optimizer run touches every training
    example on every iteration, so paying a judge model there costs more than
    it buys; the LLM judge earns its keep on the holdout.
    """

    def metric(example, prediction, trace=None, *args, **kwargs) -> float:
        """DSPy metric that turns a prediction into a scored behavioural outcome."""
        scenario = _example_to_scenario(example)
        transcript = AgentTranscript(
            prompt=scenario.prompt,
            response=getattr(prediction, "response", "") or "",
            tool_calls=_split_tool_calls(getattr(prediction, "tool_calls", "") or ""),
            scenario_id=scenario.scenario_id,
            source="dspy",
        )
        if judge is None:
            return heuristic_outcome(scenario, transcript).score
        return judge.score(scenario, transcript).score

    return metric


# ──────────────────────────────────────────────────────────────────────────
# Harnesses
# ──────────────────────────────────────────────────────────────────────────


class HarnessTimeout(RuntimeError):
    """batch_runner did not finish in time, so nothing was measured."""


class HarnessFailure(RuntimeError):
    """batch_runner exited non-zero and produced nothing to score.

    Distinct from :class:`HarnessTimeout` because the operator's next move is
    different: a timeout wants a longer budget or fewer scenarios, a crash
    wants the runner's own error read (a missing API key, a bad flag).
    """


@dataclass
class BatchRunnerHarness:
    """Run scenarios through hermes-agent's batch_runner as a subprocess.

    ``--ephemeral_system_prompt`` is the whole reason this path exists: the
    candidate prompt drives the run but is never written to the repo or saved
    into a trajectory. batch_runner is invoked, never imported - it calls
    ``fire.Fire`` at module scope and forks its own worker pool, so importing it
    would hand this process someone else's process tree.
    """

    hermes_repo: Path
    model: str = "anthropic/claude-sonnet-4.6"
    max_turns: int = 6
    num_workers: int = 4
    batch_size: int = 8
    python: str = sys.executable
    timeout: int = 3600
    dataset_dir: Optional[Path] = None

    @property
    def runner_path(self) -> Path:
        """Where the hermes-agent batch runner is expected to live."""
        return Path(self.hermes_repo) / "batch_runner.py"

    @property
    def available(self) -> bool:
        """True when the batch runner exists and can be invoked."""
        return self.runner_path.is_file()

    def build_dataset(
        self, scenarios: Sequence[BehavioralScenario], path: Path
    ) -> Path:
        """Write the JSONL batch_runner reads. One object per line, 'prompt' key."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for scenario in scenarios:
                handle.write(
                    json.dumps(
                        {
                            "prompt": scenario.prompt,
                            "scenario_id": scenario.scenario_id,
                            "category": scenario.category,
                            "section_under_test": scenario.section_under_test,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return path

    def build_command(
        self,
        dataset_file: Path,
        run_name: str,
        system_prompt: str,
        sample_count: Optional[int] = None,
        resume: bool = False,
    ) -> list[str]:
        """Assemble the batch_runner argv.

        batch_runner is a ``fire`` CLI, so flags are ``--name=value`` with
        underscores, matching its ``main()`` signature exactly.
        """
        cmd = [
            self.python,
            "batch_runner.py",
            f"--dataset_file={dataset_file}",
            f"--batch_size={self.batch_size}",
            f"--run_name={run_name}",
            f"--model={self.model}",
            f"--max_turns={self.max_turns}",
            f"--num_workers={self.num_workers}",
            f"--ephemeral_system_prompt={system_prompt}",
        ]
        if sample_count is not None:
            cmd.append(f"--max_samples={sample_count}")
        if resume:
            cmd.append("--resume")
        return cmd

    def output_dir(self, run_name: str) -> Path:
        """batch_runner writes to data/<run_name>/ relative to its own repo."""
        return Path(self.hermes_repo) / "data" / run_name

    def parse_results(self, run_name: str) -> list[AgentTranscript]:
        """Read batch_*.jsonl trajectories back into transcripts."""
        out_dir = self.output_dir(run_name)
        transcripts: list[AgentTranscript] = []
        if not out_dir.is_dir():
            return transcripts

        for batch_file in sorted(out_dir.glob("batch_*.jsonl")):
            for line in batch_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("failed"):
                    continue
                transcripts.append(_transcript_from_entry(entry))
        return transcripts

    def run(
        self,
        scenarios: Sequence[BehavioralScenario],
        system_prompt: str,
        run_name: str,
        dataset_file: Optional[Path] = None,
    ) -> list[AgentTranscript]:
        """Build the dataset, invoke batch_runner, and read the results back."""
        if not self.available:
            raise FileNotFoundError(f"no batch_runner.py at {self.runner_path}")

        # Default the scratch dataset OUT of the operator's checkout. It used
        # to land in <hermes_repo>/data/_hase, so a --no-write run left
        # untracked files behind in a repo the README promises it does not
        # touch. Callers pass their own run directory; the temp dir is only the
        # fallback. (batch_runner itself still writes its results under the
        # repo's data/ because it runs with cwd there - that is hermes-agent's
        # own layout, not something this harness chooses.)
        base = Path(
            self.dataset_dir
            or Path(tempfile.gettempdir()) / "hase-behavioral"
        )
        base.mkdir(parents=True, exist_ok=True)
        dataset_file = Path(dataset_file or (base / f"{run_name}.jsonl"))
        self.build_dataset(scenarios, dataset_file)

        cmd = self.build_command(
            dataset_file=dataset_file,
            run_name=run_name,
            system_prompt=system_prompt,
            sample_count=len(scenarios),
        )
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(self.hermes_repo),
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            # A hung batch_runner used to take the whole run down with a
            # traceback and no metrics.json. The scenarios it did not answer
            # come back unmeasured, which the paired comparison already knows
            # how to drop rather than score as failures.
            raise HarnessTimeout(
                f"batch_runner did not finish within {self.timeout}s; "
                f"no transcripts were collected for {run_name}"
            ) from exc

        transcripts = self.parse_results(run_name)
        if completed.returncode != 0 and not transcripts:
            # Discarding the exit status here lost the cause: every scenario
            # came back unmeasured and the failure only surfaced much later as
            # an unpaired holdout. A crashed runner and a runner that answered
            # none of the prompts are different problems, and only the exit
            # code tells them apart.
            raise HarnessFailure(
                f"batch_runner exited {completed.returncode} and produced no "
                f"transcripts for {run_name}; its own output has the cause"
            )
        return _match_transcripts(scenarios, transcripts)


def _transcript_from_entry(entry: Mapping) -> AgentTranscript:
    """Normalise one batch_runner trajectory entry into an AgentTranscript."""
    conversations = entry.get("conversations") or []
    prompt = ""
    response = ""
    tool_calls: list[str] = []

    for message in conversations:
        if not isinstance(message, Mapping):
            continue
        role = message.get("from") or message.get("role") or ""
        value = message.get("value")
        if value is None:
            value = message.get("content")
        if role in {"human", "user"} and not prompt:
            prompt = str(value or "").strip()
        elif role in {"gpt", "assistant"}:
            if isinstance(value, str) and value.strip():
                response = value.strip()
        elif role in {"function_call", "tool_call", "tool_calls"}:
            name = _tool_name_from_call(value)
            if name:
                tool_calls.append(name)

    # tool_stats is the authoritative record of what actually ran.
    for name in (entry.get("tool_stats") or {}):
        if name not in tool_calls:
            tool_calls.append(str(name))

    return AgentTranscript(
        prompt=prompt,
        response=response,
        tool_calls=tuple(tool_calls),
        completed=bool(entry.get("completed", True)),
        raw=dict(entry),
    )


def _tool_name_from_call(value) -> str:
    if isinstance(value, Mapping):
        return str(value.get("name") or "")
    if isinstance(value, str):
        try:
            blob = json.loads(value)
        except json.JSONDecodeError:
            match = re.search(r'"name"\s*:\s*"([^"]+)"', value)
            return match.group(1) if match else ""
        if isinstance(blob, Mapping):
            return str(blob.get("name") or "")
    return ""


def _match_transcripts(
    scenarios: Sequence[BehavioralScenario], transcripts: Sequence[AgentTranscript]
) -> list[AgentTranscript]:
    """Attach scenario ids by prompt text.

    batch_runner keys resume on prompt text rather than index, and it reorders
    across batch files, so prompt text is the only stable join available.
    """
    by_prompt = {s.prompt.strip(): s for s in scenarios}
    out: list[AgentTranscript] = []
    for transcript in transcripts:
        scenario = by_prompt.get(transcript.prompt.strip())
        if scenario is not None:
            transcript.scenario_id = scenario.scenario_id
        out.append(transcript)
    return out


@dataclass
class DirectPromptHarness:
    """Exercise a section through dspy when there is no hermes-agent to run.

    Degraded on purpose, and honest about it. There are no real tools here, so
    the model *reports* the tools it would call. That measures whether the
    guidance reads correctly, which is a weaker claim than measuring what the
    agent did, and every report built from this harness is labelled ``direct``.
    """

    model: Optional[str] = None
    predictor: object = None

    def run(
        self,
        scenarios: Sequence[BehavioralScenario],
        system_prompt: str,
        section_name: str = "SYSTEM_PROMPT",
        run_name: str = "direct",
    ) -> list[AgentTranscript]:
        """Run every scenario against *system_prompt* and collect the transcripts."""
        module = SectionBehaviorModule(
            system_prompt, section_name, predictor=self.predictor
        )
        lm = dspy.LM(self.model, temperature=0.0) if self.model else None

        transcripts: list[AgentTranscript] = []
        for scenario in scenarios:
            if lm is not None:
                with dspy.context(lm=lm):
                    prediction = module(user_message=scenario.prompt)
            else:
                prediction = module(user_message=scenario.prompt)
            transcripts.append(
                AgentTranscript(
                    prompt=scenario.prompt,
                    response=getattr(prediction, "response", "") or "",
                    tool_calls=_split_tool_calls(
                        getattr(prediction, "tool_calls", "") or ""
                    ),
                    scenario_id=scenario.scenario_id,
                    source="direct",
                )
            )
        return transcripts


def select_harness(
    hermes_repo: Optional[Path],
    model: str,
    max_turns: int = 6,
    num_workers: int = 4,
) -> tuple[object, str]:
    """Pick the strongest harness available and say why.

    Returns the harness and a one-line reason, so the CLI can tell the operator
    which claim the numbers support before they read the numbers.
    """
    if hermes_repo is not None:
        batch = BatchRunnerHarness(
            hermes_repo=Path(hermes_repo),
            model=model,
            max_turns=max_turns,
            num_workers=num_workers,
        )
        if batch.available:
            return batch, f"batch_runner at {batch.runner_path}"
    return (
        DirectPromptHarness(model=model),
        "no batch_runner.py found - falling back to the direct dspy harness "
        "(measures how the guidance reads, not what the agent does)",
    )


# ──────────────────────────────────────────────────────────────────────────
# Suite
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class BehavioralSuite:
    """A set of scenarios plus the splits an optimization run needs."""

    scenarios: list[BehavioralScenario] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.scenarios)

    def __iter__(self):
        return iter(self.scenarios)

    @classmethod
    def from_seeds(
        cls,
        sections: Sequence[str] = (),
        categories: Sequence[str] = (),
        per_category: Optional[int] = None,
        include_platform: bool = True,
    ) -> "BehavioralSuite":
        """Build a suite from the seed scenario bank."""
        return cls(
            scenarios=build_scenarios(
                sections=sections,
                categories=categories,
                per_category=per_category,
                include_platform=include_platform,
            )
        )

    def extend(self, scenarios: Iterable[BehavioralScenario]) -> "BehavioralSuite":
        """Add scenarios in place and return self, so calls can be chained."""
        self.scenarios.extend(scenarios)
        return self

    def for_section(self, section: str) -> list[BehavioralScenario]:
        """Every scenario whose section under test is *section*."""
        return [s for s in self.scenarios if s.section_under_test == section]

    def categories(self) -> dict[str, int]:
        """Scenario count per category, name-sorted."""
        counts: dict[str, int] = {}
        for scenario in self.scenarios:
            counts[scenario.category] = counts.get(scenario.category, 0) + 1
        return dict(sorted(counts.items()))

    def split(
        self,
        train_ratio: float = 0.5,
        val_ratio: float = 0.25,
        seed: int = 0,
    ) -> tuple[list[BehavioralScenario], list[BehavioralScenario], list[BehavioralScenario]]:
        """Stratified train/val/holdout split.

        Stratified by category because the categories are not interchangeable:
        a holdout that happened to be all identity scenarios would say nothing
        about whether memory guidance survived.
        """
        rng = random.Random(seed)
        buckets: dict[str, list[BehavioralScenario]] = {}
        for scenario in self.scenarios:
            buckets.setdefault(scenario.category, []).append(scenario)

        train: list[BehavioralScenario] = []
        val: list[BehavioralScenario] = []
        holdout: list[BehavioralScenario] = []

        for category in sorted(buckets):
            items = list(buckets[category])
            rng.shuffle(items)
            n = len(items)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            if n >= 3:
                n_train = max(1, n_train)
                n_val = max(1, n_val)
                if n_train + n_val >= n:
                    n_val = max(1, n - n_train - 1)
            train.extend(items[:n_train])
            val.extend(items[n_train : n_train + n_val])
            holdout.extend(items[n_train + n_val :])

        return train, val, holdout

    def save(self, path: Path) -> Path:
        """Write the suite to *path* as JSONL and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for scenario in self.scenarios:
                handle.write(json.dumps(scenario.to_dict(), ensure_ascii=False) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> "BehavioralSuite":
        """Read a suite back from a JSONL file."""
        scenarios = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                scenarios.append(BehavioralScenario.from_dict(json.loads(line)))
        return cls(scenarios=scenarios)

    def evaluate(
        self,
        system_prompt: str,
        harness,
        judge: Optional[BehavioralJudge] = None,
        section_name: str = "SYSTEM_PROMPT",
        run_name: str = "hase-behavioral",
        scenarios: Optional[Sequence[BehavioralScenario]] = None,
    ) -> BehavioralReport:
        """Run every scenario under *system_prompt* and score the results."""
        targets = list(scenarios if scenarios is not None else self.scenarios)
        if not targets:
            return BehavioralReport(outcomes=[], harness="none")

        if isinstance(harness, BatchRunnerHarness):
            transcripts = harness.run(targets, system_prompt, run_name)
            harness_name = "batch_runner"
        else:
            transcripts = harness.run(
                targets, system_prompt, section_name=section_name, run_name=run_name
            )
            harness_name = "direct"

        by_id = {t.scenario_id: t for t in transcripts if t.scenario_id}
        scorer = judge or BehavioralJudge(use_llm=False)
        report = scorer.score_all(targets, by_id)
        report.harness = harness_name
        return report
