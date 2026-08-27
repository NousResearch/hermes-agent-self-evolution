"""Evaluate skills by running the real Hermes agent.

The original fitness function was a single ``dspy.ChainOfThought`` call with
the skill text as instructions. No tools, no files, no multi-turn. It measured
how a bare completion responds to a synthetic prompt — which is only loosely
related to how the Hermes agent behaves in production with that skill loaded.

Hermes already solved this for itself. ``hermes-agent/evals/readtool/`` drives
the real ``AIAgent`` with real toolsets against deterministic fixtures, grades
answers against planted ground truth, and reports accuracy alongside API
turns, tool calls, tokens and wall time. This module follows that pattern for
skills.

The agent import is lazy and optional. Without a Hermes checkout on the path,
:class:`HermesAgentBackend` reports unavailable and callers fall back to the
completion-level evaluator — so this never becomes a hard dependency for
someone just optimizing prompt text.
"""

from __future__ import annotations

import inspect
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

# Toolsets a skill evaluation gets by default. Matches what readtool uses for
# its full-toolset arm.
DEFAULT_TOOLSETS = ("file", "terminal", "search")

# Hard ceiling on a single task so one hung run cannot stall a whole sweep.
DEFAULT_TASK_TIMEOUT_S = 300.0


# ── grading ──────────────────────────────────────────────────────────────


class Grader(Protocol):
    """Scores an agent's final answer against planted ground truth."""

    def grade(self, output: str) -> float: ...

    def describe(self) -> str: ...


@dataclass
class SubstringGrader:
    """Pass when the expected strings appear in the answer.

    ``mode="all"`` scores the fraction present, so a partially-correct answer
    is distinguishable from a wholly wrong one — GEPA needs that gradient.
    """

    needles: Sequence[str]
    mode: str = "all"
    case_sensitive: bool = False

    def grade(self, output: str) -> float:
        if not self.needles:
            return 0.0
        haystack = output if self.case_sensitive else output.lower()
        hits = 0
        for needle in self.needles:
            probe = needle if self.case_sensitive else needle.lower()
            if probe in haystack:
                hits += 1
        if self.mode == "any":
            return 1.0 if hits else 0.0
        return hits / len(self.needles)

    def describe(self) -> str:
        return f"substring[{self.mode}]: {', '.join(self.needles[:4])}"


@dataclass
class RegexGrader:
    """Pass when the answer matches the expected patterns."""

    patterns: Sequence[str]
    mode: str = "all"
    flags: int = re.IGNORECASE

    def grade(self, output: str) -> float:
        if not self.patterns:
            return 0.0
        hits = sum(1 for p in self.patterns if re.search(p, output, self.flags))
        if self.mode == "any":
            return 1.0 if hits else 0.0
        return hits / len(self.patterns)

    def describe(self) -> str:
        return f"regex[{self.mode}]: {', '.join(self.patterns[:3])}"


@dataclass
class CallableGrader:
    """Escape hatch for graders that need real logic."""

    fn: Callable[[str], float]
    label: str = "callable"

    def grade(self, output: str) -> float:
        try:
            return max(0.0, min(1.0, float(self.fn(output))))
        except Exception:
            return 0.0

    def describe(self) -> str:
        return self.label


# ── tasks and results ────────────────────────────────────────────────────


@dataclass
class AgentTask:
    """One graded task, optionally with a deterministic workspace."""

    id: str
    prompt: str
    grader: Grader
    toolsets: tuple[str, ...] = DEFAULT_TOOLSETS
    build_workspace: Optional[Callable[[Path], None]] = None
    timeout_s: float = DEFAULT_TASK_TIMEOUT_S
    category: str = "general"


@dataclass
class TurnLog:
    """What a backend reports back about one agent run."""

    output: str = ""
    api_turns: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    tools_used: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class TaskResult:
    """A graded run: accuracy plus the efficiency metrics readtool tracks."""

    task_id: str
    score: float
    api_turns: int
    tool_calls: int
    total_tokens: int
    wall_s: float
    output: str = ""
    error: str = ""
    category: str = "general"

    @property
    def ok(self) -> bool:
        return not self.error

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "score": round(self.score, 4),
            "api_turns": self.api_turns,
            "tool_calls": self.tool_calls,
            "total_tokens": self.total_tokens,
            "wall_s": round(self.wall_s, 2),
            "error": self.error,
        }


@dataclass
class EvalRun:
    """All task results for one arm of an A/B."""

    label: str
    results: list[TaskResult] = field(default_factory=list)

    def _mean(self, attr: str) -> float:
        usable = [r for r in self.results if r.ok]
        if not usable:
            return 0.0
        return sum(getattr(r, attr) for r in usable) / len(usable)

    @property
    def accuracy(self) -> float:
        return self._mean("score")

    @property
    def avg_tokens(self) -> float:
        return self._mean("total_tokens")

    @property
    def avg_tool_calls(self) -> float:
        return self._mean("tool_calls")

    @property
    def avg_api_turns(self) -> float:
        return self._mean("api_turns")

    @property
    def avg_wall_s(self) -> float:
        return self._mean("wall_s")

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if not r.ok)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "tasks": len(self.results),
            "errors": self.errors,
            "accuracy": round(self.accuracy, 4),
            "avg_tokens": round(self.avg_tokens, 1),
            "avg_tool_calls": round(self.avg_tool_calls, 2),
            "avg_api_turns": round(self.avg_api_turns, 2),
            "avg_wall_s": round(self.avg_wall_s, 2),
            "results": [r.as_dict() for r in self.results],
        }


# ── backends ─────────────────────────────────────────────────────────────


class AgentBackend(Protocol):
    """Something that can run a prompt and report what happened."""

    def available(self) -> tuple[bool, str]: ...

    def run(
        self,
        prompt: str,
        system: str,
        toolsets: Sequence[str],
        workdir: Optional[Path],
        timeout_s: float,
    ) -> TurnLog: ...


class HermesAgentBackend:
    """Drives the real ``AIAgent`` from a hermes-agent checkout.

    The constructor signature of ``AIAgent`` moves between Hermes versions, so
    arguments are filtered against the live signature rather than assumed.
    Passing an unexpected keyword would take down every task in the sweep.
    """

    def __init__(
        self,
        hermes_repo: Path,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self.hermes_repo = Path(hermes_repo)
        self.model = model
        self.provider = provider
        self._agent_cls: Any = None
        self._load_error: str = ""

    # -- import plumbing --

    def _load_agent_cls(self) -> Any:
        if self._agent_cls is not None or self._load_error:
            return self._agent_cls

        repo = str(self.hermes_repo)
        if not self.hermes_repo.is_dir():
            self._load_error = f"hermes repo not found: {repo}"
            return None

        added = False
        if repo not in sys.path:
            sys.path.insert(0, repo)
            added = True
        try:
            from run_agent import AIAgent  # type: ignore  # noqa: PLC0415

            self._agent_cls = AIAgent
        except Exception as exc:  # noqa: BLE001 — any import failure is "unavailable"
            self._load_error = f"cannot import AIAgent from {repo}: {exc}"
            if added:
                try:
                    sys.path.remove(repo)
                except ValueError:
                    pass
        return self._agent_cls

    def available(self) -> tuple[bool, str]:
        cls = self._load_agent_cls()
        if cls is None:
            return False, self._load_error or "AIAgent unavailable"
        return True, f"AIAgent from {self.hermes_repo}"

    # -- running --

    def run(
        self,
        prompt: str,
        system: str,
        toolsets: Sequence[str],
        workdir: Optional[Path],
        timeout_s: float,
    ) -> TurnLog:
        cls = self._load_agent_cls()
        if cls is None:
            return TurnLog(error=self._load_error or "AIAgent unavailable")

        kwargs = self._filter_kwargs(
            cls,
            {
                "enabled_toolsets": list(toolsets),
                "model": self.model,
                "provider": self.provider,
                "system_prompt": system or None,
                "working_directory": str(workdir) if workdir else None,
                "cwd": str(workdir) if workdir else None,
                "headless": True,
                "interactive": False,
            },
        )

        prev_cwd = Path.cwd()
        try:
            if workdir:
                os.chdir(workdir)
            agent = cls(**kwargs)
            raw = self._invoke(agent, prompt, timeout_s)
            return self._to_turn_log(agent, raw)
        except Exception as exc:  # noqa: BLE001 — a task failure must not end the sweep
            return TurnLog(error=f"{type(exc).__name__}: {exc}")
        finally:
            try:
                os.chdir(prev_cwd)
            except OSError:
                pass

    @staticmethod
    def _filter_kwargs(cls: Any, desired: dict) -> dict:
        """Keep only the kwargs this AIAgent version actually accepts."""
        try:
            params = inspect.signature(cls.__init__).parameters
        except (TypeError, ValueError):
            return {}
        accepts_any = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        out = {}
        for key, value in desired.items():
            if value is None:
                continue
            if accepts_any or key in params:
                out[key] = value
        return out

    @staticmethod
    def _invoke(agent: Any, prompt: str, timeout_s: float) -> Any:
        """Call whichever run method this Hermes version exposes."""
        for name in ("run", "chat", "send", "run_turn", "process_message"):
            fn = getattr(agent, name, None)
            if callable(fn):
                try:
                    sig = inspect.signature(fn)
                    if "timeout" in sig.parameters:
                        return fn(prompt, timeout=timeout_s)
                except (TypeError, ValueError):
                    pass
                return fn(prompt)
        raise AttributeError("AIAgent exposes no recognized run method")

    @staticmethod
    def _to_turn_log(agent: Any, raw: Any) -> TurnLog:
        messages = (
            getattr(agent, "messages", None)
            or getattr(agent, "conversation", None)
            or []
        )
        log = count_metrics(messages)
        log.output = _stringify_output(raw, messages)
        return log


class ScriptedBackend:
    """Deterministic backend for tests and dry runs.

    Takes a callable that maps a prompt to a response, so the whole evaluation
    path — grading, metrics, A/B comparison — is exercisable without a Hermes
    checkout or a single API call.
    """

    def __init__(
        self,
        responder: Callable[[str, str], str],
        tool_calls: int = 0,
        api_turns: int = 1,
        tokens: int = 0,
    ):
        self.responder = responder
        self.tool_calls = tool_calls
        self.api_turns = api_turns
        self.tokens = tokens

    def available(self) -> tuple[bool, str]:
        return True, "scripted backend"

    def run(
        self,
        prompt: str,
        system: str,
        toolsets: Sequence[str],
        workdir: Optional[Path],
        timeout_s: float,
    ) -> TurnLog:
        try:
            output = self.responder(prompt, system)
        except Exception as exc:  # noqa: BLE001
            return TurnLog(error=f"{type(exc).__name__}: {exc}")
        return TurnLog(
            output=output,
            api_turns=self.api_turns,
            tool_calls=self.tool_calls,
            total_tokens=self.tokens or len(output) // 4,
        )


def count_metrics(messages: Sequence[Any]) -> TurnLog:
    """Derive efficiency metrics from a conversation transcript.

    Messages may be dicts or objects depending on the Hermes version, so both
    shapes are read through the same accessor.
    """
    log = TurnLog()
    for msg in messages or []:
        role = _get(msg, "role", "")
        if role == "assistant":
            log.api_turns += 1
        tool_calls = _get(msg, "tool_calls", None)
        if tool_calls:
            try:
                log.tool_calls += len(tool_calls)
            except TypeError:
                log.tool_calls += 1
        name = _get(msg, "tool_name", "")
        if name:
            log.tools_used.append(name)
        tokens = _get(msg, "token_count", 0)
        if isinstance(tokens, (int, float)):
            log.total_tokens += int(tokens)
    return log


def _get(obj: Any, key: str, default: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _stringify_output(raw: Any, messages: Sequence[Any]) -> str:
    """Best-effort final answer, whatever shape the agent returned."""
    if isinstance(raw, str) and raw.strip():
        return raw
    if isinstance(raw, dict):
        for key in ("output", "content", "text", "response", "message"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value
    for msg in reversed(list(messages or [])):
        if _get(msg, "role", "") == "assistant":
            content = _get(msg, "content", "")
            if isinstance(content, str) and content.strip():
                return content
    return "" if raw is None else str(raw)


# ── the evaluator ────────────────────────────────────────────────────────


class AgentEvaluator:
    """Runs a task suite through a backend with a given skill loaded."""

    def __init__(
        self,
        backend: AgentBackend,
        reps: int = 1,
        keep_workspaces: bool = False,
    ):
        self.backend = backend
        self.reps = max(1, reps)
        self.keep_workspaces = keep_workspaces

    def evaluate(
        self,
        skill_text: str,
        tasks: Sequence[AgentTask],
        label: str = "arm",
    ) -> EvalRun:
        """Run every task ``reps`` times and return the aggregated arm.

        Repetitions matter: agent runs are stochastic, and a single sample per
        task cannot distinguish a real improvement from run-to-run noise. The
        readtool eval treats multi-rep as the baseline standard and so does
        this.
        """
        run = EvalRun(label=label)
        for task in tasks:
            for rep in range(self.reps):
                run.results.append(self._run_once(skill_text, task, rep))
        return run

    def _run_once(self, skill_text: str, task: AgentTask, rep: int) -> TaskResult:
        workdir: Optional[Path] = None
        tmp: Optional[str] = None

        if task.build_workspace is not None:
            tmp = tempfile.mkdtemp(prefix=f"evo-{task.id}-")
            workdir = Path(tmp)
            try:
                task.build_workspace(workdir)
            except Exception as exc:  # noqa: BLE001
                _cleanup(tmp, self.keep_workspaces)
                return TaskResult(
                    task_id=task.id,
                    score=0.0,
                    api_turns=0,
                    tool_calls=0,
                    total_tokens=0,
                    wall_s=0.0,
                    error=f"workspace build failed: {exc}",
                    category=task.category,
                )

        started = time.time()
        try:
            log = self.backend.run(
                prompt=task.prompt,
                system=skill_text,
                toolsets=task.toolsets,
                workdir=workdir,
                timeout_s=task.timeout_s,
            )
        finally:
            _cleanup(tmp, self.keep_workspaces)

        wall = time.time() - started
        score = 0.0 if log.error else task.grader.grade(log.output)

        return TaskResult(
            task_id=f"{task.id}#{rep}" if self.reps > 1 else task.id,
            score=score,
            api_turns=log.api_turns,
            tool_calls=log.tool_calls,
            total_tokens=log.total_tokens,
            wall_s=wall,
            output=log.output[:4000],
            error=log.error,
            category=task.category,
        )


def _cleanup(tmp: Optional[str], keep: bool) -> None:
    if tmp and not keep:
        shutil.rmtree(tmp, ignore_errors=True)


def tasks_from_examples(examples: Sequence[Any]) -> list[AgentTask]:
    """Adapt eval-dataset examples into graded agent tasks.

    ``expected_behavior`` is a rubric in prose, not an exact answer, so it is
    graded on the content words it contains rather than as a literal match.
    Weak by construction — a purpose-built fixture with planted ground truth
    is always better — but it lets an existing dataset drive real agent runs
    instead of nothing at all.
    """
    tasks: list[AgentTask] = []
    for i, ex in enumerate(examples):
        task_input = getattr(ex, "task_input", "") or ""
        expected = getattr(ex, "expected_behavior", "") or ""
        if not task_input:
            continue
        needles = _content_words(expected)
        tasks.append(
            AgentTask(
                id=f"ex{i:03d}",
                prompt=task_input,
                grader=SubstringGrader(needles=needles, mode="all"),
                category=getattr(ex, "category", "general") or "general",
            )
        )
    return tasks


_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]{3,}")
_RUBRIC_STOPWORDS = frozenset(
    ["should", "must", "will", "shall", "would", "could", "include", "includes", "including", "provide", "provides", "response", "answer", "output", "result", "user", "agent", "skill", "that", "this", "with", "from", "when", "what", "which", "have", "been", "also", "make", "sure", "clear", "concise", "correct", "appropriate", "relevant", "specific", "general", "good", "well"]
)


def _content_words(rubric: str, limit: int = 8) -> list[str]:
    seen: list[str] = []
    for match in _WORD.finditer(rubric.lower()):
        word = match.group(0)
        if word in _RUBRIC_STOPWORDS or word in seen:
            continue
        seen.append(word)
        if len(seen) >= limit:
            break
    return seen
