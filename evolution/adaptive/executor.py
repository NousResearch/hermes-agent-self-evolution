"""Execute step of the MAPE-K loop: run one chosen strategy on one task.

One small protocol, two implementations:

* ``Executor`` -- ``execute(task, strategy) -> ExecutionOutcome``.
* ``FakeExecutor`` -- deterministic and table-driven; for tests and offline
  demonstrations. No network, no clock, no randomness.
* ``HermesExecutor`` -- runs real Hermes Agent through its documented Python
  library API (``run_agent.AIAgent.run_conversation``), in a subprocess under
  Hermes' own interpreter. See ``hermes_bridge`` for the exact functions called
  and the hermes-agent files that define them.

Executors only REPORT what happened (output, cost, latency, error, policy
violations). They never score: quality comes from an independent verifier and
reward from ``core.score_outcome``. Budgets and the toolset allowlist are fixed
by the experimenter in this module or by constructor arguments; a ``Strategy``
cannot change them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Protocol, runtime_checkable

from .core import Reward

# Toolsets a strategy may enable, mapped to the tool names each one provides
# (hermes-agent toolsets.py: "todo" -> ["todo_list"], an in-session plan list).
# Everything else is refused: no terminal, code execution, file writes, browser,
# skills/memory edits, delegation, scheduling or network tools.
ALLOWED_TOOLSETS: Mapping[str, tuple[str, ...]] = MappingProxyType({"todo": ("todo_list",)})

# Always passed to Hermes as ``disabled_toolsets``. Hermes subtracts disabled
# toolsets last (model_tools._select_tool_names), so this is defence in depth
# on top of the explicit ``enabled_toolsets`` allowlist.
DENIED_TOOLSETS: tuple[str, ...] = (
    "terminal", "code_execution", "computer_use", "browser", "file", "skills",
    "memory", "delegation", "cronjob", "kanban", "session_search", "connections",
)

_BRIDGE_PATH = Path(__file__).with_name("hermes_bridge.py")


def _finite_nonneg(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


@dataclass(frozen=True)
class Task:
    """A unit of work. ``prompt`` is sent to the agent verbatim."""
    task_id: str
    prompt: str

    def __post_init__(self):
        if not self.task_id or not isinstance(self.task_id, str):
            raise ValueError("task_id must be a nonempty string")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a nonempty string")


@dataclass(frozen=True)
class Strategy:
    """A pre-approved way of running the agent, selected by the policy by ID."""
    strategy_id: str
    system_prompt: str = ""
    toolsets: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.strategy_id or not isinstance(self.strategy_id, str):
            raise ValueError("strategy_id must be a nonempty string")
        if not isinstance(self.system_prompt, str):
            raise ValueError("system_prompt must be a string")
        if not isinstance(self.toolsets, tuple) or len(set(self.toolsets)) != len(self.toolsets):
            raise ValueError("toolsets must be a tuple of unique names")
        refused = [name for name in self.toolsets if name not in ALLOWED_TOOLSETS]
        if refused:
            raise ValueError(f"toolsets not permitted for adaptive strategies: {refused}; "
                             f"allowed: {sorted(ALLOWED_TOOLSETS)}")

    @property
    def allowed_tools(self) -> frozenset[str]:
        return frozenset(t for ts in self.toolsets for t in ALLOWED_TOOLSETS[ts])


@dataclass(frozen=True)
class Budget:
    """Per-execution resource limits, identical for every compared strategy."""
    max_iterations: int = 8            # Hermes tool-calling iterations per turn
    timeout_s: float = 300.0           # Hermes run budget (run_budget_seconds)
    startup_grace_s: float = 120.0     # extra time before the subprocess is killed
    max_tokens: Optional[int] = None   # per-response output token cap
    max_cost_usd: Optional[float] = None  # exceeding it is a violation

    def __post_init__(self):
        if not isinstance(self.max_iterations, int) or isinstance(self.max_iterations, bool) \
                or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive int")
        for name in ("timeout_s", "startup_grace_s"):
            if not _finite_nonneg(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if self.max_tokens is not None and (not isinstance(self.max_tokens, int)
                                            or isinstance(self.max_tokens, bool)
                                            or self.max_tokens < 1):
            raise ValueError("max_tokens must be a positive int or None")
        if self.max_cost_usd is not None and not _finite_nonneg(self.max_cost_usd):
            raise ValueError("max_cost_usd must be a nonnegative finite number or None")


@dataclass(frozen=True)
class ExecutionOutcome:
    """What one execution produced. Not a score."""
    task_id: str
    strategy_id: str
    backend: str                          # "fake" or "hermes"
    output: str
    completed: bool                       # turn finished normally
    latency_s: float                      # agent turn wall time
    cost_usd: Optional[float]             # None = unknown, never silently zero
    cost_status: str                      # Hermes CostStatus, or "fake"
    error: Optional[str] = None
    violation_reasons: tuple[str, ...] = ()
    tool_calls: tuple[str, ...] = ()
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    api_calls: Optional[int] = None
    model: Optional[str] = None
    wall_time_s: Optional[float] = None   # includes interpreter/Hermes start-up
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not _finite_nonneg(self.latency_s):
            raise ValueError("latency_s must be a nonnegative finite number")
        if self.cost_usd is not None and not _finite_nonneg(self.cost_usd):
            raise ValueError("cost_usd must be a nonnegative finite number or None")
        if not isinstance(self.completed, bool):
            raise ValueError("completed must be a bool")
        if not all(isinstance(r, str) and r for r in self.violation_reasons):
            raise ValueError("violation_reasons must be nonempty strings")

    @property
    def violation(self) -> bool:
        return bool(self.violation_reasons)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["details"] = dict(self.details)
        data["violation"] = self.violation
        return data


@runtime_checkable
class Executor(Protocol):
    def execute(self, task: Task, strategy: Strategy) -> ExecutionOutcome: ...


def audit_violations(strategy: Strategy, budget: Budget, *, tool_calls=(),
                     cost_usd: Optional[float] = None,
                     guardrail: Optional[Mapping] = None) -> tuple[str, ...]:
    """Hard policy breaches observed in one execution (order-preserving, unique).

    * a tool call outside the strategy's allowlisted toolsets;
    * a Hermes tool-call guardrail that blocked or halted the turn;
    * known cost above ``budget.max_cost_usd``.
    """
    reasons: list[str] = []
    allowed = strategy.allowed_tools
    for name in tool_calls:
        if name not in allowed:
            reasons.append(f"disallowed_tool:{name}")
    if guardrail and guardrail.get("action") in ("block", "halt"):
        reasons.append(f"guardrail_{guardrail['action']}:{guardrail.get('code', '')}")
    if budget.max_cost_usd is not None and cost_usd is not None and cost_usd > budget.max_cost_usd:
        reasons.append("cost_budget_exceeded")
    return tuple(dict.fromkeys(reasons))


def to_reward(outcome: ExecutionOutcome, *, quality: float,
              reference_cost_usd: float, reference_latency_s: float) -> Reward:
    """Build the ``core.Reward`` input for ``score_outcome``.

    ``quality`` must come from an independent verifier; this function does not
    judge the output. Refuses to score an unknown cost as free.
    """
    if outcome.cost_usd is None:
        raise ValueError(f"cost unknown for {outcome.task_id}/{outcome.strategy_id} "
                         f"(cost_status={outcome.cost_status!r}); cannot form a cost ratio")
    for name, ref in (("reference_cost_usd", reference_cost_usd),
                      ("reference_latency_s", reference_latency_s)):
        if not _finite_nonneg(ref) or ref <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    return Reward(quality=quality,
                  cost_ratio=outcome.cost_usd / reference_cost_usd,
                  time_ratio=outcome.latency_s / reference_latency_s,
                  violation=outcome.violation)


def loop_executor(executor: Executor, catalogue: Mapping[str, Strategy],
                  task_for_context: Callable[[str], Task]) -> Callable[[str, str], ExecutionOutcome]:
    """Adapt an ``Executor`` to ``core.choose_strategy``'s ``executor(context, strategy_id)``."""
    for key, strategy in catalogue.items():
        if key != strategy.strategy_id:
            raise ValueError(f"catalogue key {key!r} != strategy_id {strategy.strategy_id!r}")
    frozen = MappingProxyType(dict(catalogue))

    def run(context: str, strategy_id: str) -> ExecutionOutcome:
        return executor.execute(task_for_context(context), frozen[strategy_id])
    return run


# --------------------------------------------------------------------------- fake

@dataclass(frozen=True)
class FakeResponse:
    """Scripted result for one (task_id, strategy_id) pair."""
    output: str = ""
    cost_usd: Optional[float] = 0.0
    latency_s: float = 0.0
    completed: bool = True
    error: Optional[str] = None
    tool_calls: tuple[str, ...] = ()
    guardrail: Optional[Mapping] = None


class FakeExecutor:
    """Deterministic executor for tests: looks responses up, never calls Hermes.

    Applies the same ``audit_violations`` policy as ``HermesExecutor``.
    """
    backend = "fake"

    def __init__(self, responses: Mapping[tuple[str, str], FakeResponse], *,
                 default: Optional[FakeResponse] = None, budget: Budget = Budget()):
        self.responses = dict(responses)
        self.default = default
        self.budget = budget
        self.calls: list[tuple[str, str]] = []

    def execute(self, task: Task, strategy: Strategy) -> ExecutionOutcome:
        self.calls.append((task.task_id, strategy.strategy_id))
        response = self.responses.get((task.task_id, strategy.strategy_id), self.default)
        if response is None:
            raise KeyError(f"no scripted response for {(task.task_id, strategy.strategy_id)}")
        return ExecutionOutcome(
            task_id=task.task_id, strategy_id=strategy.strategy_id, backend=self.backend,
            output=response.output, completed=response.completed,
            latency_s=response.latency_s, cost_usd=response.cost_usd, cost_status="fake",
            error=response.error, tool_calls=tuple(response.tool_calls),
            violation_reasons=audit_violations(strategy, self.budget,
                                               tool_calls=response.tool_calls,
                                               cost_usd=response.cost_usd,
                                               guardrail=response.guardrail),
        )


# ------------------------------------------------------------------------- hermes

class LiveExecutionDisabled(RuntimeError):
    """Raised when HermesExecutor is used without explicit opt-in."""


Runner = Callable[..., subprocess.CompletedProcess]


class HermesExecutor:
    """Run Hermes Agent for real via ``run_agent.AIAgent.run_conversation``.

    Off by default: a real run calls the configured LLM provider and spends
    money, so ``allow_live=True`` is required. The agent runs with no tools
    unless the strategy enables allowlisted ones, without memory, context
    files, trajectories or background review (see ``hermes_bridge``).

    ``python`` must be an interpreter that can import hermes-agent (its own
    environment); default ``$HERMES_PYTHON``, else the current interpreter.
    ``runner`` replaces ``subprocess.run`` in tests.

    ``model`` is required for a live run. Hermes' Python-library guide says an
    empty model resolves from the user's config, but the first live smoke run
    (2026-11-22) showed it does not: ``AIAgent(model="")`` reached the provider
    with no model and failed with ``model is required``. Pass ``model`` (and
    usually ``provider``) explicitly.
    """
    backend = "hermes"

    def __init__(self, *, hermes_repo: Optional[str | Path] = None,
                 python: Optional[str] = None, model: Optional[str] = None,
                 provider: Optional[str] = None, budget: Budget = Budget(),
                 allow_live: bool = False, runner: Optional[Runner] = None):
        self.hermes_repo = hermes_repo
        self.python = python or os.environ.get("HERMES_PYTHON") or sys.executable
        self.model = model
        self.provider = provider
        self.budget = budget
        self.allow_live = allow_live
        self._runner = runner or subprocess.run

    def _repo(self) -> Path:
        from evolution.core.config import resolve_hermes_agent_path
        repo = resolve_hermes_agent_path(str(self.hermes_repo) if self.hermes_repo else None)
        if not (repo / "run_agent.py").is_file():
            raise FileNotFoundError(f"{repo} has no run_agent.py; not a hermes-agent checkout")
        return repo.resolve()

    def build_request(self, task: Task, strategy: Strategy, repo: Path) -> dict:
        return {
            "hermes_repo": str(repo),
            "task_id": task.task_id,
            "prompt": task.prompt,
            "system_prompt": strategy.system_prompt,
            "enabled_toolsets": list(strategy.toolsets),
            "disabled_toolsets": [t for t in DENIED_TOOLSETS if t not in strategy.toolsets],
            "max_iterations": self.budget.max_iterations,
            "run_budget_seconds": self.budget.timeout_s,
            "max_tokens": self.budget.max_tokens,
            "model": self.model,
            "provider": self.provider,
        }

    def execute(self, task: Task, strategy: Strategy) -> ExecutionOutcome:
        if not self.allow_live:
            raise LiveExecutionDisabled(
                "HermesExecutor calls a real LLM provider; construct it with allow_live=True")
        if not self.model:
            raise ValueError("HermesExecutor needs an explicit model for a live run; "
                             "AIAgent(model='') does not resolve the configured model")
        repo = self._repo()
        with tempfile.TemporaryDirectory(prefix="kafo-hermes-") as tmp:
            request_path = os.path.join(tmp, "request.json")
            result_path = os.path.join(tmp, "result.json")
            with open(request_path, "w", encoding="utf-8") as handle:
                json.dump(self.build_request(task, strategy, repo), handle)
            argv = [self.python, "-c", _BRIDGE_PATH.read_text(encoding="utf-8"),
                    request_path, result_path]
            start = time.perf_counter()
            try:
                proc = self._runner(argv, cwd=str(repo), stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, check=False,
                                    timeout=self.budget.timeout_s + self.budget.startup_grace_s)
            except subprocess.TimeoutExpired:
                return self._failure(task, strategy, time.perf_counter() - start,
                                     f"timeout: killed after {self.budget.timeout_s + self.budget.startup_grace_s:g}s")
            wall = time.perf_counter() - start
            try:
                with open(result_path, encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, ValueError):
                stderr = (getattr(proc, "stderr", "") or "")[-1000:]
                return self._failure(task, strategy, wall,
                                     f"bridge_failed: exit {proc.returncode}: {stderr}".strip())
        if not isinstance(payload, dict) or not payload.get("ok"):
            error = payload.get("error") if isinstance(payload, dict) else None
            return self._failure(task, strategy, wall, f"hermes_error: {error or 'unknown'}")
        return self._outcome(task, strategy, payload, wall)

    def _failure(self, task, strategy, wall, error) -> ExecutionOutcome:
        # Spend before the failure is not observable here, so cost is unknown.
        return ExecutionOutcome(task_id=task.task_id, strategy_id=strategy.strategy_id,
                                backend=self.backend, output="", completed=False,
                                latency_s=wall, cost_usd=None, cost_status="unknown",
                                error=error, wall_time_s=wall, model=self.model)

    def _outcome(self, task, strategy, payload: dict, wall: float) -> ExecutionOutcome:
        completed = (payload.get("completed") is True and not payload.get("failed")
                     and not payload.get("partial") and not payload.get("interrupted"))
        cost_status = payload.get("cost_status") or "unknown"
        cost = payload.get("estimated_cost_usd")
        cost_usd = None if cost_status == "unknown" or not _finite_nonneg(cost) else float(cost)
        error = payload.get("error")
        if not completed and not error:
            error = payload.get("turn_exit_reason") or "incomplete"
        tool_calls = tuple(payload.get("tool_calls") or ())
        latency = payload.get("latency_s")
        details = {k: payload[k] for k in ("turn_exit_reason", "cost_source", "provider",
                                           "total_tokens", "cache_read_tokens",
                                           "cache_write_tokens", "failure_reason", "guardrail",
                                           "failed", "partial", "interrupted")
                   if payload.get(k) is not None}
        return ExecutionOutcome(
            task_id=task.task_id, strategy_id=strategy.strategy_id, backend=self.backend,
            output=payload.get("final_response") or "", completed=completed,
            latency_s=float(latency) if _finite_nonneg(latency) else wall,
            cost_usd=cost_usd, cost_status=cost_status, error=error,
            violation_reasons=audit_violations(strategy, self.budget, tool_calls=tool_calls,
                                               cost_usd=cost_usd,
                                               guardrail=payload.get("guardrail")),
            tool_calls=tool_calls, input_tokens=payload.get("input_tokens"),
            output_tokens=payload.get("output_tokens"), api_calls=payload.get("api_calls"),
            model=payload.get("model") or self.model, wall_time_s=wall, details=details,
        )
