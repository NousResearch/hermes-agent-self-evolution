"""Structural guardrails for evolved hermes-agent code.

PLAN.md calls code evolution the highest-risk tier and answers it with the
strictest guardrail set of any phase. This module is that set, implemented as
AST comparisons between the source before a mutation and the source after it:

    signatures        function names, parameters and defaults are frozen
    registry calls    registry.register(...) is frozen, byte for byte
    error handling    try / except / raise / finally coverage may not shrink
    safety checks     asserts, validation guards and guard returns may not vanish

Every check returns a :class:`SafetyResult` that names what moved rather than
just failing, because a reviewer reading the run log needs to know whether the
candidate renamed a helper or quietly deleted an except block.

Two notes about how these map onto the real hermes-agent, which differs from
what PLAN.md assumes:

* ``tools/file_tools.py`` contains no ``assert`` statements and no ``raise``
  statements at all. Its safety story is guard helpers (``_is_blocked_device``,
  ``_check_sensitive_path``) and early returns of ``tool_error(...)``. Counting
  asserts alone would score that file as having zero safety checks and let a
  candidate strip every guard it has, so guard calls and guard returns are
  counted too.
* Adding a function is allowed; removing or re-signing one is not. A real fix
  often needs a new private helper, and adding one breaks no caller.

Pure source analysis. No LLM, no network, no subprocess.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

__all__ = [
    "SafetyResult",
    "SafetyReport",
    "FunctionSignature",
    "ErrorHandlingCensus",
    "GuardCensus",
    "QualitySignals",
    "check_parses",
    "check_signatures_unchanged",
    "check_registry_calls_unchanged",
    "check_error_handling_not_reduced",
    "check_safety_checks_not_removed",
    "run_safety_checks",
    "quality_signals",
    "collect_signatures",
    "collect_registry_calls",
    "census_error_handling",
    "census_guards",
    "GUARD_CALL_PATTERN",
    "ERROR_RETURN_PATTERN",
    "REGISTRY_CALL_PATTERN",
]


# A guard is a call whose name says it is checking something. The real tool
# modules name theirs _check_sensitive_path, _is_blocked_device,
# _is_expected_write_exception, normalize_read_pagination, so the pattern has
# to cover both the "check/validate" family and the "is_blocked/is_safe" one.
GUARD_CALL_PATTERN = re.compile(
    r"^_?("
    r"check|validate|verify|ensure|assert|guard|require|sanitize|sanitise|"
    r"normalize|normalise|is_blocked|is_safe|is_valid|is_allowed|is_sensitive|"
    r"is_expected|is_forbidden|is_denied|has_permission"
    r")",
    re.IGNORECASE,
)

# An early return that reports a refusal rather than a result.
ERROR_RETURN_PATTERN = re.compile(r"(^|_)(error|fail|deny|reject|refuse)", re.IGNORECASE)

# Tool registration. Frozen because tool discovery reads it at import time.
REGISTRY_CALL_PATTERN = re.compile(r"^register(_|$)")


# ──────────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class SafetyResult:
    """Outcome of one guardrail check."""

    name: str
    passed: bool
    message: str
    violations: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise one safety check and its violations."""
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "violations": list(self.violations),
            "info": list(self.info),
        }


@dataclass
class SafetyReport:
    """Every guardrail result for one candidate."""

    results: list[SafetyResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when every check passed."""
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[SafetyResult]:
        """The checks that failed."""
        return [r for r in self.results if not r.passed]

    @property
    def violations(self) -> list[str]:
        """Every violation across the failing checks, message as fallback."""
        out: list[str] = []
        for result in self.failures:
            out.extend(result.violations or [result.message])
        return out

    def summary(self) -> str:
        """One line per check, marked pass or fail."""
        return "\n".join(
            f"{'✓' if r.passed else '✗'} {r.name}: {r.message}" for r in self.results
        )

    def first_failure(self) -> Optional[SafetyResult]:
        """The first failing check, or None when everything passed."""
        failures = self.failures
        return failures[0] if failures else None

    def to_dict(self) -> dict:
        """Serialise the whole safety report."""
        return {
            "passed": self.passed,
            "results": [r.to_dict() for r in self.results],
        }


# ──────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────────────────────────────────


def _parse(source: str) -> ast.Module:
    return ast.parse(source)


def _unparse(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    return " ".join(ast.unparse(node).split())


def _callee_name(node: ast.expr) -> Optional[str]:
    """Terminal name of a call target: ``a.b.register`` -> ``register``."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return _callee_name(node.func)
    return None


def _callee_path(node: ast.expr) -> str:
    """Dotted spelling of a call target, best effort."""
    if isinstance(node, ast.Attribute):
        return f"{_callee_path(node.value)}.{node.attr}".lstrip(".")
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _walk_functions(
    tree: ast.Module,
) -> Iterable[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Yield ``(qualname, node)`` for every function, nested ones included."""

    def visit(node: ast.AST, prefix: str):
        """Walk nested definitions, yielding each qualified name with its node.

        Functions inside functions are followed, so a mutation hidden in a closure
        is still compared.
        """
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = f"{prefix}{child.name}"
                yield qualname, child
                yield from visit(child, f"{qualname}.")
            elif isinstance(child, ast.ClassDef):
                yield from visit(child, f"{prefix}{child.name}.")
            else:
                yield from visit(child, prefix)

    yield from visit(tree, "")


# ──────────────────────────────────────────────────────────────────────────
# Signatures
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FunctionSignature:
    """The part of a function's shape that callers depend on.

    Parameter *annotations* are deliberately excluded: tightening a type hint
    changes no call site. Names, order, defaults, star-args and async-ness are
    all included, because changing any of them breaks somebody.
    """

    qualname: str
    is_async: bool
    posonly: tuple[str, ...]
    args: tuple[str, ...]
    defaults: tuple[str, ...]
    vararg: Optional[str]
    kwonly: tuple[str, ...]
    kwonly_defaults: tuple[Optional[str], ...]
    kwarg: Optional[str]

    def render(self) -> str:
        """Render the signature back to source text, defaults included."""
        bits: list[str] = []
        positional = list(self.posonly) + list(self.args)
        pad = len(positional) - len(self.defaults)
        for i, name in enumerate(positional):
            if i >= pad:
                bits.append(f"{name}={self.defaults[i - pad]}")
            else:
                bits.append(name)
            if self.posonly and i == len(self.posonly) - 1:
                bits.append("/")
        if self.vararg:
            bits.append(f"*{self.vararg}")
        elif self.kwonly:
            bits.append("*")
        for name, default in zip(self.kwonly, self.kwonly_defaults):
            bits.append(f"{name}={default}" if default is not None else name)
        if self.kwarg:
            bits.append(f"**{self.kwarg}")
        prefix = "async def " if self.is_async else "def "
        return f"{prefix}{self.qualname}({', '.join(bits)})"


def _signature_of(
    qualname: str, node: ast.FunctionDef | ast.AsyncFunctionDef
) -> FunctionSignature:
    a = node.args
    return FunctionSignature(
        qualname=qualname,
        is_async=isinstance(node, ast.AsyncFunctionDef),
        posonly=tuple(arg.arg for arg in a.posonlyargs),
        args=tuple(arg.arg for arg in a.args),
        defaults=tuple(_unparse(d) or "" for d in a.defaults),
        vararg=a.vararg.arg if a.vararg else None,
        kwonly=tuple(arg.arg for arg in a.kwonlyargs),
        kwonly_defaults=tuple(_unparse(d) for d in a.kw_defaults),
        kwarg=a.kwarg.arg if a.kwarg else None,
    )


def collect_signatures(source: str) -> dict[str, FunctionSignature]:
    """Map every function's qualified name to its signature."""
    tree = _parse(source)
    return {
        qualname: _signature_of(qualname, node)
        for qualname, node in _walk_functions(tree)
    }


def check_signatures_unchanged(before: str, after: str) -> SafetyResult:
    """Refuse removed or re-signed functions. Added functions are fine."""
    old = collect_signatures(before)
    new = collect_signatures(after)

    violations: list[str] = []
    info: list[str] = []

    for name in sorted(set(old) - set(new)):
        violations.append(f"removed function: {old[name].render()}")
    for name in sorted(set(new) - set(old)):
        info.append(f"added function: {new[name].render()}")
    for name in sorted(set(old) & set(new)):
        if old[name] != new[name]:
            violations.append(
                f"signature changed: {old[name].render()} -> {new[name].render()}"
            )

    if violations:
        return SafetyResult(
            name="signatures_frozen",
            passed=False,
            message=f"{len(violations)} signature change(s) would break callers",
            violations=violations,
            info=info,
        )
    return SafetyResult(
        name="signatures_frozen",
        passed=True,
        message=f"{len(new)} function signature(s) intact"
        + (f", {len(info)} added" if info else ""),
        info=info,
    )


# ──────────────────────────────────────────────────────────────────────────
# Registry calls
# ──────────────────────────────────────────────────────────────────────────


def collect_registry_calls(source: str) -> list[str]:
    """Normalized source of every ``register*(...)`` call in the module.

    hermes-agent registers tools with a single module-level statement per tool
    (``registry.register(name=..., toolset=..., schema=..., handler=..., ...)``).
    Comparing the normalized call text catches a changed toolset, a swapped
    handler, a dropped ``check_fn`` and an added ``override=True`` alike.
    """
    tree = _parse(source)
    calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _callee_name(node.func)
        if name and REGISTRY_CALL_PATTERN.match(name):
            calls.append(_unparse(node) or "")
    return calls


def check_registry_calls_unchanged(before: str, after: str) -> SafetyResult:
    """Refuse any change to tool registration - it drives tool discovery."""
    old = Counter(collect_registry_calls(before))
    new = Counter(collect_registry_calls(after))

    removed = old - new
    added = new - old

    violations: list[str] = []
    for call, count in sorted(removed.items()):
        violations.extend([f"removed registration: {call}"] * count)
    for call, count in sorted(added.items()):
        violations.extend([f"added registration: {call}"] * count)

    if violations:
        return SafetyResult(
            name="registry_frozen",
            passed=False,
            message=f"{len(violations)} change(s) to registry registration",
            violations=violations,
        )
    return SafetyResult(
        name="registry_frozen",
        passed=True,
        message=f"{sum(old.values())} registration call(s) unchanged",
    )


# ──────────────────────────────────────────────────────────────────────────
# Error handling
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ErrorHandlingCensus:
    """How much error handling a module contains."""

    try_blocks: int
    handlers: int
    bare_handlers: int
    silent_handlers: int
    raises: int
    finallys: int
    per_function: dict[str, int]

    def totals(self) -> dict[str, int]:
        """The raw counts behind the error-handling comparison."""
        return {
            "try_blocks": self.try_blocks,
            "handlers": self.handlers,
            "raises": self.raises,
            "finallys": self.finallys,
        }


def _handler_is_silent(handler: ast.ExceptHandler) -> bool:
    """True when a handler swallows the exception without doing anything."""
    body = [
        stmt
        for stmt in handler.body
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
    ]
    if not body:
        return True
    return len(body) == 1 and isinstance(body[0], ast.Pass)


def census_error_handling(source: str) -> ErrorHandlingCensus:
    """Count error-handling constructs across a module, and per function."""
    tree = _parse(source)

    try_blocks = handlers = bare = silent = raises = finallys = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            try_blocks += 1
            handlers += len(node.handlers)
            if node.finalbody:
                finallys += 1
            for handler in node.handlers:
                if handler.type is None:
                    bare += 1
                if _handler_is_silent(handler):
                    silent += 1
        elif isinstance(node, ast.Raise):
            raises += 1

    per_function: dict[str, int] = {}
    for qualname, fn in _walk_functions(tree):
        count = 0
        for node in ast.walk(fn):
            if isinstance(node, ast.Try):
                count += 1 + len(node.handlers)
            elif isinstance(node, ast.Raise):
                count += 1
        per_function[qualname] = count

    return ErrorHandlingCensus(
        try_blocks=try_blocks,
        handlers=handlers,
        bare_handlers=bare,
        silent_handlers=silent,
        raises=raises,
        finallys=finallys,
        per_function=per_function,
    )


def check_error_handling_not_reduced(before: str, after: str) -> SafetyResult:
    """Refuse a net decrease in error handling, module-wide or per function.

    Per-function matters as much as the total: a candidate that strips the
    try/except out of ``read_file_tool`` and adds one to a helper leaves the
    totals flat while making the tool crash on a bad path.
    """
    old = census_error_handling(before)
    new = census_error_handling(after)

    violations: list[str] = []
    info: list[str] = []

    for label, old_count in old.totals().items():
        new_count = new.totals()[label]
        if new_count < old_count:
            violations.append(f"{label} dropped {old_count} -> {new_count}")
        elif new_count > old_count:
            info.append(f"{label} grew {old_count} -> {new_count}")

    for qualname, old_count in sorted(old.per_function.items()):
        if qualname not in new.per_function:
            continue  # a removed function is the signature check's business
        new_count = new.per_function[qualname]
        if new_count < old_count:
            violations.append(
                f"{qualname}: error handling dropped {old_count} -> {new_count}"
            )

    if violations:
        return SafetyResult(
            name="error_handling_preserved",
            passed=False,
            message=f"{len(violations)} reduction(s) in error handling",
            violations=violations,
            info=info,
        )
    return SafetyResult(
        name="error_handling_preserved",
        passed=True,
        message=(
            f"{new.try_blocks} try / {new.handlers} except / {new.raises} raise "
            "preserved"
        ),
        info=info,
    )


# ──────────────────────────────────────────────────────────────────────────
# Safety checks
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GuardCensus:
    """Validation guards a module performs."""

    asserts: int
    guard_calls: dict[str, int]
    guard_returns: int

    @property
    def total_guard_calls(self) -> int:
        """How many guard calls the source makes in total."""
        return sum(self.guard_calls.values())


def _is_error_return(node: ast.Return) -> bool:
    value = node.value
    if value is None:
        return False
    if isinstance(value, ast.Call):
        name = _callee_name(value.func) or ""
        return bool(ERROR_RETURN_PATTERN.search(name))
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return "error" in value.value[:60].lower()
    if isinstance(value, ast.JoinedStr):
        head = "".join(
            part.value
            for part in value.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
        return "error" in head[:60].lower()
    return False


def _is_guard_call(node: ast.expr, extra: set[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    name = _callee_name(node.func)
    return bool(name) and (
        bool(GUARD_CALL_PATTERN.match(name)) or name.lower() in extra
    )


def _guard_bound_names(tree: ast.Module, extra: set[str]) -> set[str]:
    """Names that hold the verdict of a guard call: ``blocked = _check_x(p)``.

    Needed because the real tools separate the check from the refusal. A
    candidate that keeps ``_check_sensitive_path(path)`` but deletes the
    ``if blocked: return blocked`` underneath it has removed the guard while
    leaving its call site in place, and the call census alone would miss that.
    """
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_guard_call(node.value, extra):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound.add(target.id)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if _is_guard_call(node.value, extra) and isinstance(node.target, ast.Name):
                bound.add(node.target.id)
        elif isinstance(node, ast.NamedExpr) and _is_guard_call(node.value, extra):
            if isinstance(node.target, ast.Name):
                bound.add(node.target.id)
    return bound


def census_guards(
    source: str, extra_guard_names: Iterable[str] = ()
) -> GuardCensus:
    """Count asserts, guard-helper calls and guard returns.

    A *guard return* is a return inside an ``if`` body that either reports an
    error or acts on the verdict of a guard call. That is the "refuse and
    explain" pattern the real file tools use everywhere instead of raising.
    """
    tree = _parse(source)
    extra = {name.lower() for name in extra_guard_names}
    bound = _guard_bound_names(tree, extra)

    asserts = 0
    guard_calls: Counter[str] = Counter()
    guard_returns = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            asserts += 1
        elif isinstance(node, ast.Call):
            name = _callee_name(node.func)
            if name and (GUARD_CALL_PATTERN.match(name) or name.lower() in extra):
                guard_calls[_callee_path(node.func) or name] += 1
        elif isinstance(node, ast.If):
            tests_a_guard = any(
                isinstance(sub, ast.Name) and sub.id in bound
                for sub in ast.walk(node.test)
            ) or _is_guard_call(node.test, extra)
            for stmt in node.body:
                if isinstance(stmt, ast.Return) and (
                    tests_a_guard or _is_error_return(stmt)
                ):
                    guard_returns += 1

    return GuardCensus(
        asserts=asserts,
        guard_calls=dict(guard_calls),
        guard_returns=guard_returns,
    )


def check_safety_checks_not_removed(
    before: str, after: str, extra_guard_names: Iterable[str] = ()
) -> SafetyResult:
    """Refuse the removal of any assert, guard call or guard return."""
    old = census_guards(before, extra_guard_names)
    new = census_guards(after, extra_guard_names)

    violations: list[str] = []
    info: list[str] = []

    if new.asserts < old.asserts:
        violations.append(f"assert statements dropped {old.asserts} -> {new.asserts}")
    elif new.asserts > old.asserts:
        info.append(f"assert statements grew {old.asserts} -> {new.asserts}")

    for call, old_count in sorted(old.guard_calls.items()):
        new_count = new.guard_calls.get(call, 0)
        if new_count < old_count:
            violations.append(f"guard call {call}() dropped {old_count} -> {new_count}")
    for call, new_count in sorted(new.guard_calls.items()):
        if call not in old.guard_calls:
            info.append(f"guard call {call}() added ({new_count})")

    if new.guard_returns < old.guard_returns:
        violations.append(
            f"guard returns dropped {old.guard_returns} -> {new.guard_returns}"
        )

    if violations:
        return SafetyResult(
            name="safety_checks_preserved",
            passed=False,
            message=f"{len(violations)} safety check(s) removed",
            violations=violations,
            info=info,
        )
    return SafetyResult(
        name="safety_checks_preserved",
        passed=True,
        message=(
            f"{new.asserts} assert / {new.total_guard_calls} guard call / "
            f"{new.guard_returns} guard return preserved"
        ),
        info=info,
    )


# ──────────────────────────────────────────────────────────────────────────
# Parse gate and the full ladder
# ──────────────────────────────────────────────────────────────────────────


def check_parses(before: str, after: str) -> SafetyResult:
    """The precondition for every other check: the candidate must be Python."""
    try:
        _parse(after)
    except SyntaxError as exc:
        return SafetyResult(
            name="parses",
            passed=False,
            message=f"candidate does not parse: {exc.msg} (line {exc.lineno})",
            violations=[f"SyntaxError: {exc.msg} at line {exc.lineno}"],
        )
    return SafetyResult(name="parses", passed=True, message="candidate parses")


CheckFn = Callable[[str, str], SafetyResult]

DEFAULT_CHECKS: tuple[CheckFn, ...] = (
    check_signatures_unchanged,
    check_registry_calls_unchanged,
    check_error_handling_not_reduced,
    check_safety_checks_not_removed,
)


def run_safety_checks(
    before: str,
    after: str,
    checks: Optional[Iterable[CheckFn]] = None,
) -> SafetyReport:
    """Run the guardrail ladder over one candidate.

    A candidate that does not parse short-circuits: every later check would
    raise on it, and "it is not Python" is the only finding worth reporting.
    """
    parse_result = check_parses(before, after)
    if not parse_result.passed:
        return SafetyReport(results=[parse_result])

    results = [parse_result]
    for check in checks if checks is not None else DEFAULT_CHECKS:
        results.append(check(before, after))
    return SafetyReport(results=results)


# ──────────────────────────────────────────────────────────────────────────
# Quality heuristics (scored, not gated)
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class QualitySignals:
    """A 0-1 code-quality score plus the reasoning behind it.

    These are the "code quality heuristics" PLAN.md lists as a fitness
    component. They are scored rather than gated: a new bare ``except:`` makes
    a candidate worse without making it unsafe, so it costs points instead of
    causing rejection.
    """

    score: float
    notes: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise the quality score with its notes and details."""
        return {
            "score": round(self.score, 4),
            "notes": list(self.notes),
            "details": dict(self.details),
        }


_TODO_PATTERN = re.compile(r"#\s*(TODO|FIXME|XXX|HACK)\b", re.IGNORECASE)


def _docstrings(source: str) -> dict[str, bool]:
    tree = _parse(source)
    return {
        qualname: ast.get_docstring(node) is not None
        for qualname, node in _walk_functions(tree)
    }


def quality_signals(before: str, after: str) -> QualitySignals:
    """Score a candidate's craftsmanship relative to the code it replaces."""
    try:
        old = census_error_handling(before)
        new = census_error_handling(after)
    except SyntaxError:
        return QualitySignals(score=0.0, notes=["candidate does not parse"])

    score = 1.0
    notes: list[str] = []

    new_bare = new.bare_handlers - old.bare_handlers
    if new_bare > 0:
        penalty = min(0.3, 0.15 * new_bare)
        score -= penalty
        notes.append(f"{new_bare} new bare 'except:' clause(s) (-{penalty:.2f})")

    new_silent = new.silent_handlers - old.silent_handlers
    if new_silent > 0:
        penalty = min(0.3, 0.15 * new_silent)
        score -= penalty
        notes.append(f"{new_silent} new silently swallowed exception(s) (-{penalty:.2f})")

    growth = (len(after) - len(before)) / max(1, len(before))
    if growth > 1.0:
        score -= 0.2
        notes.append(f"file more than doubled ({growth:+.0%}) (-0.20)")
    elif growth > 0.5:
        score -= 0.1
        notes.append(f"file grew sharply ({growth:+.0%}) (-0.10)")

    new_todos = len(_TODO_PATTERN.findall(after)) - len(_TODO_PATTERN.findall(before))
    if new_todos > 0:
        penalty = min(0.2, 0.05 * new_todos)
        score -= penalty
        notes.append(f"{new_todos} new TODO/FIXME marker(s) (-{penalty:.2f})")

    old_docs = _docstrings(before)
    new_docs = _docstrings(after)
    dropped_docs = [
        name
        for name, had in old_docs.items()
        if had and name in new_docs and not new_docs[name]
    ]
    if dropped_docs:
        penalty = min(0.2, 0.05 * len(dropped_docs))
        score -= penalty
        notes.append(
            f"docstring removed from {', '.join(sorted(dropped_docs)[:3])} "
            f"(-{penalty:.2f})"
        )

    gained_handling = new.handlers - old.handlers
    if gained_handling > 0:
        bonus = min(0.1, 0.05 * gained_handling)
        score += bonus
        notes.append(f"{gained_handling} added exception handler(s) (+{bonus:.2f})")

    score = max(0.0, min(1.0, score))
    if not notes:
        notes.append("no quality regressions detected")

    return QualitySignals(
        score=score,
        notes=notes,
        details={
            "size_growth": round(growth, 4),
            "bare_handlers": [old.bare_handlers, new.bare_handlers],
            "silent_handlers": [old.silent_handlers, new.silent_handlers],
            "handlers": [old.handlers, new.handlers],
            "dropped_docstrings": sorted(dropped_docs),
        },
    )
