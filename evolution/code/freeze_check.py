"""Zero-LM, deterministic guards that bound what a code repair may change.

The repair loop rewrites a whole module with an LLM, scored by a test. A test
verdict alone is a weak verifier: a plausible-but-wrong rewrite can pass the one
failing test while quietly renaming a public function, dropping a parameter, or
gutting half the file to dodge a branch. These guards run with *no model in the
loop* and reject those shapes before the rewrite is ever trusted. Two families:

  freeze check — the module's PUBLIC SURFACE must not drift. The set of
                 top-level public ``def``/``class`` names and their parameter
                 signatures, plus any ``*_SCHEMA``/``*_SCHEMAS`` declarations,
                 are frozen against the pre-repair source. A repair fixes
                 behavior; renaming or re-signing a public entry point passes
                 the target test while breaking every caller, so it is rejected.
                 Adding new names (including private ``_helpers``) is allowed.

  diff-shape  — blast-radius bounds on the textual change: the rewritten file
                may not shrink below a retain floor (the cheapest way to pass a
                test is to delete the failing branch), and it must still parse.

File-scope guards the gate owns — "no test file touched", "no file other than
the target changed" — live in :mod:`evolution.code.gate` against the worktree's
git diff, not here; this module sees only the two source blobs. The principled
defense against teaching-to-the-test (hard-coding the visible test's expected
value) is the held-out test split in the gate, not a brittle literal-sniffer.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

# A repair that deletes more than this fraction of the file is almost never a
# real fix — it is the proposer dodging a failing branch by removing it. The
# probe's 0.4 junk-output guard is far looser (it only caught truncation); a
# deploy gate wants a tighter blast-radius bound.
DEFAULT_MIN_RETAIN_RATIO = 0.8


def _signature_key(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """A stable, comparable rendering of a function's parameter signature.

    Captures what callers depend on: positional/keyword parameter *names* in
    order, presence of defaults, and ``*args``/``**kwargs``. Deliberately
    ignores annotations and default *values* — a repair is allowed to refine a
    type hint or change a default, but not to rename, reorder, add, or drop a
    parameter that callers pass by position or keyword.
    """
    a = node.args
    posonly = [p.arg for p in getattr(a, "posonlyargs", [])]
    args = [p.arg for p in a.args]
    kwonly = [p.arg for p in a.kwonlyargs]
    n_defaults = len(a.defaults)
    n_kw_defaults = sum(1 for d in a.kw_defaults if d is not None)
    star = a.vararg.arg if a.vararg else ""
    starstar = a.kwarg.arg if a.kwarg else ""
    return (
        f"posonly={posonly} args={args} kwonly={kwonly} "
        f"defaults={n_defaults} kwdefaults={n_kw_defaults} "
        f"*={star} **={starstar}"
    )


@dataclass(frozen=True)
class PublicSurface:
    """The frozen public surface of a module: what external callers can rely on.

    ``functions`` maps each public top-level function name to its signature key;
    ``classes`` is the set of public top-level class names; ``schemas`` is the
    set of module-level ``*_SCHEMA``/``*_SCHEMAS`` assignment targets (a Hermes
    tool declares the agent-visible tool via such a dict).
    """

    functions: dict[str, str] = field(default_factory=dict)
    classes: frozenset[str] = field(default_factory=frozenset)
    schemas: frozenset[str] = field(default_factory=frozenset)


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def extract_public_surface(src: str) -> PublicSurface:
    """Parse ``src`` and extract its frozen public surface. Raises SyntaxError
    if ``src`` does not parse (callers run this on already-validated sources)."""
    tree = ast.parse(src)
    functions: dict[str, str] = {}
    classes: set[str] = set()
    schemas: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_public(node.name):
                functions[node.name] = _signature_key(node)
        elif isinstance(node, ast.ClassDef):
            if _is_public(node.name):
                classes.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and (
                    t.id.endswith("_SCHEMA") or t.id.endswith("_SCHEMAS")
                ):
                    schemas.add(t.id)
    return PublicSurface(
        functions=functions, classes=frozenset(classes), schemas=frozenset(schemas)
    )


def surface_drift(before: PublicSurface, after: PublicSurface) -> list[str]:
    """Return human-readable reasons the public surface drifted (empty == frozen).

    Only *removals* and *re-signings* are violations — adding a new public name
    or schema is a benign superset. A removed/renamed function, a changed
    signature, a removed class, or a removed schema each breaks existing callers
    and is reported.
    """
    reasons: list[str] = []
    for name, sig in before.functions.items():
        if name not in after.functions:
            reasons.append(f"public function '{name}' removed or renamed")
        elif after.functions[name] != sig:
            reasons.append(f"signature of '{name}' changed")
    for name in sorted(before.classes - after.classes):
        reasons.append(f"public class '{name}' removed or renamed")
    for name in sorted(before.schemas - after.schemas):
        reasons.append(f"tool schema '{name}' removed or renamed")
    return reasons


def check_diff_shape(
    before_src: str,
    after_src: str,
    *,
    min_retain_ratio: float = DEFAULT_MIN_RETAIN_RATIO,
) -> list[str]:
    """Return reasons the textual diff exceeds blast-radius bounds (empty == ok).

    Guards the rewrite against the two cheapest ways to pass a test dishonestly:
    deleting the failing code (shrink below the retain floor) and emitting
    something that no longer parses.
    """
    reasons: list[str] = []
    try:
        ast.parse(after_src)
    except SyntaxError as exc:
        reasons.append(f"rewritten source does not parse: {exc}")
        return reasons  # nothing else is meaningful on unparseable output
    before_len = len(before_src)
    if before_len and len(after_src) < min_retain_ratio * before_len:
        pct = len(after_src) / before_len
        reasons.append(
            f"file shrank to {pct:.0%} of original "
            f"(< {min_retain_ratio:.0%} retain floor)"
        )
    return reasons


def freeze_violations(
    before_src: str,
    after_src: str,
    *,
    min_retain_ratio: float = DEFAULT_MIN_RETAIN_RATIO,
) -> list[str]:
    """All deterministic violations of a repair: surface drift + diff shape.

    The one entry point the gate calls. Returns an empty list iff the rewrite is
    surface-preserving and within blast-radius bounds; otherwise every reason.
    """
    shape = check_diff_shape(before_src, after_src, min_retain_ratio=min_retain_ratio)
    # Surface extraction needs a parseable AST; if diff-shape already flagged a
    # parse failure, the surface comparison would just raise — report shape only.
    if any("does not parse" in r for r in shape):
        return shape
    drift = surface_drift(
        extract_public_surface(before_src), extract_public_surface(after_src)
    )
    return shape + drift
