"""Iterative test-feedback repair engine.

Productionizes the proposer-gradient probe that gave the campaign's first
green: an LLM proposes a whole-file fix from the current (buggy) source plus the
captured failing-test output, the fix is run against the *visible* test split,
and the failure is fed back for another round. On authentic tool bugs this has a
real reachable gradient (most fixes land in a few rounds, none one-shot) — the
one surface where artifact quality is not decoupled from a capable agent's
behavior, because the proposer mutates *executed* code under a *deterministic*
test with no agent between the artifact and the verdict.

The engine only makes the visible test pass; the deploy decision belongs to the
gate. But it folds the zero-LM freeze checks into its acceptance condition and
feeds violations back, so it converges toward surface-preserving fixes rather
than burning rounds on rewrites the gate would reject anyway. A round is
accepted only when the visible test passes *and* the rewrite is surface-clean.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from evolution.code.freeze_check import DEFAULT_MIN_RETAIN_RATIO, freeze_violations
from evolution.code.worktree import WorktreeEnv

# A proposer maps (module_path, current_source, failing_output) -> fixed source,
# or None if it could not produce parseable, non-truncated output this round.
Proposer = Callable[[str, str, str], Optional[str]]

# Loose floor that rejects truncated/junk LM output early (distinct from the
# gate's tighter blast-radius retain floor). Below this the output is not a
# plausible whole-file rewrite at all.
_TRUNCATION_FLOOR = 0.4


@dataclass
class RoundRecord:
    """What happened in one repair round (persisted into the repair trace)."""

    round: int
    proposed: bool
    freeze_violations: list[str] = field(default_factory=list)
    test_passed: bool = False
    output_tail: str = ""


@dataclass
class RepairResult:
    """The candidate a repair run produced, with its full round-by-round trace.

    ``fixed`` is true only when a round both passed the visible test and was
    surface-clean; ``final_source`` is that accepted rewrite. When the loop
    exhausts its rounds, ``fixed`` is false and ``final_source`` is None even if
    some round passed the test while violating the freeze (the gate would reject
    it, so the engine does not present it as a fix).
    """

    fixed: bool
    fixed_round: Optional[int]
    final_source: Optional[str]
    rounds: list[RoundRecord]


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return t


def build_dspy_proposer(lm: object) -> Proposer:
    """Wrap a ``dspy.LM`` as a :data:`Proposer`.

    The signature asks for the COMPLETE corrected module — whole-file rewrites,
    not diffs — which is what made the probe's gradient measurable. Output is
    fence-stripped, truncation-guarded, and AST-validated; an unusable round
    returns None rather than a bad candidate.
    """
    import dspy

    class _Fix(dspy.Signature):
        """Fix a bug in a Python module so its test suite passes. Given the
        current (buggy) module source and the failing pytest output, return the
        COMPLETE corrected module source — the entire file, ready to write.
        Output only the file content (no prose, no markdown fences). Fix the
        module to satisfy the tests; never change the tests, and never rename or
        change the signatures of public (non-underscore) functions or classes."""

        module_path: str = dspy.InputField()
        current_source: str = dspy.InputField(desc="current (buggy) module source")
        failing_test_output: str = dspy.InputField(
            desc="pytest output for the failing test"
        )
        fixed_source: str = dspy.OutputField(desc="the COMPLETE corrected module source")

    proposer = dspy.ChainOfThought(_Fix)

    def _propose(module_path: str, current_source: str, failing_output: str) -> Optional[str]:
        try:
            with dspy.context(lm=lm):
                r = proposer(
                    module_path=module_path,
                    current_source=current_source,
                    failing_test_output=failing_output or "(no test output provided)",
                )
            code = _strip_fences(r.fixed_source)
            if len(code) < len(current_source) * _TRUNCATION_FLOOR:
                return None
            ast.parse(code)
            return code
        except Exception:
            return None

    return _propose


class RepairEngine:
    """Drives the propose → guard → test → feed-back loop in a worktree."""

    def __init__(
        self,
        propose: Proposer,
        *,
        max_rounds: int = 5,
        min_retain_ratio: float = DEFAULT_MIN_RETAIN_RATIO,
    ):
        self.propose = propose
        self.max_rounds = max_rounds
        self.min_retain_ratio = min_retain_ratio

    def repair(
        self, env: WorktreeEnv, tool_relpath: str,
        visible_tests: "str | tuple[str, ...]",
    ) -> RepairResult:
        """Repair ``tool_relpath`` until the visible test target passes.

        ``visible_tests`` is a single test path (held-out-split callers) or a
        tuple of bug-test node-ids (the measurement campaign). The worktree must
        already be authoritative (caller has run
        :meth:`WorktreeEnv.assert_authoritative`). Each round writes the
        candidate, runs only the visible target, and feeds its failure (and any
        freeze violations) back into the next proposal.
        """
        visible = (visible_tests,) if isinstance(visible_tests, str) else tuple(visible_tests)
        original = env.read_tool(tool_relpath)
        current = original
        # Seed the first feedback with the bug's own failing output so round 1
        # has signal, mirroring the probe (blind round-1 fixing fails).
        seed = env.run_test(*visible)
        last_feedback = seed.output
        rounds: list[RoundRecord] = []

        for rnd in range(1, self.max_rounds + 1):
            candidate = self.propose(tool_relpath, current, last_feedback)
            if candidate is None:
                rounds.append(RoundRecord(round=rnd, proposed=False,
                                          output_tail="(proposer returned no usable output)"))
                last_feedback = "Your previous output did not parse or was truncated. " \
                                "Return the COMPLETE corrected module source."
                continue

            violations = freeze_violations(
                original, candidate, min_retain_ratio=self.min_retain_ratio
            )
            if violations:
                # Don't waste a test run on a rewrite the gate would reject;
                # surface the violations so the proposer can preserve the API.
                rounds.append(RoundRecord(round=rnd, proposed=True,
                                          freeze_violations=violations,
                                          output_tail="(rejected before test: freeze violation)"))
                last_feedback = (
                    "Your fix changed the module's public surface, which is not "
                    "allowed:\n- " + "\n- ".join(violations)
                    + "\nKeep all public function/class names and signatures "
                    "exactly as in the original; fix only the behavior."
                )
                current = candidate
                continue

            env.write_tool(tool_relpath, candidate)
            run = env.run_test(*visible)
            rounds.append(RoundRecord(round=rnd, proposed=True, test_passed=run.passed,
                                      output_tail=run.output))
            if run.passed:
                return RepairResult(fixed=True, fixed_round=rnd,
                                    final_source=candidate, rounds=rounds)
            current, last_feedback = candidate, run.output

        return RepairResult(fixed=False, fixed_round=None, final_source=None, rounds=rounds)
