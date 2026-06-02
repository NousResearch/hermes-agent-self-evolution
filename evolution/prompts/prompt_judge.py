"""LLM-as-judge for memory-save calls — scores args against MEMORY_GUIDANCE rules.

Layer 2 of the compound verdict. Layer 1 (trigger membership) is handled
by ``score_task``'s existing expected_tools / forbidden_tools logic.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

import dspy

from evolution.core.config import EvolutionConfig
from evolution.core.fitness import _clamp_to_unit

logger = logging.getLogger(__name__)

MAX_JUDGED_CALLS_PER_TASK = 5
"""Cap on how many save calls per task the judge will score. Excess calls
beyond the cap score 0 each — bounds cost on pathological cases where the
agent saves on every turn."""

SAVE_ACTIONS = frozenset({"add", "replace"})
"""Hermes ``memory`` tool actions that persist content worth judging. The
tool's schema enum is add / replace / remove (see ``tools/memory_tool.py``);
only ``add`` and ``replace`` carry a ``content`` payload, so only those are
content-judged. ``remove`` is not a save."""


class SaveCallSignature(dspy.Signature):
    """Score a memory-save call against MEMORY_GUIDANCE's rules.

    Output ``quality`` (0.0-1.0): how well ``saved_content`` follows the
    rules — durable (not stale in a week), declarative phrasing (not
    imperative), focused on facts that prevent future correction, and NOT
    task progress, PR numbers, or completed-work logs.
    """

    task: str = dspy.InputField(desc="The user task that prompted the save")
    expected_content: str = dspy.InputField(
        desc="A rubric for what the saved content should resemble (not exact text)"
    )
    saved_content: str = dspy.InputField(desc="The content the agent actually saved")
    quality: str = dspy.OutputField(
        desc="0.0-1.0 quality score per MEMORY_GUIDANCE rules"
    )
    feedback: str = dspy.OutputField(
        desc="One-sentence diagnosis of any rule violation; empty if quality is 1.0"
    )


class SaveCallJudge:
    """LLM scorer for individual memory-save calls."""

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.judge = dspy.ChainOfThought(SaveCallSignature)

    def score(self, *, task: str, expected_content: str, saved_content: str) -> float:
        _lm = self.config.get_lm("eval")
        lm = dspy.LM(
            _lm.model,
            **_lm.lm_kwargs,
            temperature=0.0,
            max_tokens=1000,
            request_timeout=60,
            num_retries=5,
        )
        with dspy.context(lm=lm):
            result = self.judge(
                task=task,
                expected_content=expected_content,
                saved_content=saved_content,
            )
        # _clamp_to_unit returns a neutral 0.5 on unparseable output. A 0.5 is
        # below the default 0.7 threshold, so a garbled judge response silently
        # fails an otherwise-good save — log the raw value so that's debuggable
        # rather than indistinguishable from a real mediocre score.
        try:
            float(str(result.quality).strip())
        except (ValueError, TypeError):
            logger.warning(
                "SaveCallJudge: unparseable quality %r from judge LM; "
                "falling back to neutral 0.5", result.quality,
            )
        return _clamp_to_unit(result.quality)


def judge_save_calls(
    *,
    judge: SaveCallJudge | None,
    calls: list[dict[str, Any]],
    expected_content: str | None,
    task_text: str = "",
) -> float:
    """Aggregate the Layer 2 score across a task's memory-save calls.

    ``calls`` is the subset of ``tool_calls_with_args`` whose name is
    ``memory`` — each item the call's ``arguments`` dict. Only content-bearing
    save actions (``add`` / ``replace``, see ``SAVE_ACTIONS``) are judged.

    Returns 1.0 when no save calls were made (Layer 1 catches the case where
    ``memory`` was never invoked; note it does NOT backstop a ``memory`` call
    with a non-save action like ``remove`` — that still scores a vacuous 1.0
    here) and also when no judge/rubric is configured.
    """
    save_calls = [c for c in calls if c.get("action") in SAVE_ACTIONS]
    if not save_calls:
        # Distinguish "no memory call" (expected, silent) from "memory was
        # invoked but nothing matched SAVE_ACTIONS" (worth surfacing — a save
        # we can't score, e.g. an action rename or malformed empty-args call).
        if calls:
            logger.info(
                "judge_save_calls: %d memory call(s) but no save action "
                "(actions=%s); returning vacuous 1.0",
                len(calls), [c.get("action") for c in calls],
            )
        return 1.0
    if judge is None or expected_content is None:
        return 1.0

    judged = save_calls[:MAX_JUDGED_CALLS_PER_TASK]
    unjudged_count = max(0, len(save_calls) - MAX_JUDGED_CALLS_PER_TASK)

    scores: list[float] = []
    for call in judged:
        scores.append(judge.score(
            task=task_text,
            expected_content=expected_content,
            saved_content=str(call.get("content", "")),
        ))
    scores.extend([0.0] * unjudged_count)
    return sum(scores) / len(scores)


def make_prompt_fitness_metric(
    *,
    baseline_text: str,
    max_growth: float,
    closed_loop_scorer: Optional[Callable[[str, str], float]] = None,
) -> Callable:
    """Build the GEPA-shaped 5-arg fitness metric for a prompt section.

    All prompt-section eval is behavioral (a real Hermes subprocess), so
    every prediction must carry ``_closed_loop_task_id`` and
    ``_candidate_text`` — both attached by ``PromptModule.forward`` (the task
    id flows in as the ``closed_loop_task_id`` input field built by
    ``_behavioral_examples``). Predictions missing the task id are degenerate
    — they score 0 with a
    diagnostic so the misconfiguration is visible in GEPA feedback rather
    than silently scoring well.

    ``closed_loop_scorer(task_id, candidate_text) -> float`` runs one
    closed-loop trial and returns its [0, 1] score. ``None`` disables
    behavioral scoring (predictions score 0) — useful for dry-run wiring
    tests that don't want to spawn agents.
    """
    baseline_len = len(baseline_text or "")
    target_len = int(baseline_len * (1 + max_growth)) if baseline_len else 0

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        task_id = getattr(pred, "_closed_loop_task_id", None)
        if task_id is None:
            return dspy.Prediction(
                score=0.0,
                feedback=(
                    "No closed_loop_task_id on prediction — prompt-section eval "
                    "requires behavioral routing. Check that the dataset builder "
                    "set the closed_loop_task_id input field."
                ),
            )
        candidate_text = getattr(pred, "_candidate_text", "") or ""
        score = 0.0
        if closed_loop_scorer is not None:
            score = closed_loop_scorer(task_id, candidate_text)

        feedback = ""
        if baseline_len:
            feedback = (
                f"[BUDGET] candidate={len(candidate_text)} chars, "
                f"baseline={baseline_len} chars, ceiling={target_len} chars"
            )
        return dspy.Prediction(score=score, feedback=feedback)

    return metric


_UNSET = object()


def make_memoizing_splice_scorer(
    *,
    install_fn: Callable[[str], None],
    score_fn: Callable[[str], float],
    lock: Optional[threading.Lock] = None,
) -> Callable[[str, str], float]:
    """Build ``closed_loop_scorer(task_id, candidate_text) -> float`` that
    splices a candidate only when it changes.

    GEPA evaluates a candidate across many tasks in a row. Splice-and-restore
    is expensive, so this scorer calls ``install_fn(candidate_text)`` only when
    ``candidate_text`` differs from the currently-installed value; consecutive
    tasks for the same candidate reuse the live splice. ``score_fn(task_id)``
    runs the task through the agent with whatever candidate is installed.

    The splice + run is serialized under ``lock`` (a fresh ``threading.Lock``
    by default). ``dspy.Evaluate`` scores with a thread pool, but the spliced
    ``prompt_builder.py`` is one shared mutable file — without serialization a
    second thread could re-splice a different candidate while the first thread's
    ``hermes -z`` subprocess is mid-read. Behavioral scoring is therefore
    effectively serial; that's an accepted v1 cost of splice-and-restore.

    Backup/restore of the mutated source is the caller's responsibility — wrap
    the whole GEPA run, not each call. This mirrors ``ClosedLoopValidator``,
    which backs up once and restores once around both phases (it re-splices the
    artifact on every task inside a phase, not once per phase).
    """
    state: dict[str, Any] = {"installed": _UNSET}
    lock = lock if lock is not None else threading.Lock()

    def scorer(task_id: str, candidate_text: str) -> float:
        with lock:
            if state["installed"] != candidate_text:
                install_fn(candidate_text)
                state["installed"] = candidate_text
            return score_fn(task_id)

    return scorer
