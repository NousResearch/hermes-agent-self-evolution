"""Mine Hermes session logs for tool-selection evaluation data.

The tool-path analog of ``evolution.core.external_importers``. Walks
Hermes session JSON via the shared ``iter_hermes_sessions`` helper,
extracts ``(user_task, invoked_tool)`` tuples from
``role: user → role: assistant.tool_calls`` sequences, and (in a follow-up
commit) re-judges each tuple against the *current* manifest with a
confidence-banded LLM judge.

Claude Code's ``~/.claude/history.jsonl`` and Copilot's
``~/.copilot/session-state/*/events.jsonl`` carry only user/assistant text, no
``tool_use`` blocks, so they aren't mined for tool selection.

Claude Code's richer *project* transcripts (``~/.claude/projects/*/*.jsonl``)
DO carry ``tool_use`` blocks, but are deliberately NOT mined here either. A
spike measured why: (1) the only framework-evolvable tools are MCP
descriptions, and within the privacy-safe scope (a single project's own
transcripts) MCP calls are ~0; (2) Claude turns are multi-step (≈14 tool
calls/turn), so judging one call out of a turn with no turn context — the
``(task, invoked_tool, manifest)`` shape below — flips its misselection label
~20% of the time vs. judging it with context, i.e. the label is a context
artifact that would poison GEPA, not a description defect. Turn-level
tool-sequence evaluation (judge whole turns with context, confusable pairs
only) is tracked as a separate roadmap item; the single-call miner is not
viable for Claude.

Tests monkeypatch the session directory via
``evolution.core.external_importers.HermesSessionImporter.SESSION_DIR``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import dspy

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import EvalDataset, EvalExample, split_examples
from evolution.core.external_importers import (
    contains_secret,
    iter_hermes_sessions,
    parse_scoring_json,
)
from evolution.core.hermes_provider import resolve_default_lm
from evolution.tools.tool_source import ToolManifest

logger = logging.getLogger(__name__)


# Mirrors the skill-path threshold in `HermesSessionImporter.extract_messages`.
MIN_TASK_LENGTH = 10


def _extract_tool_name(tool_call: dict) -> Optional[str]:
    """Pull a tool name from a tool_call object across the two shapes
    Hermes sessions emit in practice: OpenAI-style nested
    ``{"function": {"name": ...}}`` and the flat ``{"name": ...}``.
    """
    function = tool_call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if name:
            return name
    return tool_call.get("name")


class HermesToolImporter:
    """Extract ``(task, invoked_tool)`` candidates from Hermes session logs.

    Stateless. Returns ``(candidates, drop_counts)`` so the orchestrator
    can surface the drop breakdown in ``gate_decision.json``.

    Per-user-message rule: scan forward to the next ``role: assistant``
    message *before* the next ``role: user``; if that assistant emitted
    ``tool_calls``, take the **last** call's function name. The last call
    is more often the one that resolved the user's intent — earlier
    calls in a chain tend to be "get oriented" reads.
    """

    @staticmethod
    def extract_candidates(
        manifest: ToolManifest,
        limit: int = 0,
    ) -> tuple[list[dict], dict[str, int]]:
        """Walk all Hermes sessions and return candidate (task, tool) pairs.

        Args:
            manifest: Current manifest. Used to filter out invocations of
                tools that aren't being evolved (and don't exist in the
                manifest under evolution).
            limit: Cap on emitted candidates. ``0`` means no cap.

        Returns:
            ``(candidates, drops)`` where ``candidates`` is a list of
            ``{source, task_input, invoked_tool, session_id}`` dicts and
            ``drops`` is a per-reason counter. The orchestrator passes
            both into the next pipeline stage.
        """
        manifest_names = {t.name for t in manifest.tools}
        candidates: list[dict] = []
        drops = {
            "short_task": 0,
            "slash_command": 0,
            "secret": 0,
            "no_tool_calls": 0,
            "non_manifest": 0,
        }

        for session_id, msg_list in iter_hermes_sessions():
            for i, msg in enumerate(msg_list):
                if msg.get("role") != "user":
                    continue

                user_text = msg.get("content", "") or ""
                if len(user_text) < MIN_TASK_LENGTH:
                    drops["short_task"] += 1
                    continue
                if user_text.lstrip().startswith("/"):
                    drops["slash_command"] += 1
                    continue
                if contains_secret(user_text):
                    drops["secret"] += 1
                    continue

                invoked = _find_invoked_tool(msg_list, start=i + 1)
                if invoked is None:
                    drops["no_tool_calls"] += 1
                    continue
                if invoked not in manifest_names:
                    drops["non_manifest"] += 1
                    continue

                candidates.append({
                    "source": "hermes",
                    "task_input": user_text,
                    "invoked_tool": invoked,
                    "session_id": session_id,
                })

                if limit and len(candidates) >= limit:
                    return candidates, drops

        return candidates, drops


def _find_invoked_tool(messages: list[dict], start: int) -> Optional[str]:
    """Scan forward for the next assistant tool_call, stopping at the
    next user message. Returns the last tool_call's name on that
    assistant turn, or None if no tool call appeared before the next
    user message.
    """
    for j in range(start, len(messages)):
        role = messages[j].get("role")
        if role == "user":
            return None
        if role != "assistant":
            continue
        tool_calls = messages[j].get("tool_calls") or []
        if not tool_calls:
            continue
        last = tool_calls[-1]
        if isinstance(last, dict):
            name = _extract_tool_name(last)
            if name:
                return name
    return None


# Confidence-band thresholds for the judge's misselection label flip.
# The judge reads the same manifest the agent saw, so it shares the
# agent's blind spot for whatever tool is currently being evolved —
# the high threshold is the mitigation. Drop the noisy middle entirely:
# false-flipped labels poison GEPA's reflective feedback.
HIGH_CONFIDENCE_THRESHOLD = 0.85
LOW_CONFIDENCE_THRESHOLD = 0.6

CATEGORY_AGREED = "agreed"
CATEGORY_MISSELECTION = "misselection"


class ToolRelevanceFilter:
    """LLM-judge re-assessment of candidates against the current manifest.

    For each ``(task, invoked_tool)`` candidate, the judge picks the tool
    it would have chosen given the current manifest and reports a
    confidence in [0, 1]. The label decision table:

    ===========================  ===========================  ===========
    Judge state                  Action                        Category
    ===========================  ===========================  ===========
    ``relevant=False``           drop                          —
    agree, any confidence        keep                          ``agreed``
    disagree, conf ≥ 0.85        keep with flipped label       ``misselection``
    disagree, 0.6 ≤ conf < 0.85  drop (noisy middle)           —
    disagree, conf < 0.6         drop                          —
    ===========================  ===========================  ===========

    The judge can hallucinate a ``correct_tool`` not in the manifest;
    those disagreements are dropped with their own counter.
    """

    class ScoreToolSelection(dspy.Signature):
        """Score whether the invoked tool was the right choice for the task.

        Return JSON with:
        - relevant: boolean (is this a realistic tool-selection task at all?)
        - correct_tool: string (which tool from the manifest you would pick)
        - confidence: float in [0, 1] (how confident you are in correct_tool)
        """
        task: str = dspy.InputField(desc="The user's task")
        invoked_tool: str = dspy.InputField(desc="The tool the agent actually invoked")
        manifest_summary: str = dspy.InputField(
            desc="Current manifest: each line is '- <tool_name>: <description>'"
        )
        scoring: str = dspy.OutputField(
            desc="JSON object with: relevant, correct_tool, confidence"
        )

    def __init__(self, model: str, manifest: ToolManifest, seed: int = 42):
        self.scorer = dspy.ChainOfThought(self.ScoreToolSelection)
        self.model = model
        self.manifest = manifest
        self.manifest_names = {t.name for t in manifest.tools}
        self.seed = seed
        self._manifest_summary = _render_manifest_summary(manifest)

    def filter_and_score(
        self,
        candidates: list[dict],
        max_examples: int = 200,
    ) -> tuple[list[EvalExample], dict[str, int]]:
        """Score candidates; emit EvalExamples per the band rule.

        Cost ceiling: at most ``max_examples * 2`` judge calls. Stops
        early once ``max_examples`` examples have been collected.

        Returns ``(examples, band_drops)`` where ``band_drops`` counts
        the reasons candidates were rejected by the judge layer.
        """
        examples: list[EvalExample] = []
        band_drops = {
            "judge_irrelevant": 0,
            "judge_error": 0,
            "noisy_middle": 0,
            "low_confidence": 0,
            "unknown_correct_tool": 0,
        }

        _lm = resolve_default_lm(role="judge", explicit_model=self.model)
        lm = dspy.LM(_lm.model, **_lm.lm_kwargs, temperature=0.0, max_tokens=2000)
        budget = candidates[: max_examples * 2]

        for cand in budget:
            scoring = self._score_one(cand, lm)
            if scoring is None:
                band_drops["judge_error"] += 1
                continue
            if not scoring.get("relevant", False):
                band_drops["judge_irrelevant"] += 1
                continue

            correct_tool = (scoring.get("correct_tool") or "").strip()
            try:
                confidence = float(scoring.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0

            invoked = cand["invoked_tool"]
            decision = self._decide(invoked, correct_tool, confidence)
            if decision is None:
                # Bookkeeping for the drop band.
                if correct_tool and correct_tool != invoked and correct_tool not in self.manifest_names:
                    band_drops["unknown_correct_tool"] += 1
                elif confidence < LOW_CONFIDENCE_THRESHOLD:
                    band_drops["low_confidence"] += 1
                else:
                    band_drops["noisy_middle"] += 1
                continue

            expected, category = decision
            examples.append(EvalExample(
                task_input=cand["task_input"][:2000],
                expected_behavior=expected,
                difficulty="medium",
                category=category,
                source=cand["source"],
            ))

            if len(examples) >= max_examples:
                break

        return examples, band_drops

    def _score_one(self, candidate: dict, lm) -> Optional[dict]:
        try:
            with dspy.context(lm=lm):
                result = self.scorer(
                    task=candidate["task_input"][:1000],
                    invoked_tool=candidate["invoked_tool"],
                    manifest_summary=self._manifest_summary,
                )
        except Exception:
            return None
        return parse_scoring_json(result.scoring)

    def _decide(
        self,
        invoked: str,
        correct_tool: str,
        confidence: float,
    ) -> Optional[tuple[str, str]]:
        """Apply the confidence-band decision table.

        Returns ``(expected_behavior, category)`` if the example is kept,
        ``None`` if dropped (the caller bookkeeps which band it fell in).
        """
        if correct_tool == invoked:
            return invoked, CATEGORY_AGREED
        if correct_tool not in self.manifest_names:
            return None
        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            return correct_tool, CATEGORY_MISSELECTION
        return None


def _render_manifest_summary(manifest: ToolManifest) -> str:
    """Render the manifest as the judge's reference list."""
    return "\n".join(f"- {t.name}: {t.description}" for t in manifest.tools)


def build_tool_dataset_from_sessions(
    manifest: ToolManifest,
    target_tool: str,
    output_path: Optional[Path],
    model: str,
    max_examples: int = 200,
    seed: int = 42,
) -> tuple[EvalDataset, dict[str, int]]:
    """Mine Hermes sessions, re-judge against the current manifest, and split.

    ``target_tool`` is informational in v1 (used for logging only) — the
    sampling is uniform across whatever the importer surfaces. v2 may bias
    toward examples whose ``expected_behavior == target_tool``.

    Returns ``(dataset, drops)`` where ``drops`` is a flat dict combining
    the importer-stage and judge-stage drop counters:

    Importer-stage keys (set by ``HermesToolImporter``):
      short_task, slash_command, secret, no_tool_calls, non_manifest

    Judge-stage keys (set by ``ToolRelevanceFilter``):
      judge_irrelevant, judge_error, noisy_middle, low_confidence,
      unknown_correct_tool

    Both stages contribute to the dataset audit so the operator can see
    why N candidates became M examples. The ``non_manifest`` count
    specifically lands in ``gate_decision.json.dataset`` as
    ``dropped_non_manifest_count`` (the most operator-relevant figure).

    Writes per-split JSONL to ``output_path`` if given.
    """
    candidates, importer_drops = HermesToolImporter.extract_candidates(
        manifest, limit=max_examples * 2,
    )
    logger.info(
        "tool sessiondb: target=%s — found %d candidates from Hermes sessions; "
        "importer drops: %s",
        target_tool, len(candidates), importer_drops,
    )

    if not candidates:
        return EvalDataset(), dict(importer_drops)

    judge = ToolRelevanceFilter(model=model, manifest=manifest, seed=seed)
    examples, judge_drops = judge.filter_and_score(candidates, max_examples=max_examples)
    logger.info(
        "tool sessiondb: target=%s — judge produced %d examples; "
        "judge drops: %s",
        target_tool, len(examples), judge_drops,
    )

    drops = {**importer_drops, **judge_drops}

    if not examples:
        return EvalDataset(), drops

    config = EvolutionConfig()
    dataset = split_examples(
        examples,
        seed=seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        holdout_ratio=config.holdout_ratio,
    )

    if output_path is not None:
        dataset.save(output_path)

    return dataset, drops
