"""Preference signals — the "alpha imprint" the evolutionary critic learns from.

GEPA is only as good as its critic. Today that critic scores a candidate's
output against a *synthetic* rubric the system invented ("expected_behavior").
That rubric is a guess about what "good" means, and every phase (skills, tools,
prompts, code) inherits the guess.

This module lets the critic learn from something better: the real verdicts of
the agent's users. When a person taps 👍 or 👎 on one of Hermes' replies (an
"imprint"), they are telling the agent, with zero extra effort, "more like
this" or "less like this". Aggregated, those taps are an internalized standard
of what the agent's community approves of — the same role Max Pollard's
imprinting model gives the "alpha imprint": the voice of the tribe that rates
the self without being the self. Here it rates the *evolving* self, so variants
are selected by how well they match what the tribe approved and avoid what it
rejected.

Two design rules keep this sound:

1. The alpha imprint only speaks when the tribe has spoken. A signal influences
   a candidate's score only to the extent that real feedback is relevant to the
   task at hand (:meth:`PreferenceBook.reference_for` returns weight 0 when the
   book is empty or nothing is relevant). With no feedback the fitness is
   byte-for-byte what it is today.

2. It nudges, never overrides. The synthetic rubric stays the backbone; the
   preference signal can only shift the score within a capped budget
   (:func:`blend_preference`), so sparse or noisy feedback cannot hijack
   evolution.

The module is deliberately dependency-free (stdlib only) so the deterministic
core — aggregation, relevance, recency decay, the blend math — is fully
testable without an LLM. The LLM half (scoring a candidate against the
retrieved exemplars) lives in :mod:`evolution.core.fitness`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

APPROVE = "approve"
REJECT = "reject"

# Accept the vocabularies that show up across feedback sources so an importer
# never silently drops a signal over a synonym.
_VALENCE_ALIASES = {
    "approve": APPROVE, "up": APPROVE, "like": APPROVE, "liked": APPROVE,
    "positive": APPROVE, "good": APPROVE, "1": APPROVE, "+1": APPROVE,
    "reject": REJECT, "down": REJECT, "dislike": REJECT, "disliked": REJECT,
    "negative": REJECT, "bad": REJECT, "-1": REJECT, "0": REJECT,
}

_SECONDS_PER_DAY = 86_400.0

# A tiny stoplist so shared filler words ("the", "a", "how") do not inflate the
# overlap between an eval task and a stored reply.
_STOPWORDS = frozenset(
    "a an and are as at be but by for from how i if in is it of on or that the "
    "this to was what when which who why with you your".split()
)
_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_valence(value: str) -> Optional[str]:
    """Map any known feedback vocabulary to APPROVE/REJECT, or None if unknown."""
    if value is None:
        return None
    return _VALENCE_ALIASES.get(str(value).strip().lower())


def _tokens(text: str) -> set[str]:
    """Content tokens of a string (lowercased, stopwords and 1-char noise dropped)."""
    return {t for t in _WORD_RE.findall((text or "").lower()) if t not in _STOPWORDS and len(t) > 1}


def overlap(a: str, b: str) -> float:
    """Jaccard overlap of the content tokens of two strings, in [0, 1]."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if inter == 0:
        return 0.0
    return inter / len(ta | tb)


@dataclass
class PreferenceSignal:
    """One human verdict: a real reply the tribe approved or rejected.

    ``context`` is the prompt that drew the reply, when it is known; imprints
    captured on the reply alone leave it empty, in which case the reply text is
    what relevance matches against (a style signal rather than a topical one).
    """

    response: str
    valence: str
    context: str = ""
    weight: float = 1.0
    ts: float = 0.0
    source: str = "imprint"

    def __post_init__(self) -> None:
        v = normalize_valence(self.valence)
        if v is None:
            raise ValueError(f"unknown preference valence: {self.valence!r}")
        self.valence = v
        # Guard against dirty inputs so downstream math stays in range.
        self.weight = max(0.0, min(1.0, float(self.weight)))

    def key(self) -> tuple:
        """Identity for de-duplication (same reply + verdict = one signal)."""
        return (self.valence, " ".join(_tokens(self.response)) or self.response.strip())


@dataclass
class PreferenceReference:
    """Exemplars the tribe has approved/rejected that bear on a given task,
    plus ``weight`` in [0, 1]: how strongly and how relevantly they bear on it.
    ``weight`` 0 means the tribe has said nothing relevant, so it must not move
    the score.
    """

    approved: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    weight: float = 0.0

    @property
    def is_empty(self) -> bool:
        return self.weight <= 0.0 or (not self.approved and not self.rejected)


class PreferenceBook:
    """The internalized tribe: an aggregate of preference signals.

    Retrieval (:meth:`reference_for`) is deterministic — token overlap for
    relevance, exponential recency decay, and a noisy-OR combination so several
    weak-but-consistent signals corroborate into more confidence without ever
    exceeding 1.
    """

    def __init__(self, signals: Optional[Iterable[PreferenceSignal]] = None):
        self._signals: list[PreferenceSignal] = []
        if signals:
            self.extend(signals)

    def __len__(self) -> int:
        return len(self._signals)

    @property
    def is_empty(self) -> bool:
        return not self._signals

    @property
    def signals(self) -> list[PreferenceSignal]:
        return list(self._signals)

    def add(self, signal: PreferenceSignal) -> None:
        """Add a signal, keeping the most recent when a verdict is repeated."""
        key = signal.key()
        for i, existing in enumerate(self._signals):
            if existing.key() == key:
                if signal.ts >= existing.ts:
                    self._signals[i] = signal
                return
        self._signals.append(signal)

    def extend(self, signals: Iterable[PreferenceSignal]) -> None:
        for s in signals:
            self.add(s)

    def _decay(self, signal: PreferenceSignal, *, now: float, half_life_days: float) -> float:
        """Recency weight in (0, 1]: halves every ``half_life_days``.

        Signals with no timestamp (ts <= 0) are treated as timeless (no decay),
        so imported data without times still counts fully.
        """
        if signal.ts <= 0 or half_life_days <= 0:
            return 1.0
        age_days = max(0.0, (now - signal.ts) / _SECONDS_PER_DAY)
        return 0.5 ** (age_days / half_life_days)

    def reference_for(
        self,
        context: str,
        *,
        max_examples: int = 3,
        min_overlap: float = 0.05,
        half_life_days: float = 30.0,
        now: Optional[float] = None,
    ) -> PreferenceReference:
        """Retrieve the approved/rejected exemplars most relevant to ``context``.

        For each signal, effective strength = overlap x recency-decay x its own
        weight. Signals below ``min_overlap`` are ignored. The top
        ``max_examples`` per side are returned, and the reference weight is the
        noisy-OR of the selected effective strengths, so corroboration raises
        confidence but the result is always in [0, 1].
        """
        now = time.time() if now is None else now
        scored: list[tuple[float, PreferenceSignal]] = []
        for s in self._signals:
            target = s.context if s.context else s.response
            rel = overlap(context, target)
            if rel < min_overlap:
                continue
            eff = rel * self._decay(s, now=now, half_life_days=half_life_days) * s.weight
            if eff > 0:
                scored.append((eff, s))

        if not scored:
            return PreferenceReference()

        scored.sort(key=lambda pair: pair[0], reverse=True)
        approved = [s.response for eff, s in scored if s.valence == APPROVE][:max_examples]
        rejected = [s.response for eff, s in scored if s.valence == REJECT][:max_examples]

        selected = set(approved) | set(rejected)
        selected_effs = [eff for eff, s in scored if s.response in selected]
        weight = _noisy_or(selected_effs)
        return PreferenceReference(approved=approved, rejected=rejected, weight=weight)


def _noisy_or(values: Iterable[float]) -> float:
    """Combine independent evidence in [0, 1]: 1 - prod(1 - v). Bounded [0, 1]."""
    product = 1.0
    for v in values:
        product *= (1.0 - max(0.0, min(1.0, v)))
    return 1.0 - product


def lexical_alignment(candidate: str, reference: "PreferenceReference") -> float:
    """A cheap, deterministic stand-in for the LLM preference judge, in [0, 1].

    Does the candidate reply look more like the approved exemplars than the
    rejected ones? 1.0 = clearly approved-style, 0.5 = neutral (or no signal),
    0.0 = clearly rejected-style. Uses the same token overlap as retrieval, so
    it is free to compute inside a GEPA metric that runs on every candidate.
    """
    approve_sim = max((overlap(candidate, ex) for ex in reference.approved), default=0.0)
    reject_sim = max((overlap(candidate, ex) for ex in reference.rejected), default=0.0)
    return max(0.0, min(1.0, 0.5 + 0.5 * (approve_sim - reject_sim)))


def blend_preference(base: float, alignment: float, weight: float, influence: float) -> float:
    """Fold a preference-alignment score into a base fitness score.

    ``base`` and ``alignment`` are in [0, 1]. ``weight`` (how relevant the
    tribe's verdict is) and ``influence`` (the global cap on how far preference
    may move any score) are in [0, 1]. Let ``share = influence * weight`` and
    ``signed = 2 * alignment - 1`` in [-1, 1] (the direction and strength of the
    verdict, 0 at the neutral midpoint 0.5):

        approved  (signed >= 0):  result = base + share * (1 - base) * signed
        rejected  (signed <  0):  result = base + share * base       * signed

    So a neutral alignment (0.5) never moves the score, approval rewards toward
    1 and rejection penalizes toward 0, the result stays in [0, 1] without
    clamping, and share 0 (no relevant feedback) returns ``base`` unchanged.
    Rewarding toward the *remaining* headroom (1 - base) rather than averaging
    toward ``alignment`` keeps a strong rubric score from being dragged down by
    a merely-neutral verdict.
    """
    base = max(0.0, min(1.0, base))
    alignment = max(0.0, min(1.0, alignment))
    share = max(0.0, min(1.0, influence)) * max(0.0, min(1.0, weight))
    signed = 2.0 * alignment - 1.0
    if signed >= 0.0:
        result = base + share * (1.0 - base) * signed
    else:
        result = base + share * base * signed
    return max(0.0, min(1.0, result))


# ---------------------------------------------------------------------------
# Importers — turn on-disk feedback into a PreferenceBook.
# ---------------------------------------------------------------------------

def _signal_from_imprint(row: dict) -> Optional[PreferenceSignal]:
    """Parse one line of hermes-agent's imprints.jsonl.

    Shape: {"ts", "valence" ("up"/"down"), "excerpt", "session_id",
    "message_id"}. The reply excerpt is the response; the context is left empty
    unless the row carries one (a future join against the transcript can fill
    it), so these act as style signals.
    """
    valence = normalize_valence(row.get("valence", ""))
    response = str(row.get("excerpt") or row.get("response") or "").strip()
    if valence is None or not response:
        return None
    return PreferenceSignal(
        response=response,
        valence=valence,
        context=str(row.get("context") or "").strip(),
        weight=float(row.get("weight", 1.0) or 1.0),
        ts=float(row.get("ts", 0.0) or 0.0),
        source=str(row.get("source") or "imprint"),
    )


def _signal_from_generic(row: dict) -> Optional[PreferenceSignal]:
    """Parse a generic preference row: {"context", "response", "valence",
    optional "weight"/"ts"/"source"}."""
    valence = normalize_valence(row.get("valence", ""))
    response = str(row.get("response") or row.get("excerpt") or "").strip()
    if valence is None or not response:
        return None
    return PreferenceSignal(
        response=response,
        valence=valence,
        context=str(row.get("context") or "").strip(),
        weight=float(row.get("weight", 1.0) or 1.0),
        ts=float(row.get("ts", 0.0) or 0.0),
        source=str(row.get("source") or "preference"),
    )


def load_preference_file(path: Path) -> list[PreferenceSignal]:
    """Read one JSONL feedback file into signals, skipping malformed lines.

    A file named ``imprints.jsonl`` is read with the imprint parser; anything
    else uses the generic parser. Both tolerate the other's field names.
    """
    path = Path(path)
    if not path.exists():
        return []
    parser = _signal_from_imprint if path.name == "imprints.jsonl" else _signal_from_generic
    out: list[PreferenceSignal] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(row, dict):
            continue
        try:
            signal = parser(row)
        except ValueError:
            continue
        if signal is not None:
            out.append(signal)
    return out


def discover_preference_paths(hermes_home: Optional[Path] = None) -> list[Path]:
    """Default on-disk locations for feedback, most authoritative first.

    Looks under the resolved HERMES_HOME (or ~/.hermes) for the imprint log
    that hermes-agent writes.
    """
    import os

    home = Path(hermes_home) if hermes_home else Path(os.getenv("HERMES_HOME") or (Path.home() / ".hermes"))
    candidates = [home / "memories" / "imprints.jsonl"]
    return [p for p in candidates if p.exists()]


def load_preference_book(
    paths: Optional[Iterable[Path]] = None,
    *,
    hermes_home: Optional[Path] = None,
) -> PreferenceBook:
    """Build a PreferenceBook from the given files, or from auto-discovered ones.

    Always returns a book; an empty one when no feedback exists, so callers can
    unconditionally consult it and simply get no influence when the tribe has
    been silent.
    """
    if paths is None:
        paths = discover_preference_paths(hermes_home)
    book = PreferenceBook()
    for path in paths:
        book.extend(load_preference_file(Path(path)))
    return book
