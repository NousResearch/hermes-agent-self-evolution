"""Versioned registry of skill candidates for the adaptive strategy catalogue.

A *candidate* is one exact version of a skill file. Baselines are the
already-approved skills; GEPA candidates are proposals produced by
``evolution.skills.evolve_skill`` and registered through
``evolution.adaptive.gepa_adapter``.

Schema (``SCHEMA_VERSION = "candidate-registry/1"``)
---------------------------------------------------
``Candidate``: ``candidate_id``, ``skill_name``, ``parent_id``,
``content_sha256``, ``source`` (``gepa`` | ``baseline``), ``status``
(``proposed`` | ``admitted`` | ``rejected`` | ``retired``), ``created_at``
(UTC ISO 8601) and ``provenance`` (JSON object, informational only).

* ``content_sha256`` is the SHA-256 of the raw UTF-8 bytes of the *full*
  skill file, frontmatter included (GEPA's ``evolved_skill.md`` and
  ``baseline_skill.md`` are both full files). The exact bytes are stored
  and re-checked against the hash whenever they are read.
* ``candidate_id`` is ``"<skill_name>@<first 12 hex of content_sha256>"``.
  Identical content for the same skill is the same candidate.
* Baselines have ``parent_id=None`` and enter as ``admitted``. Every other
  candidate must name an existing parent of the same skill and enters as
  ``proposed``. ``propose`` has no status argument: nothing in this module
  or the GEPA adapter can admit a candidate.
* Status changes only through :meth:`CandidateRegistry.transition`, which
  follows ``ALLOWED_TRANSITIONS``, requires ``reason``, ``decided_by`` and an
  ``evidence`` object, and appends a row to an append-only history.
* ``admitted`` requires a gate 6 admission decision as evidence:
  ``admitted`` is true, ``candidate_id`` matches, and ``policy_sha256`` is a
  fingerprint registered by ``evolution.adaptive.admission`` through
  :func:`register_admission_policy` (done when that module is imported). With
  no registered policy, nothing can be admitted. This module does not import
  the admission gate, so it stays free of scoring code. SQLite triggers refuse edits to history rows and to a
  candidate's identity or content.

Connection to strategies
------------------------
Only ``admitted`` candidates become strategies. :func:`candidate_strategy`
turns a candidate into an ``executor.Strategy`` whose ``toolsets`` are
copied from a fixed base strategy: a candidate carries skill text only and
cannot change tools, budgets, reward or permissions. ``provenance`` (for
example GEPA's own self-reported scores) must never drive admission.

Offline and stdlib-only. Nothing here deploys a skill or writes to a Hermes
installation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from types import MappingProxyType
from typing import Mapping

from .executor import Strategy

SCHEMA_VERSION = "candidate-registry/1"

PROPOSED, ADMITTED, REJECTED, RETIRED = "proposed", "admitted", "rejected", "retired"
STATUSES = (PROPOSED, ADMITTED, REJECTED, RETIRED)
GEPA, BASELINE = "gepa", "baseline"
SOURCES = (GEPA, BASELINE)

# Terminal states (rejected, retired) have no way out; re-proposing the same
# content is refused, so a rejected candidate cannot be laundered back in.
ALLOWED_TRANSITIONS: Mapping[str, frozenset[str]] = MappingProxyType({
    PROPOSED: frozenset({ADMITTED, REJECTED}),
    ADMITTED: frozenset({RETIRED}),
    REJECTED: frozenset(),
    RETIRED: frozenset(),
})

_SKILL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Admission policy fingerprints registered by evolution.adaptive.admission.
_ADMISSION_POLICIES: set[str] = set()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RegistryError(Exception):
    """Base class for registry errors."""


class TransitionError(RegistryError, ValueError):
    """A status change that ALLOWED_TRANSITIONS does not permit."""


class IntegrityError(RegistryError):
    """Stored content no longer matches its recorded hash."""


def register_admission_policy(fingerprint: str) -> None:
    """Accept admission decisions made under this policy fingerprint.

    Called by ``evolution.adaptive.admission`` at import; a static test
    allows no other caller.
    """
    if not isinstance(fingerprint, str) or not _SHA256.match(fingerprint):
        raise ValueError("fingerprint must be a lowercase SHA-256 hex digest")
    _ADMISSION_POLICIES.add(fingerprint)


def registered_admission_policies() -> frozenset[str]:
    return frozenset(_ADMISSION_POLICIES)


def _check_admission_evidence(candidate_id: str, evidence: Mapping) -> None:
    if not _ADMISSION_POLICIES:
        raise TransitionError("no admission policy is registered; admission must go "
                              "through evolution.adaptive.admission")
    if evidence.get("admitted") is not True:
        raise TransitionError("admission evidence must be a decision with admitted=true")
    if evidence.get("policy_sha256") not in _ADMISSION_POLICIES:
        raise TransitionError("admission evidence policy_sha256 is not a registered "
                              "admission policy")
    if evidence.get("candidate_id") != candidate_id:
        raise TransitionError("admission evidence is for a different candidate")


def content_sha256(content: bytes) -> str:
    """Hash rule used for every candidate: SHA-256 of the raw file bytes."""
    if not isinstance(content, (bytes, bytearray)):
        raise TypeError("skill content must be bytes (raw UTF-8 file contents)")
    return hashlib.sha256(bytes(content)).hexdigest()


def candidate_id_for(skill_name: str, sha256: str) -> str:
    return f"{skill_name}@{sha256[:12]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_object(value, name: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    if nonempty and not value:
        raise ValueError(f"{name} must not be empty")
    try:
        return json.dumps(dict(value), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-serialisable without NaN/inf: {exc}") from exc


def _check_text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    skill_name: str
    parent_id: str | None
    content_sha256: str
    source: str
    status: str
    created_at: str
    provenance: Mapping = MappingProxyType({})

    def __hash__(self) -> int:
        return hash(self.candidate_id)

    def to_dict(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, "candidate_id": self.candidate_id,
                "skill_name": self.skill_name, "parent_id": self.parent_id,
                "content_sha256": self.content_sha256, "source": self.source,
                "status": self.status, "created_at": self.created_at,
                "provenance": dict(self.provenance)}


@dataclass(frozen=True)
class Transition:
    candidate_id: str
    from_status: str | None   # None for the initial registration
    to_status: str
    reason: str
    decided_by: str
    evidence: Mapping
    at: str


class CandidateRegistry:
    """SQLite-backed candidate registry. Usable as a context manager."""

    def __init__(self, path: str | Path):
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        try:
            self._migrate()
        except Exception:
            self.conn.close()
            raise

    # ── schema ────────────────────────────────────────────────────────────
    def _migrate(self) -> None:
        statuses = ",".join(f"'{s}'" for s in STATUSES)
        sources = ",".join(f"'{s}'" for s in SOURCES)
        self.conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id TEXT PRIMARY KEY,
            skill_name TEXT NOT NULL,
            parent_id TEXT REFERENCES candidates(candidate_id),
            content_sha256 TEXT NOT NULL,
            source TEXT NOT NULL CHECK (source IN ({sources})),
            status TEXT NOT NULL CHECK (status IN ({statuses})),
            created_at TEXT NOT NULL,
            provenance_json TEXT NOT NULL,
            content BLOB NOT NULL,
            UNIQUE (skill_name, content_sha256)
        );
        CREATE TABLE IF NOT EXISTS transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
            from_status TEXT,
            to_status TEXT NOT NULL,
            reason TEXT NOT NULL,
            decided_by TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            at TEXT NOT NULL
        );
        CREATE TRIGGER IF NOT EXISTS transitions_append_only_u
            BEFORE UPDATE ON transitions
            BEGIN SELECT RAISE(ABORT, 'transition history is append-only'); END;
        CREATE TRIGGER IF NOT EXISTS transitions_append_only_d
            BEFORE DELETE ON transitions
            BEGIN SELECT RAISE(ABORT, 'transition history is append-only'); END;
        CREATE TRIGGER IF NOT EXISTS candidates_no_delete
            BEFORE DELETE ON candidates
            BEGIN SELECT RAISE(ABORT, 'candidates cannot be deleted'); END;
        CREATE TRIGGER IF NOT EXISTS candidates_identity_fixed
            BEFORE UPDATE OF candidate_id, skill_name, parent_id, content_sha256,
                             source, created_at, content ON candidates
            BEGIN SELECT RAISE(ABORT, 'candidate identity and content are immutable'); END;
        """)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.commit()

    # ── registration ──────────────────────────────────────────────────────
    def register_baseline(self, skill_name: str, content: bytes, *,
                          provenance: Mapping | None = None,
                          created_at: str | None = None) -> Candidate:
        """Register an already-approved skill as an admitted baseline.

        Idempotent: the same content returns the existing baseline.
        """
        return self._insert(skill_name, content, parent_id=None, source=BASELINE,
                            status=ADMITTED, provenance=provenance,
                            created_at=created_at,
                            reason="pre-approved baseline skill",
                            decided_by="registry:baseline")

    def propose(self, skill_name: str, content: bytes, *, parent_id: str,
                source: str = GEPA, provenance: Mapping | None = None,
                created_at: str | None = None) -> Candidate:
        """Register a new candidate. It is always ``proposed``, never admitted.

        Idempotent: identical content already proposed returns the existing
        candidate. Content that was rejected or retired is refused.
        """
        if source == BASELINE:
            raise ValueError("use register_baseline for baselines")
        parent = self.get(_check_text(parent_id, "parent_id"))
        if parent.skill_name != skill_name:
            raise ValueError(f"parent {parent_id!r} belongs to skill "
                             f"{parent.skill_name!r}, not {skill_name!r}")
        return self._insert(skill_name, content, parent_id=parent_id, source=source,
                            status=PROPOSED, provenance=provenance,
                            created_at=created_at,
                            reason="candidate proposed", decided_by=f"registry:{source}")

    def _insert(self, skill_name, content, *, parent_id, source, status, provenance,
                created_at, reason, decided_by) -> Candidate:
        if not isinstance(skill_name, str) or not _SKILL_NAME.match(skill_name):
            raise ValueError(f"invalid skill_name {skill_name!r}")
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        sha = content_sha256(content)
        cid = candidate_id_for(skill_name, sha)
        prov_json = _json_object(provenance or {}, "provenance")
        existing = self._row(cid)
        if existing is not None:
            found = self._candidate(existing)
            if found.source != source:
                raise RegistryError(f"{cid} is already registered as {found.source!r}")
            if found.status in (REJECTED, RETIRED):
                raise RegistryError(f"{cid} was {found.status}; it cannot be re-registered")
            return found
        created_at = created_at or _now()
        with self.conn:
            self.conn.execute(
                "INSERT INTO candidates (candidate_id, skill_name, parent_id, content_sha256,"
                " source, status, created_at, provenance_json, content)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (cid, skill_name, parent_id, sha, source, status, created_at,
                 prov_json, bytes(content)))
            self._log(cid, None, status, reason, decided_by, "{}", created_at)
        return self.get(cid)

    # ── status changes ────────────────────────────────────────────────────
    def transition(self, candidate_id: str, to_status: str, *, reason: str,
                   decided_by: str, evidence: Mapping) -> Candidate:
        """Change status along ALLOWED_TRANSITIONS and record why.

        Admission (``admitted``) needs a gate 6 decision as ``evidence`` (see
        the module docstring). This module never calls it.
        """
        current = self.get(candidate_id)
        if to_status not in STATUSES:
            raise TransitionError(f"unknown status {to_status!r}")
        if to_status not in ALLOWED_TRANSITIONS[current.status]:
            raise TransitionError(f"{candidate_id}: {current.status} -> {to_status} "
                                  "is not allowed")
        _check_text(reason, "reason")
        _check_text(decided_by, "decided_by")
        evidence_json = _json_object(evidence, "evidence", nonempty=to_status == ADMITTED)
        self.content(candidate_id)  # refuse to act on tampered content
        if to_status == ADMITTED:
            _check_admission_evidence(candidate_id, evidence)
        at = _now()
        with self.conn:
            updated = self.conn.execute(
                "UPDATE candidates SET status=? WHERE candidate_id=? AND status=?",
                (to_status, candidate_id, current.status)).rowcount
            if updated != 1:
                raise TransitionError(f"{candidate_id} changed status concurrently")
            self._log(candidate_id, current.status, to_status, reason, decided_by,
                      evidence_json, at)
        return self.get(candidate_id)

    def _log(self, cid, from_status, to_status, reason, decided_by, evidence_json, at):
        self.conn.execute(
            "INSERT INTO transitions (candidate_id, from_status, to_status, reason,"
            " decided_by, evidence_json, at) VALUES (?,?,?,?,?,?,?)",
            (cid, from_status, to_status, reason, decided_by, evidence_json, at))

    # ── reads ─────────────────────────────────────────────────────────────
    def _row(self, candidate_id: str):
        return self.conn.execute("SELECT * FROM candidates WHERE candidate_id=?",
                                 (candidate_id,)).fetchone()

    @staticmethod
    def _candidate(row) -> Candidate:
        return Candidate(row["candidate_id"], row["skill_name"], row["parent_id"],
                         row["content_sha256"], row["source"], row["status"],
                         row["created_at"],
                         MappingProxyType(json.loads(row["provenance_json"])))

    def get(self, candidate_id: str) -> Candidate:
        row = self._row(candidate_id)
        if row is None:
            raise KeyError(f"unknown candidate {candidate_id!r}")
        return self._candidate(row)

    def candidates(self, *, skill_name: str | None = None,
                   status: str | None = None) -> tuple[Candidate, ...]:
        query, args = "SELECT * FROM candidates WHERE 1=1", []
        if skill_name is not None:
            query, args = query + " AND skill_name=?", args + [skill_name]
        if status is not None:
            query, args = query + " AND status=?", args + [status]
        rows = self.conn.execute(query + " ORDER BY created_at, candidate_id", args)
        return tuple(self._candidate(r) for r in rows)

    def content(self, candidate_id: str) -> bytes:
        """Stored skill bytes, re-verified against the recorded hash."""
        row = self._row(candidate_id)
        if row is None:
            raise KeyError(f"unknown candidate {candidate_id!r}")
        data = bytes(row["content"])
        if content_sha256(data) != row["content_sha256"]:
            raise IntegrityError(f"{candidate_id}: stored content does not match its hash")
        return data

    def history(self, candidate_id: str | None = None) -> tuple[Transition, ...]:
        query, args = "SELECT * FROM transitions", []
        if candidate_id is not None:
            query, args = query + " WHERE candidate_id=?", [candidate_id]
        rows = self.conn.execute(query + " ORDER BY id", args)
        return tuple(Transition(r["candidate_id"], r["from_status"], r["to_status"],
                                r["reason"], r["decided_by"],
                                MappingProxyType(json.loads(r["evidence_json"])), r["at"])
                     for r in rows)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "CandidateRegistry":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ── Strategy catalogue ────────────────────────────────────────────────────

def candidate_strategy(registry: CandidateRegistry, candidate_id: str,
                       base: Strategy) -> Strategy:
    """Strategy for an admitted candidate. Tools come from ``base`` only."""
    cand = registry.get(candidate_id)
    if cand.status != ADMITTED:
        raise RegistryError(f"{candidate_id} is {cand.status}; only admitted candidates "
                            "may enter the strategy catalogue")
    text = registry.content(candidate_id).decode("utf-8")
    return Strategy(strategy_id=f"{base.strategy_id}:{cand.candidate_id}",
                    system_prompt=text, toolsets=base.toolsets)


def evaluation_strategy(registry: CandidateRegistry, candidate_id: str,
                        base: Strategy) -> Strategy:
    """Strategy for EVALUATING a proposed candidate (or its admitted baseline).

    Same rule as :func:`candidate_strategy` (skill text as system prompt, tools
    from ``base`` only), but the ID is prefixed ``eval:`` so it can never be
    mistaken for a catalogue strategy. It changes no status and never admits;
    only :func:`candidate_strategy` and :func:`admitted_strategies` build
    catalogue strategies, and they accept admitted candidates only.
    """
    cand = registry.get(candidate_id)
    if cand.status not in (PROPOSED, ADMITTED):
        raise RegistryError(f"{candidate_id} is {cand.status}; only proposed candidates "
                            "and admitted baselines can be evaluated")
    text = registry.content(candidate_id).decode("utf-8")
    return Strategy(strategy_id=f"eval:{base.strategy_id}:{cand.candidate_id}",
                    system_prompt=text, toolsets=base.toolsets)


def admitted_strategies(registry: CandidateRegistry, skill_name: str,
                        base: Strategy) -> tuple[Strategy, ...]:
    """Every admitted candidate of a skill as a strategy, oldest first."""
    return tuple(candidate_strategy(registry, c.candidate_id, base)
                 for c in registry.candidates(skill_name=skill_name, status=ADMITTED))

