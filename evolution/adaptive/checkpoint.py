"""Policy/skill checkpoints and rollback (KAFO-EVO-001 gate 7).

A checkpoint is a versioned, checksummed snapshot of one strategy-selection
policy (bandit estimates, counts, RNG state, policy version) plus the ID of
the skill version active with it. Checkpoints live in the same SQLite
knowledge store as ``core.ExperienceStore`` (separate tables; the experience
schema is unchanged).

Safety properties:

* **Whitelisted state.** Each policy kind is serialised field by field. On
  restore every field is type- and range-checked, and the policy's
  configuration (strategies, epsilon, LinUCB dim/alpha/ridge, fixed action,
  policy version) must match the live policy exactly. A checkpoint can only
  restore learned estimates, counts and RNG state; it cannot change
  hyperparameters, reward weights, budgets, permissions or evaluation data.
* **Checksummed record.** SHA-256 over canonical JSON of the whole record
  (sequence number, kind, policy version, skill version, creation time,
  validation result and state). The duplicated metadata columns must agree
  with the checksummed payload, so editing a column alone is detected.
  The checksum detects corruption; it is not a signature against a writer
  who can recompute it (the whitelist check above covers that case).
* **Verify before change.** A corrupt or mismatched checkpoint raises
  ``CheckpointCorrupted``/``CheckpointMismatch`` and leaves the live policy
  and the store unchanged (a ``corruption_detected`` event is logged).
* **Atomic, logged rollback.** Status changes, the active-checkpoint pointer
  and the log row are written in one SQLite transaction. The in-memory policy
  is changed only after that transaction commits.
* **Fixed trigger.** ``rollback_decision`` rolls back when post-admission
  validation utility is strictly below the pre-admission checkpoint's utility
  minus ``ROLLBACK_MARGIN``, or when any critical violation occurs.
* **Policy lines.** One store may hold checkpoints for several policies.
  Each checkpoint belongs to the line ``(kind, policy_version)``; the active
  pointer and "last good" are tracked per line, so one policy never rolls
  back onto another's checkpoint. Two live policies with the same kind and
  version must use separate stores.
* **Last good fallback.** Rollback targets the newest ``good`` checkpoint on
  the same line that is older than the rejected one. A target that fails
  verification is marked ``corrupt`` and the next older one is tried. If none
  verifies, rollback fails closed: the policy, the rejected checkpoint and the
  active pointer are unchanged and ``NoGoodCheckpoint`` is raised (corrupt
  targets found on the way stay marked ``corrupt``).

The skill version is an identifier only. Restoring returns it to the caller;
nothing here deploys or rewrites skills.

Validation data reaches this module only through ``benchmark.load_benchmark``
for the ``validation`` split, inside ``benchmark.optimisation_phase()``.
Results from a fake executor or the hand-written benchmark are SYNTHETIC.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import random
import sqlite3
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

from .benchmark import load_benchmark, optimisation_phase
from .compare import CaseBenchmark, OutcomeTable, run_policy
from .core import ContextualBandit, ExperienceStore
from .policies import EpsilonGreedyPolicy, FixedPolicy, LinUCBPolicy, RandomPolicy

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT = 1
# Utility units (valid rewards lie in [-0.30, 1.0]). Fixed: not settable by
# the optimising agent or by a checkpoint.
ROLLBACK_MARGIN = 0.02

STATUS_GOOD = "good"
STATUS_CANDIDATE = "candidate"
STATUS_ROLLED_BACK = "rolled_back"
STATUS_CORRUPT = "corrupt"
STATUSES = (STATUS_GOOD, STATUS_CANDIDATE, STATUS_ROLLED_BACK, STATUS_CORRUPT)

class CheckpointError(Exception):
    """Base class for checkpoint failures."""


class CheckpointCorrupted(CheckpointError):
    """Stored checkpoint fails checksum, parsing or schema validation."""


class CheckpointMismatch(CheckpointError):
    """Checkpoint is intact but belongs to a differently configured policy."""


class NoGoodCheckpoint(CheckpointError):
    """No verified good checkpoint is available to roll back to."""


# ───────────────────────────────────────────────────────────── canonical JSON

def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      ensure_ascii=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ───────────────────────────────────────────────────────────── field checks

def _corrupt(msg: str):
    raise CheckpointCorrupted(msg)


def _real(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        _corrupt(f"{name} must be a finite number")
    return float(value)


def _count(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _corrupt(f"{name} must be an int >= {minimum}")
    return value


def _str(value, name: str) -> str:
    if not isinstance(value, str) or not value:
        _corrupt(f"{name} must be a nonempty string")
    return value


def _dict(value, name: str, keys: set) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        _corrupt(f"{name} must be an object with keys {sorted(keys)}")
    return value


def _list(value, name: str, length: Optional[int] = None) -> list:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        _corrupt(f"{name} must be a list" + ("" if length is None else f" of {length}"))
    return value


def _rng_dump(rng: random.Random) -> list:
    version, internal, gauss = rng.getstate()
    return [version, list(internal), gauss]


def _rng_load(value) -> tuple:
    value = _list(value, "rng", 3)
    _count(value[0], "rng version")
    internal = _list(value[1], "rng state")
    for i, word in enumerate(internal):
        _count(word, f"rng state[{i}]")
    gauss = value[2] if value[2] is None else _real(value[2], "rng gauss")
    state = (value[0], tuple(internal), gauss)
    try:
        random.Random().setstate(state)      # validate on a scratch generator
    except (TypeError, ValueError, OverflowError) as exc:
        raise CheckpointCorrupted(f"invalid rng state: {exc}") from None
    return state


def _same(live, stored, name: str):
    if live != stored:
        raise CheckpointMismatch(f"{name} differs: live {live!r}, checkpoint {stored!r}")


# ───────────────────────────────────────────────────────── per-kind codecs
# dump(policy) -> JSON-safe dict; prepare(policy, state) validates fully and
# returns a zero-argument function that applies the state (cannot fail).

def _bandit_dump(b: ContextualBandit) -> dict:
    keys = sorted(b.counts)
    return {"strategies": list(b.strategies), "epsilon": b.epsilon, "version": b.version,
            "counts": [[c, a, b.counts[(c, a)]] for c, a in keys],
            "means": [[c, a, b.means[(c, a)]] for c, a in keys],
            "rng": _rng_dump(b.rng)}


def _bandit_prepare(b: ContextualBandit, state) -> Callable[[], None]:
    state = _dict(state, "bandit", {"strategies", "epsilon", "version", "counts",
                                    "means", "rng"})
    _same(list(b.strategies), state["strategies"], "strategies")
    _same(b.epsilon, _real(state["epsilon"], "epsilon"), "epsilon")
    _same(b.version, state["version"], "bandit version")
    counts, means = {}, {}
    for label, rows, out, check in (
            ("counts", state["counts"], counts, lambda v, n: _count(v, n, 1)),
            ("means", state["means"], means, _real)):
        for i, row in enumerate(_list(rows, label)):
            ctx, action, value = _list(row, f"{label}[{i}]", 3)
            _str(ctx, f"{label}[{i}] context")
            if action not in b.strategies:
                _corrupt(f"{label}[{i}] has unknown strategy {action!r}")
            if (ctx, action) in out:
                _corrupt(f"{label} repeats {(ctx, action)!r}")
            out[(ctx, action)] = check(value, f"{label}[{i}] value")
    if set(counts) != set(means):
        _corrupt("counts and means cover different (context, strategy) pairs")
    rng = _rng_load(state["rng"])

    def apply():
        b.counts, b.means = counts, means
        b.rng.setstate(rng)
    return apply


def _eg_dump(p: EpsilonGreedyPolicy) -> dict:
    return {"version": p.version, "bandit": _bandit_dump(p.bandit)}


def _eg_prepare(p: EpsilonGreedyPolicy, state) -> Callable[[], None]:
    state = _dict(state, "epsilon-greedy", {"version", "bandit"})
    _same(p.version, state["version"], "policy version")
    return _bandit_prepare(p.bandit, state["bandit"])


def _linucb_dump(p: LinUCBPolicy) -> dict:
    return {"strategies": list(p.strategies), "version": p.version, "dim": p.dim,
            "alpha": p.alpha, "ridge": p.ridge,
            "a_inv": {a: [row[:] for row in p.a_inv[a]] for a in p.strategies},
            "b": {a: p.b[a][:] for a in p.strategies},
            "counts": dict(p.counts), "rng": _rng_dump(p.rng)}


def _linucb_prepare(p: LinUCBPolicy, state) -> Callable[[], None]:
    state = _dict(state, "linucb", {"strategies", "version", "dim", "alpha", "ridge",
                                    "a_inv", "b", "counts", "rng"})
    _same(list(p.strategies), state["strategies"], "strategies")
    _same(p.version, state["version"], "policy version")
    _same(p.dim, _count(state["dim"], "dim", 1), "dim")
    _same(p.alpha, _real(state["alpha"], "alpha"), "alpha")
    _same(p.ridge, _real(state["ridge"], "ridge"), "ridge")
    keys, d = set(p.strategies), p.dim
    a_inv = {a: [[_real(v, f"a_inv[{a}]") for v in _list(row, f"a_inv[{a}] row", d)]
                 for row in _list(m, f"a_inv[{a}]", d)]
             for a, m in _dict(state["a_inv"], "a_inv", keys).items()}
    b = {a: [_real(v, f"b[{a}]") for v in _list(vec, f"b[{a}]", d)]
         for a, vec in _dict(state["b"], "b", keys).items()}
    counts = {a: _count(n, f"counts[{a}]")
              for a, n in _dict(state["counts"], "counts", keys).items()}
    rng = _rng_load(state["rng"])

    def apply():
        p.a_inv = {a: a_inv[a] for a in p.strategies}
        p.b = {a: b[a] for a in p.strategies}
        p.counts = {a: counts[a] for a in p.strategies}
        p.rng.setstate(rng)
    return apply


def _fixed_dump(p: FixedPolicy) -> dict:
    return {"strategies": list(p.strategies), "version": p.version, "action": p.action}


def _fixed_prepare(p: FixedPolicy, state) -> Callable[[], None]:
    state = _dict(state, "fixed", {"strategies", "version", "action"})
    _same(list(p.strategies), state["strategies"], "strategies")
    _same(p.version, state["version"], "policy version")
    _same(p.action, state["action"], "fixed action")
    return lambda: None


def _random_dump(p: RandomPolicy) -> dict:
    return {"strategies": list(p.strategies), "version": p.version, "rng": _rng_dump(p.rng)}


def _random_prepare(p: RandomPolicy, state) -> Callable[[], None]:
    state = _dict(state, "random", {"strategies", "version", "rng"})
    _same(list(p.strategies), state["strategies"], "strategies")
    _same(p.version, state["version"], "policy version")
    rng = _rng_load(state["rng"])
    return lambda: p.rng.setstate(rng)


# Exact types only: a subclass may carry state this codec does not know about.
_CODECS = {
    EpsilonGreedyPolicy: ("epsilon-greedy", _eg_dump, _eg_prepare),
    ContextualBandit: ("contextual-bandit", _bandit_dump, _bandit_prepare),
    LinUCBPolicy: ("linucb", _linucb_dump, _linucb_prepare),
    FixedPolicy: ("fixed", _fixed_dump, _fixed_prepare),
    RandomPolicy: ("random", _random_dump, _random_prepare),
}


def _codec(policy):
    try:
        return _CODECS[type(policy)]
    except KeyError:
        raise TypeError(f"no checkpoint codec for {type(policy).__name__}") from None


def policy_state(policy) -> tuple[str, dict]:
    """(kind, JSON-safe state) for a supported policy."""
    kind, dump, _ = _codec(policy)
    state = dump(policy)
    canonical_json(state)          # fails early on non-finite values
    return kind, state


def prepare_restore(policy, kind: str, state) -> Callable[[], None]:
    """Validate ``state`` against ``policy`` and return the apply step."""
    live_kind, _, prepare = _codec(policy)
    if kind != live_kind:
        raise CheckpointMismatch(f"checkpoint kind {kind!r} != policy kind {live_kind!r}")
    return prepare(policy, state)


# ─────────────────────────────────────────────────────────────── records

@dataclass(frozen=True)
class Validation:
    """Validation-split result used by the rollback trigger."""
    utility: float
    critical_violation: bool
    n_tasks: int = 0
    synthetic: bool = True

    def __post_init__(self):
        if isinstance(self.utility, bool) or not isinstance(self.utility, (int, float)) \
                or not math.isfinite(self.utility):
            raise ValueError("validation utility must be a finite number")
        if not isinstance(self.critical_violation, bool):
            raise ValueError("critical_violation must be a bool")

    def to_dict(self) -> dict:
        return {"utility": float(self.utility), "critical_violation": self.critical_violation,
                "n_tasks": self.n_tasks, "synthetic": self.synthetic}


@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: int
    seq: int
    kind: str
    policy_version: str
    skill_version: str
    label: str
    created_at: str
    validation: Optional[dict]
    state: dict
    sha256: str
    status: str


# ─────────────────────────────────────────────────────────────── store

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS policy_checkpoints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        seq INTEGER NOT NULL UNIQUE,
        kind TEXT NOT NULL,
        policy_version TEXT NOT NULL,
        skill_version TEXT NOT NULL,
        label TEXT NOT NULL,
        created_at TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        status TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS checkpoint_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT NOT NULL,
        checkpoint_id INTEGER,
        from_checkpoint_id INTEGER,
        reason TEXT NOT NULL,
        details_json TEXT NOT NULL,
        logged_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS checkpoint_active (
        line TEXT PRIMARY KEY,
        checkpoint_id INTEGER NOT NULL
    )""",
)


# Written only by the store itself; callers cannot forge them via record_event.
_RESERVED_EVENTS = ("save", "restore", "promote", "rollback", "corruption_detected",
                    "mismatch_skipped")


def policy_line(kind: str, policy_version: str) -> str:
    """Key of the checkpoint history a policy belongs to."""
    return canonical_json([kind, policy_version])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CheckpointStore:
    """Checkpoint tables in the SQLite knowledge store.

    Pass an ``ExperienceStore`` to share its database (and connection), or a
    path. Usable as a context manager when it owns its connection.
    """

    def __init__(self, store: ExperienceStore | str | Path):
        if isinstance(store, ExperienceStore):
            self.conn, self._owns = store.conn, False
        else:
            self.conn, self._owns = sqlite3.connect(str(store)), True
        with self._transaction():
            for ddl in _SCHEMA:
                self.conn.execute(ddl)

    # -- infrastructure -------------------------------------------------
    @contextmanager
    def _transaction(self) -> Iterator[None]:
        if self.conn.in_transaction:
            raise CheckpointError("another transaction is open on this connection")
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.conn.rollback()
            raise
        self.conn.commit()

    def _log(self, event: str, checkpoint_id, from_id, reason: str,
             details: Mapping[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO checkpoint_log (event,checkpoint_id,from_checkpoint_id,reason,"
            "details_json,logged_at) VALUES (?,?,?,?,?,?)",
            (event, checkpoint_id, from_id, reason, canonical_json(dict(details)), _now()))

    def _set_status(self, checkpoint_id: int, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        self.conn.execute("UPDATE policy_checkpoints SET status=? WHERE id=?",
                          (status, checkpoint_id))

    def _set_active(self, line: str, checkpoint_id: int) -> None:
        self.conn.execute("INSERT INTO checkpoint_active (line, checkpoint_id) VALUES (?, ?) "
                          "ON CONFLICT(line) DO UPDATE SET checkpoint_id=excluded.checkpoint_id",
                          (line, checkpoint_id))

    def _active(self, line: str) -> Optional[int]:
        row = self.conn.execute("SELECT checkpoint_id FROM checkpoint_active WHERE line=?",
                                (line,)).fetchone()
        return None if row is None else row[0]

    def _log_corruption(self, checkpoint_id: int, error: Exception, *, mark: bool) -> None:
        logger.error("checkpoint %s failed verification: %s", checkpoint_id, error)
        with self._transaction():
            if mark:
                self._set_status(checkpoint_id, STATUS_CORRUPT)
            self._log("corruption_detected", checkpoint_id, None, str(error),
                      {"error_type": type(error).__name__, "marked_corrupt": mark})

    # -- save / load -----------------------------------------------------
    def save(self, policy, *, skill_version: str, label: str,
             status: str = STATUS_GOOD, validation: Optional[Validation] = None,
             activate: bool = True) -> Checkpoint:
        if not isinstance(skill_version, str) or not skill_version:
            raise ValueError("skill_version must be a nonempty string")
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a nonempty string")
        if status not in (STATUS_GOOD, STATUS_CANDIDATE):
            raise ValueError("new checkpoints are 'good' or 'candidate'")
        kind, state = policy_state(policy)
        created_at = _now()
        with self._transaction():
            seq = self.conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM policy_checkpoints").fetchone()[0]
            payload = canonical_json({
                "format": CHECKPOINT_FORMAT, "seq": seq, "kind": kind,
                "policy_version": policy.version, "skill_version": skill_version,
                "label": label, "created_at": created_at,
                "validation": None if validation is None else validation.to_dict(),
                "state": state})
            digest = _sha256(payload)
            cur = self.conn.execute(
                "INSERT INTO policy_checkpoints (seq,kind,policy_version,skill_version,label,"
                "created_at,payload_json,sha256,status) VALUES (?,?,?,?,?,?,?,?,?)",
                (seq, kind, policy.version, skill_version, label, created_at, payload,
                 digest, status))
            checkpoint_id = cur.lastrowid
            if activate:
                self._set_active(policy_line(kind, policy.version), checkpoint_id)
            self._log("save", checkpoint_id, None, label,
                      {"seq": seq, "sha256": digest, "status": status, "activate": activate})
        return self.load(checkpoint_id)

    def load(self, checkpoint_id: int) -> Checkpoint:
        """Read and verify one checkpoint. Raises ``CheckpointCorrupted``."""
        row = self.conn.execute(
            "SELECT id,seq,kind,policy_version,skill_version,label,created_at,payload_json,"
            "sha256,status FROM policy_checkpoints WHERE id=?", (checkpoint_id,)).fetchone()
        if row is None:
            raise KeyError(f"no checkpoint {checkpoint_id}")
        (cid, seq, kind, policy_version, skill_version, label, created_at, payload,
         digest, status) = row
        if not isinstance(payload, str) or _sha256(payload) != digest:
            raise CheckpointCorrupted(f"checkpoint {cid}: checksum mismatch")
        try:
            doc = json.loads(payload, parse_constant=_reject_constant)
        except ValueError as exc:
            raise CheckpointCorrupted(f"checkpoint {cid}: unreadable payload ({exc})") from None
        doc = _dict(doc, "payload", {"format", "seq", "kind", "policy_version",
                                     "skill_version", "label", "created_at",
                                     "validation", "state"})
        if canonical_json(doc) != payload:
            raise CheckpointCorrupted(f"checkpoint {cid}: payload is not canonical")
        if doc["format"] != CHECKPOINT_FORMAT:
            raise CheckpointCorrupted(f"checkpoint {cid}: unknown format {doc['format']!r}")
        columns = {"seq": seq, "kind": kind, "policy_version": policy_version,
                   "skill_version": skill_version, "label": label, "created_at": created_at}
        for name, value in columns.items():
            if doc[name] != value:
                raise CheckpointCorrupted(
                    f"checkpoint {cid}: column {name} disagrees with checksummed payload")
        if kind not in {k for k, _, _ in _CODECS.values()}:
            raise CheckpointCorrupted(f"checkpoint {cid}: unknown policy kind {kind!r}")
        if status not in STATUSES:
            raise CheckpointCorrupted(f"checkpoint {cid}: unknown status {status!r}")
        validation = doc["validation"]
        if validation is not None:
            validation = _dict(validation, "validation",
                               {"utility", "critical_violation", "n_tasks", "synthetic"})
            _real(validation["utility"], "validation utility")
        return Checkpoint(cid, seq, kind, policy_version, skill_version, label, created_at,
                          validation, doc["state"], digest, status)

    def active_id(self, policy) -> Optional[int]:
        """Active checkpoint on ``policy``'s line, or None."""
        return self._active(policy_line(_codec(policy)[0], policy.version))

    def record_event(self, event: str, *, reason: str, checkpoint_id: Optional[int] = None,
                     from_checkpoint_id: Optional[int] = None,
                     details: Mapping = ()) -> None:
        """Append one row to the checkpoint log (its own transaction)."""
        if not isinstance(event, str) or not event or event in _RESERVED_EVENTS:
            raise ValueError(f"event must be a nonempty string, not one of {_RESERVED_EVENTS}")
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a nonempty string")
        with self._transaction():
            self._log(event, checkpoint_id, from_checkpoint_id, reason, dict(details))

    def log(self) -> list[dict]:
        cur = self.conn.execute(
            "SELECT id,event,checkpoint_id,from_checkpoint_id,reason,details_json,logged_at "
            "FROM checkpoint_log ORDER BY id")
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row)) for row in cur]

    # -- restore / promote / rollback -----------------------------------
    def _verified(self, policy, checkpoint_id: int, *, mark: bool):
        try:
            ckpt = self.load(checkpoint_id)
            apply = prepare_restore(policy, ckpt.kind, ckpt.state)
        except CheckpointCorrupted as exc:
            self._log_corruption(checkpoint_id, exc, mark=mark)
            raise
        return ckpt, apply

    def restore(self, policy, checkpoint_id: int, *, reason: str = "manual restore"
                ) -> Checkpoint:
        """Restore ``policy`` to a checkpoint. Idempotent; logged.

        Returns the checkpoint (its ``skill_version`` is the skill to use).
        On any verification failure, policy and store are unchanged.
        """
        ckpt, apply = self._verified(policy, checkpoint_id, mark=False)
        if ckpt.status in (STATUS_ROLLED_BACK, STATUS_CORRUPT):
            raise CheckpointError(f"checkpoint {checkpoint_id} is {ckpt.status}")
        line = policy_line(ckpt.kind, ckpt.policy_version)
        with self._transaction():
            previous = self._active(line)
            self._set_active(line, ckpt.checkpoint_id)
            self._log("restore", ckpt.checkpoint_id, previous, reason,
                      {"sha256": ckpt.sha256, "skill_version": ckpt.skill_version})
        apply()
        return ckpt

    def promote(self, checkpoint_id: int, *, reason: str, details: Mapping = ()) -> None:
        """Mark a candidate checkpoint good and make it active."""
        ckpt = self.load(checkpoint_id)
        if ckpt.status != STATUS_CANDIDATE:
            raise CheckpointError(f"checkpoint {checkpoint_id} is {ckpt.status}, not candidate")
        line = policy_line(ckpt.kind, ckpt.policy_version)
        with self._transaction():
            previous = self._active(line)
            self._set_status(checkpoint_id, STATUS_GOOD)
            self._set_active(line, checkpoint_id)
            self._log("promote", checkpoint_id, previous, reason, dict(details))

    def last_good(self, kind: str, policy_version: str, *,
                  before_seq: Optional[int] = None) -> list[int]:
        """IDs of good checkpoints on one policy line, newest first."""
        sql = "SELECT id FROM policy_checkpoints WHERE status=? AND kind=? AND policy_version=?"
        args: list = [STATUS_GOOD, kind, policy_version]
        if before_seq is not None:
            sql += " AND seq < ?"
            args.append(before_seq)
        return [r[0] for r in self.conn.execute(sql + " ORDER BY seq DESC", args)]

    def rollback(self, policy, *, from_checkpoint_id: int, reason: str,
                 details: Mapping = ()) -> Checkpoint:
        """Atomically roll back to the last good checkpoint on the same policy
        line that is older than the rejected one.

        In one transaction: rejected checkpoint -> ``rolled_back``, active
        pointer -> target, ``rollback`` log row. The policy is updated only
        after commit. Corrupt targets are marked ``corrupt`` and skipped. If no
        target verifies, raises ``NoGoodCheckpoint``; the policy, the rejected
        checkpoint and the active pointer are unchanged.
        """
        rejected = self.load(from_checkpoint_id)
        line = policy_line(rejected.kind, rejected.policy_version)
        if line != policy_line(_codec(policy)[0], policy.version):
            raise CheckpointMismatch(
                f"checkpoint {from_checkpoint_id} belongs to another policy")
        for target_id in self.last_good(rejected.kind, rejected.policy_version,
                                        before_seq=rejected.seq):
            try:
                target, apply = self._verified(policy, target_id, mark=True)
            except CheckpointCorrupted:
                continue
            except CheckpointMismatch as exc:
                logger.error("checkpoint %s skipped: %s", target_id, exc)
                with self._transaction():
                    self._log("mismatch_skipped", target_id, rejected.checkpoint_id,
                              str(exc), {})
                continue
            with self._transaction():
                self._set_status(rejected.checkpoint_id, STATUS_ROLLED_BACK)
                self._set_active(line, target.checkpoint_id)
                self._log("rollback", target.checkpoint_id, rejected.checkpoint_id, reason,
                          {**dict(details), "to_sha256": target.sha256,
                           "to_skill_version": target.skill_version,
                           "from_skill_version": rejected.skill_version})
            apply()
            logger.warning("rolled back policy %s from checkpoint %s to %s: %s",
                           policy.version, rejected.checkpoint_id, target.checkpoint_id, reason)
            return target
        raise NoGoodCheckpoint(
            f"no verified good checkpoint older than {from_checkpoint_id}; nothing changed")

    def close(self) -> None:
        if self._owns:
            self.conn.close()

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _reject_constant(name: str):
    raise ValueError(f"non-finite constant {name}")


# ──────────────────────────────────────────────────────────── trigger

def rollback_decision(pre_utility: float, post: Validation, *,
                      margin: float = ROLLBACK_MARGIN) -> Optional[str]:
    """Reason to roll back, or None to keep the candidate.

    Rolls back on any critical violation, or when post-admission utility is
    strictly below ``pre_utility - margin``.
    """
    for name, value in (("pre_utility", pre_utility), ("margin", margin)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    if not isinstance(post, Validation):
        raise TypeError("post must be a Validation")
    if post.critical_violation:
        return "critical violation on post-admission validation"
    if post.utility < pre_utility - margin:
        return (f"validation utility {post.utility:.4f} below pre-admission "
                f"{pre_utility:.4f} - margin {margin:g}")
    return None


@dataclass(frozen=True)
class AdmissionResult:
    admitted: bool
    reason: str
    pre_checkpoint: Checkpoint
    candidate_checkpoint: Optional[Checkpoint]
    active_checkpoint: Checkpoint
    skill_version: str
    pre_validation: Validation
    post_validation: Optional[Validation]
    synthetic: bool


def admit_with_rollback(store: CheckpointStore, policy, *, skill_version: str,
                        admit: Callable[[Any], str], validate: Callable[[Any], Validation],
                        label: str = "admission") -> AdmissionResult:
    """Checkpoint, admit a change, validate, then keep it or roll back.

    1. Validate the current policy and save a ``good`` pre-admission checkpoint.
    2. ``admit(policy)`` applies the change and returns the new skill version.
    3. Validate again and save a ``candidate`` checkpoint.
    4. ``rollback_decision`` against the pre-admission checkpoint's recorded
       utility: promote the candidate, or roll back atomically.

    If ``admit`` or post validation raises, the policy is rolled back to the
    pre-admission checkpoint before the error propagates. ``validate`` is
    given a deep copy, so validation never changes the live policy.
    Runs inside ``benchmark.optimisation_phase()``.
    """
    with optimisation_phase():
        return _admit_with_rollback(store, policy, skill_version=skill_version,
                                    admit=admit, validate=validate, label=label)


def _admit_with_rollback(store, policy, *, skill_version, admit, validate, label):
    pre_validation = validate(copy.deepcopy(policy))
    if not isinstance(pre_validation, Validation):
        raise TypeError("validate must return a Validation")
    pre = store.save(policy, skill_version=skill_version, label=f"{label}:pre",
                     validation=pre_validation)
    try:
        new_skill = admit(policy)
        if not isinstance(new_skill, str) or not new_skill:
            raise ValueError("admit must return the new skill version")
        post_validation = validate(copy.deepcopy(policy))
        if not isinstance(post_validation, Validation):
            raise TypeError("validate must return a Validation")
    except BaseException as exc:
        restored = store.restore(policy, pre.checkpoint_id,
                                 reason=f"admission failed: {type(exc).__name__}: {exc}")
        logger.warning("admission %s failed; restored checkpoint %s", label, restored.checkpoint_id)
        raise
    candidate = store.save(policy, skill_version=new_skill, label=f"{label}:candidate",
                           status=STATUS_CANDIDATE, validation=post_validation,
                           activate=False)
    pre_utility = store.load(pre.checkpoint_id).validation["utility"]
    reason = rollback_decision(pre_utility, post_validation)
    synthetic = pre_validation.synthetic or post_validation.synthetic
    details = {"pre_utility": pre_utility, "post_utility": post_validation.utility,
               "margin": ROLLBACK_MARGIN,
               "critical_violation": post_validation.critical_violation,
               "synthetic": synthetic}
    if reason is None:
        store.promote(candidate.checkpoint_id, reason="post-admission validation passed",
                      details=details)
        active = store.load(candidate.checkpoint_id)
        return AdmissionResult(True, "admitted", pre, candidate, active, new_skill,
                               pre_validation, post_validation, synthetic)
    active = store.rollback(policy, from_checkpoint_id=candidate.checkpoint_id,
                            reason=reason, details=details)
    return AdmissionResult(False, reason, pre, store.load(candidate.checkpoint_id), active,
                           active.skill_version, pre_validation, post_validation, synthetic)


# ──────────────────────────────────────────────── validation-split utility

def validation_utility(policy, executor, quality_fn, catalogue, encoder, scoring
                       ) -> Validation:
    """Mean utility and any critical violation of ``policy`` on the validation split.

    Loads only the ``validation`` split through ``benchmark.load_benchmark``
    inside ``optimisation_phase()``. Learning is off and a deep copy of the
    policy is used, so the live policy (including its RNG) is unchanged.
    """
    with optimisation_phase():
        cases = load_benchmark(("validation",))["validation"]
        bench = CaseBenchmark(cases)
        items = bench.items("validation")
        table = OutcomeTable(executor, catalogue, quality_fn, scoring)
        records = run_policy("validation", copy.deepcopy(policy), items, encoder, table,
                             phase="validation", seed=0, learn=False, oracle=False)
        if not records:
            raise ValueError("validation split is empty")
        synthetic = bool(bench.synthetic) or getattr(executor, "backend", None) != "hermes"
        return Validation(utility=sum(r.reward for r in records) / len(records),
                          critical_violation=any(r.violation for r in records),
                          n_tasks=len(records), synthetic=synthetic)
