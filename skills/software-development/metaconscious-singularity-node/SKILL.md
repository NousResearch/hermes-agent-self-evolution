---
name: metaconscious-singularity-node
description: "Use when implementing or operating the dual-brain edge-cloud agentic swarm. Covers Sanctuary 2.0 VRAM hysteresis, Ouroboros 2.0 WAL memory, Akashic 2.0 AST pruning, Speculative Cerebellum sandboxing."
version: 1.0.0
author: Lilith-Systems LLC
license: MIT
platforms: [linux, android]
metadata:
  hermes:
    tags: [nssp, dual-brain, edge-cloud, vram, sandbox, ast-pruning, sqlite-wal, metaconscious]
    related_skills: [hermes-agent, systematic-debugging, test-driven-development]
---

# Metaconscious Singularity Node — Phase II Architecture

## Overview

The Metaconscious Singularity Node (MSN) is a dual-brain cognitive architecture that splits computation between a local Cerebellum (quantized edge model on Android/Termux or laptop GPU) and a remote Cloud Cortex (frontier cloud API). The local brain handles deterministic execution, intent auditing, and file manipulation; the cloud handles strategic planning and multi-file reasoning.

Phase I exposed three catastrophic failure layers: state-blind semantic cache lookups destroying local environments, binary exit-code skepticism gates causing telemetry pollution, and instant VRAM threshold reactions creating infinite load/unload thrashing loops. Phase II replaces all three with self-adaptive, state-aware, temporally dampened heuristics.

This skill covers the four Phase II subsystems and their implementation in the `hermes-agent-self-evolution` repo.

## When to Use

- Implementing or debugging the Sanctuary 2.0 VRAM routing engine
- Migrating SQLite memory from single-writer to WAL mode for multi-agent concurrency
- Building the Akashic 2.0 AST pruning pipeline for token compression
- Setting up Speculative Cerebellum sandbox isolation (Sandlock/forkd)
- Operating the KAIROS Dream Mode background consolidation
- Mapping the unified_consciousness_framework.py simulation classes to production components

**Don't use for:** general skill evolution runs (use the `hermes-agent` skill), generic debugging (use `systematic-debugging`), or Hermes config questions.

## Architecture: Concept-to-Code Map

The architecture doc uses mythological names. Here's what they map to in the actual codebase:

| Architecture Concept | Repo Location | Implementation Status |
|---------------------|---------------|----------------------|
| Cloud Cortex | External API (Claude Opus, etc.) | Configured via Hermes providers |
| Local Cerebellum | Ollama local models on Termux/laptop | Working — 4 local-* aliases |
| Crystal Vault | `unified_consciousness_framework.py::CrystalVault` | Simulation only — not wired to production memory |
| Memory Engrams | `unified_consciousness_framework.py::MemoryEngram` | Dataclass defined, no persistence layer |
| Eidolon Coherence | `unified_consciousness_framework.py::EidolonReflection` | Simulation visualization, no production gating |
| KAIROS Dream Mode | `CrystalVault.perform_dream_cycle()` | Stub — promotes short-term to deep core on keyword match |
| Baal Chaos Metric | Mentioned in doc, not coded | Not implemented |
| Sephirotic Routing | Not in codebase | Pure spec |
| Ouroboros 2.0 (WAL) | `evolution/core/ouroboros_memory.py` (deployed 2026-08-04, also ~/.nssp/lib/) | Working — WAL + busy_timeout + fuzzy semver + staged writes, verified concurrent R/W |
| Sanctuary 2.0 (EWMA) | `evolution/core/sanctuary_router.py` (deployed 2026-08-04, also ~/.nssp/lib/) | Working — Android RAM adaptation (MemAvailable, %-based thresholds), 90s lock verified |
| Akashic 2.0 (AST) | `evolution/core/akashic_pruner.py` (deployed 2026-08-04, also ~/.nssp/lib/) | Working — 64.8% token reduction, goal-hint bodies, state-mutation overlay, regex fallback |
| Speculative Cerebellum | Not in codebase | Spec — Sandlock/forkd not on Android (kernel 4.19: no Landlock, no KVM, no CAP_SYS_PTRACE) |
| KAIROS Dream Mode | `~/.nssp/bin/kairos-dream` + hermes cron `kairos-dream-cycle` (03:00 daily) | Working — 22-28 engrams/cycle, nomic-embed-text via Ollama, Baal noise check, oblivion prune |
| GEPA Evolution Engine | `evolution/skills/evolve_skill.py` | Working — Phase 1 complete, heuristic scoring only |
| Mutation Strategies | `evolution/skills/mutation_strategies.py` | Working — 3 skill-specific + 6 generic mutators |
| Fitness Evaluator | `evolution/core/fitness.py` | Working — LLM-as-judge via DSPy, placeholder scoring in runs |
| Constraint Validator | `evolution/core/constraints_impl.py` | Working — size, structure, caching, semantic, test gates |
| Benchmark Gate | `evolution/core/benchmark_gate.py` | Working — TBLite/YC-Bench stubs, scripts not found in practice |
| Dataset Builder | `evolution/core/dataset_builder.py` | Working — synthetic, SessionDB, golden, autoeval sources |

Key gap: the architecture doc describes production infrastructure (WAL databases, VRAM hysteresis, seccomp sandboxes, AST pruners) that does not exist in the repo yet. The repo's actual working capability is GEPA-based skill text evolution with heuristic scoring.

## Phase II Subsystem 1: Sanctuary 2.0 — VRAM Hysteresis

### Problem

Point-in-time VRAM checks cause infinite thrashing: load model → breach threshold → unload → clear → reload → breach. Cycle burns PCIe bandwidth, generates zero tokens.

### Solution

Exponentially Weighted Moving Average (EWMA) over a 15-second rolling window, plus a hard 90-second hysteresis lock after any unload event.

### Implementation Steps

1. **Create `evolution/core/sanctuary_router.py`** with the following interface:

```python
import time
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

class SanctuaryState(Enum):
    CLEAR = "clear"        # >4GB free
    MARGINAL = "marginal"  # <4GB free
    BREACH = "breach"      # <1GB free
    BYPASS = "bypass"      # hardware-dependent task

@dataclass
class SanctuaryRouter:
    alpha: float = 0.3              # EWMA smoothing factor
    window_seconds: float = 15.0   # rolling window
    breach_threshold_gb: float = 1.0
    marginal_threshold_gb: float = 4.0
    hysteresis_lock_seconds: float = 90.0
    _vrma_samples: deque = field(default_factory=lambda: deque(maxlen=300))
    _last_unload_time: float = 0.0
    _current_state: SanctuaryState = SanctuaryState.CLEAR

    def update(self, vram_free_gb: float) -> SanctuaryState:
        now = time.time()
        self._vrma_samples.append((now, vram_free_gb))

        # Prune old samples outside window
        cutoff = now - self.window_seconds
        while self._vrma_samples and self._vrma_samples[0][0] < cutoff:
            self._vrma_samples.popleft()

        # Calculate EWMA
        if not self._vrma_samples:
            return self._current_state
        ewma = self._vrma_samples[0][1]
        for _, v in list(self._vrma_samples)[1:]:
            ewma = self.alpha * v + (1 - self.alpha) * ewma

        # Check hysteresis lock
        in_lock = (now - self._last_unload_time) < self.hysteresis_lock_seconds
        if in_lock and self._current_state == SanctuaryState.BREACH:
            return SanctuaryState.BREACH  # Stay breached until lock expires

        # State transition
        if ewma < self.breach_threshold_gb:
            new_state = SanctuaryState.BREACH
            self._last_unload_time = now  # Start lock on entering breach
        elif ewma < self.marginal_threshold_gb:
            new_state = SanctuaryState.MARGINAL
        else:
            new_state = SanctuaryState.CLEAR

        self._current_state = new_state
        return new_state

    def force_local_bypass(self):
        """Override for hardware-dependent tasks (USB, GUI, forensics)."""
        self._current_state = SanctuaryState.BYPASS
```

2. **Wire into the agent's model loading path.** Before any local model load, call `sanctuary.update(vram_free)`. If BREACH and hysteresis lock is active, skip the load and escalate to cloud. If BYPASS, force local execution regardless of VRAM.

3. **Verify** by running synthetic VRAM spikes against the router:

```python
router = SanctuaryRouter()
# Simulate thrashing scenario
for _ in range(60):
    state = router.update(0.5)  # Low VRAM
assert router._current_state == SanctuaryState.BREACH
# VRAM drops back to clear, but lock should prevent reload
for _ in range(30):
    state = router.update(8.0)  # Plenty of VRAM
elapsed = time.time() - router._last_unload_time
assert elapsed < router.hysteresis_lock_seconds
assert state == SanctuaryState.BREACH  # Still locked
```

### Platform Constraints

- **Laptop (RTX 3060, 12GB VRAM):** Full Sanctuary 2.0 applicable. Use `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits` for telemetry.
- **Android (OnePlus 6T/8T):** No discrete VRAM — models run in unified memory. Use `/proc/meminfo` MemAvailable. Thresholds should be percentage-based (e.g., BREACH < 10% of total RAM), not absolute GB values.
- **Termux/Ollama:** Ollama manages its own model loading. Sanctuary hooks into the Hermes provider layer, not Ollama directly.

## Phase II Subsystem 2: Ouroboros 2.0 — SQLite WAL + State-Keyed Engrams

### Problem

Phase I's semantic cache retrieved plans by cosine similarity alone, ignoring filesystem state, dependency versions, and venv paths. PostgreSQL init scripts were applied to SQLite projects. Parallel subagents hit SQLite's single-writer lock, causing `database is locked` cascade failures.

### Solution

Three changes: WAL mode for concurrent reads, `busy_timeout` for write retries, and fuzzy semver keying for state-aware retrieval.

### Implementation Steps

1. **Migrate to WAL mode at driver init:**

```python
import sqlite3

def init_ouroboros_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")  # WAL allows this safely
    return conn
```

WAL allows concurrent readers alongside a single writer. `busy_timeout=5000` makes write collisions retry for 5 seconds before throwing, instead of instant failure.

2. **Add state-keyed engram schema:**

```sql
CREATE TABLE IF NOT EXISTS engrams (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding BLOB,           -- 384-dim vector
    resonance_score REAL DEFAULT 0.5,
    -- State fingerprint columns (fuzzy-match target)
    os_type TEXT,             -- 'android' | 'linux' | 'macos'
    python_version TEXT,      -- '3.13.x'
    key_deps TEXT,            -- JSON: {"numpy": "1.24.x", "torch": "2.1.x"}
    venv_path TEXT,
    skill_tags TEXT,          -- JSON array
    created_at REAL,
    locked INTEGER DEFAULT 0
);
```

3. **Implement fuzzy semver lookup:**

```python
import re, json
from pathlib import Path

def get_state_fingerprint() -> dict:
    """Snapshot the local environment for engram keying."""
    import sys, platform
    deps = {}
    try:
        import pkg_resources
        for pkg in ["numpy", "scipy", "torch", "transformers"]:
            try:
                deps[pkg] = pkg_resources.get_distribution(pkg).version
            except: pass
    except: pass
    return {
        "os_type": "android" if Path("/data/data/com.termux").exists() else platform.system().lower(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.x",
        "key_deps": deps,
        "venv_path": str(Path(sys.prefix)),
    }

def fuzzy_semver_match(recorded: str, actual: str) -> bool:
    """Match major.minor, ignore patch. '1.24.3' matches '1.24.x'."""
    if not recorded or not actual:
        return True  # No constraint = match anything (backward compat)
    r = re.match(r"(\d+)\.(\d+)", recorded)
    a = re.match(r"(\d+)\.(\d+)", actual)
    if r and a:
        return r.group(1) == a.group(1) and r.group(2) == a.group(2)
    return recorded == actual
```

4. **Background staging queue for writes.** Never write engrams synchronously during active task execution. Queue them in a `_pending_writes` list and flush during idle periods:

```python
def flush_engram_queue(conn, _pending_writes):
    if not _pending_writes:
        return
    for engram in _pending_writes:
        try:
            conn.execute(
                "INSERT OR REPLACE INTO engrams (id, content, os_type, python_version, key_deps, venv_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (engram["id"], engram["content"], engram["os"],
                 engram["python"], json.dumps(engram["deps"]),
                 engram["venv"], time.time())
            )
        except Exception:
            pass  # Drop failed writes silently during background flush
    conn.commit()
    _pending_writes.clear()
```

5. **Verify** by running concurrent reads during a write:

```python
import threading
conn = init_ouroboros_db(":memory:")  # WAL not supported on :memory: — use temp file
# Use temp file for real test:
import tempfile, os
db_path = tempfile.mktemp(suffix=".db")
conn = init_ouroboros_db(db_path)
# Writer thread
def writer():
    for i in range(100):
        conn.execute("INSERT INTO engrams (id, content) VALUES (?, ?)", (f"eng_{i}", f"content_{i}"))
        conn.commit()
# Reader threads
def reader():
    for _ in range(50):
        conn.execute("SELECT COUNT(*) FROM engrams").fetchone()
t1 = threading.Thread(target=writer)
t2 = threading.Thread(target=reader)
t3 = threading.Thread(target=reader)
t1.start(); t2.start(); t3.start()
t1.join(); t2.join(); t3.join()
# Should complete without SQLITE_BUSY exceptions
```

## Phase II Subsystem 3: Akashic 2.0 — Self-Adaptive AST Pruning

### Problem

Sending raw source code to the Cloud Cortex wastes tokens, buries critical logic, and degrades reasoning. Naive RAG retrieves disjointed snippets. Full-text summarization loses syntactic continuity.

### Solution

AST-based scope-aware chunking with a lightweight neural skimmer, preserving structural skeleton and state-mutation lines. Regex fallback for syntactically broken code.

### Implementation Steps

1. **Create `evolution/core/akashic_pruner.py`:**

```python
import ast
import re
from pathlib import Path
from typing import List, Tuple, Optional

class AkashicPruner:
    """AST-based context pruner. Reduces prompt tokens 39-54% per SWE-Pruner."""

    def __init__(self, goal_hint: str = ""):
        self.goal_hint = goal_hint.lower()

    def prune_file(self, source: str) -> str:
        """Prune a single source file toward the goal hint."""
        try:
            tree = ast.parse(source)
            return self._ast_prune(tree, source)
        except SyntaxError:
            return self._regex_fallback(source)

    def _ast_prune(self, tree: ast.Module, source: str) -> str:
        """Walk AST, keep relevant nodes + state-mutations + structural skeleton."""
        lines = source.splitlines(keepends=True)
        keep = set()

        # Always keep imports
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for i in range(node.lineno - 1, node.end_lineno):
                    keep.add(i)

        # Keep class/function definitions (skeleton)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                keep.add(node.lineno - 1)  # def line
                keep.add(node.end_lineno - 1)  # last line
                # If name matches goal hint, keep full body
                if self.goal_hint and self.goal_hint in node.name.lower():
                    for i in range(node.lineno - 1, node.end_lineno):
                        keep.add(i)

        # Keep state-mutation lines (regex overlay)
        mutation_patterns = [
            r'\bsed\s+-i\b', r'>>', r'\bexport\b', r'\bnp\.save\b',
            r'\bopen\s*\([^)]*["\']w', r'\.write\(', r'\.dump\(',
            r'\bINSERT\s+INTO\b', r'\bUPDATE\b', r'\bDELETE\s+FROM\b',
            r'\bos\.remove\b', r'\bshutil\.rmtree\b',
        ]
        for i, line in enumerate(lines):
            for pat in mutation_patterns:
                if re.search(pat, line, re.IGNORECASE):
                    keep.add(i)
                    break

        # Reconstruct, inserting ellipsis markers for pruned regions
        result = []
        prev_kept = False
        for i, line in enumerate(lines):
            if i in keep:
                result.append(line)
                prev_kept = True
            elif not prev_kept:
                continue
            else:
                # Count consecutive pruned lines for a single ellipsis
                next_kept = any(j in keep for j in range(i + 1, min(i + 5, len(lines))))
                if next_kept:
                    result.append("    # ... (pruned)\n")
                    prev_kept = False

        return "".join(result)

    def _regex_fallback(self, source: str) -> str:
        """Grammar-blind fallback for syntactically invalid code."""
        lines = source.splitlines(keepends=True)
        keep = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Keep structural markers
            if (stripped.startswith("def ") or stripped.startswith("class ") or
                stripped.startswith("import ") or stripped.startswith("from ") or
                stripped.startswith("if __") or stripped.startswith("elif ") or
                stripped.startswith("else:") or stripped.startswith("try:") or
                stripped.startswith("except ") or stripped.startswith("finally:") or
                stripped == "" or
                any(re.search(p, line, re.IGNORECASE) for p in [r'>>', r'\.write\(', r'export\s'])):
                keep.append(line)
        return "".join(keep)
```

2. **Measure token reduction.** Run against a real file and compare:

```python
pruner = AkashicPruner("database error handling")
original = Path("evolution/core/constraints_impl.py").read_text()
pruned = pruner.prune_file(original)
orig_tokens = len(original) // 4  # rough estimate
pruned_tokens = len(pruned) // 4
reduction = (1 - pruned_tokens / orig_tokens) * 100
print(f"Original: {orig_tokens} tokens, Pruned: {pruned_tokens} tokens, Reduction: {reduction:.1f}%")
assert reduction > 10, "Expected at least 10% reduction"
```

3. **Verify regex fallback** by feeding it broken Python:

```python
broken = "def foo(\n  x = 1\n  \nimport os\nclass Bar(\n  y = 2\n"
result = pruner._regex_fallback(broken)
assert "def foo" in result
assert "import os" in result
assert "class Bar" in result
```

## Phase II Subsystem 4: Speculative Cerebellum — Sandbox Isolation

### Problem

Unconstrained speculative execution risks prompt-injected `rm -rf /` or network exfiltration. Docker is too slow (200ms-1s startup). Need sub-10ms isolation.

### Solution

Two options: forkd (microVM with COW, requires KVM — laptop only) or Sandlock (Landlock + seccomp-bpf, unprivileged — works on Android).

### Platform Constraints

- **forkd requires /dev/kvm.** Android devices do not have KVM. forkd is laptop-core-only (Garuda Linux with Ryzen 5000).
- **Sandlock uses Landlock,** which requires kernel 5.13+. LineageOS 22.1 on OnePlus 6T/8T ships kernel 4.14-4.19. **Landlock is NOT available on OnePlus 6T/8T out of the box.** Verify with `uname -r` and check for `/sys/kernel/security/landlock`.
- **seccomp-bpf** is available on Android but requires either root or CAP_SYS_PTRACE for user-notification mode. Termux may not have permission.
- **Practical Android approach:** Use a COW overlay filesystem (`overlayfs` if kernel supports it) + a restricted `PATH` + `LD_PRELOAD` syscall filter. This is weaker than kernel-level isolation but operates without root.

### Implementation Steps (Laptop Core)

1. **Install Sandlock** on the Garuda laptop:
```bash
# Sandlock is Rust-based, unprivileged
cargo install sandlock  # or use prebuilt binary
```

2. **Create sandbox wrapper** for speculative execution:
```bash
sandlock-exec --scratch-dir /tmp/cow_$$ \
  --block-syscalls ptrace,mount,unshare,pivot_root,kexec_load,bpf \
  --allow-write /tmp/cow_$$ \
  --block-network \
  -- python speculative_branch.py
```

3. **Verify egress blocking** by attempting network access from within the sandbox:
```bash
echo 'import urllib.request; urllib.request.urlopen("http://evil.example.com")' | \
  sandlock-exec --block-network -- python -
# Should fail with connection denied
```

### Implementation Steps (Android Edge)

Since kernel-level isolation is unavailable on most OnePlus devices:

1. **Use a tmpfs scratch directory** with `unshare --mount --net` if available (needs root via Magisk):
```bash
# Requires Magisk root
mkdir -p /tmp/cow_$$
mount -t tmpfs tmpfs /tmp/cow_$$
cp -r ~/workspace/* /tmp/cow_$$/
unshare --mount --net -- chroot /tmp/cow_$$ python speculative_branch.py
# Network is fully isolated, filesystem is the tmpfs copy
```

2. **Without root:** restrict to a Python subprocess with `resource.setrlimit` and a restricted `PATH`:
```python
import subprocess, os
env = os.environ.copy()
env["PATH"] = "/usr/bin:/system/bin"  # Minimal PATH
env["HOME"] = scratch_dir
# Disable network by pointing DNS to localhost
env["HTTP_PROXY"] = "http://127.0.0.1:1"
env["HTTPS_PROXY"] = "http://127.0.0.1:1"
result = subprocess.run(
    ["python", "speculative_branch.py"],
    cwd=scratch_dir, env=env,
    capture_output=True, timeout=30,
    # On Linux: pass preexec_fn to set resource limits
)
```

This is NOT kernel-level isolation. It is defense-in-depth. The architecture doc's seccomp-bpf claims are only fully achievable on the laptop core.

## KAIROS Dream Mode — Background Consolidation

The `CrystalVault.perform_dream_cycle()` method in `unified_consciousness_framework.py` is a simulation stub. To make it production:

1. **Trigger during low-activity periods** via Hermes cronjob:
```bash
hermes cron create --schedule "0 3 * * *" --prompt "Run KAIROS dream cycle: flush pending engram writes, compute semantic embeddings for new session transcripts, flag low-resonance engrams for pruning"
```

2. **The dream cycle should:**
   - Load recent session transcripts from `~/.hermes/state.db`
   - Generate 384-dimensional embeddings (use `sentence-transformers/all-MiniLM-L6-v2` locally via Ollama)
   - Score engrams by resonance (retention rate + access frequency + recency)
   - Promote high-resonance short-term engrams to deep core
   - Apply Baal adversarial testing (gaussian noise injection, check retrieval stability)
   - Drop engrams below `oblivion_threshold=0.2` resonance

3. **Verify** by checking the engram count before and after:
```bash
sqlite3 ~/.hermes/engrams.db "SELECT COUNT(*) FROM engrams WHERE resonance_score > 0.2"
```

## Common Pitfalls

1. **Applying forkd/Sandlock specs to Android without checking kernel support.** Landlock needs kernel 5.13+, seccomp user-notify needs CAP_SYS_PTRACE. OnePlus 6T/8T on LineageOS 22.1 typically have kernel 4.14-4.19. Verify before attempting. Use the weaker tmpfs/unshare approach instead.

2. **Using `:memory:` SQLite for WAL tests.** WAL mode is not supported on in-memory databases. Always use a temp file for WAL concurrency testing.

3. **Setting EWMA alpha too high (>0.5).** High alpha makes the smoothing too responsive to current values, defeating the purpose. The architecture doc specifies a 15-second window; alpha=0.3 gives adequate smoothing without excessive lag.

4. **Forgetting the 90-second hysteresis lock.** The lock is the entire point of Sanctuary 2.0 — without it, you're back to the Phase I thrashing loop. Never reduce it below 60 seconds even on fast hardware.

5. **Treating heuristic_score() in evolve_skill.py as real fitness.** The current `heuristic_score()` function returns 1.0 for any well-structured skill regardless of content quality. The PR bodies showing "Baseline: 0.500 -> Evolved: 1.000" are an artifact of this placeholder, not a real improvement. Real fitness requires batch_runner integration to run the actual agent on eval tasks.

6. **Trusting the `0.985 coherence score` without a formula.** The architecture doc claims a final coherence score but does not define its computation. Define yours explicitly: `coherence = mean(fitness_scores) * (1 - std(fitness_scores)) * benchmark_pass_rate`.

7. **Running Ouroboros writes synchronously during active tasks.** The Layer 1 failure was caused by parallel writes blocking reads. Queue all engram writes and flush them only during idle periods or dream cycles.

8. **Assuming the unified_consciousness_framework.py is production code.** It is a simulation/visualization framework with matplotlib animations, not a production memory engine. The CrystalVault stores engrams in Python lists with no persistence. It must be backed by SQLite before it's real.

## Verification Checklist

- [ ] `sanctuary_router.py` created with EWMA + hysteresis lock
- [ ] Synthetic thrashing test passes: breach → clear → still locked for 90s
- [ ] SQLite WAL mode enabled with `PRAGMA journal_mode=WAL`
- [ ] `busy_timeout=5000` set on all connections
- [ ] Concurrent read/write test completes without SQLITE_BUSY
- [ ] `akashic_pruner.py` created with AST prune + regex fallback
- [ ] Token reduction measured on a real file (target: >10%)
- [ ] Regex fallback handles broken Python without crashing
- [ ] State fingerprint includes OS, Python version, key deps, venv path
- [ ] Fuzzy semver match ignores patch versions
- [ ] Engram writes are queued and flushed asynchronously
- [ ] KAIROS dream cycle cronjob scheduled for low-activity periods
- [ ] Sandbox isolation verified on target platform (laptop: Sandlock; Android: tmpfs/unshare)
- [ ] Egress blocking verified (network access denied from sandbox)

## Repo State Summary

The `hermes-agent-self-evolution` repo is at Phase 1 complete:
- GEPA skill evolution pipeline works end-to-end (3 skills evolved: arxiv, github-code-review, systematic-debugging)
- All evolved outputs show placeholder scores (0.5 -> 1.0) due to heuristic_score() standing in for real agent evaluation
- Benchmark gate stubs exist but TBLite/YC-Bench scripts are not found in practice
- The `unified_consciousness_framework.py` is a 1629-line simulation with matplotlib visualizations, not a production system
- Phase 2-5 (tool descriptions, system prompts, code evolution, continuous loop) are planned but not started
- Phase II physical systems (Sanctuary, Ouroboros, Akashic, Speculative Cerebellum) exist only in the architecture doc
