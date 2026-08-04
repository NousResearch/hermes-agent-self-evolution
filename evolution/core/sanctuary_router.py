#!/usr/bin/env python3
"""
Sanctuary 2.0 — Grounded Telemetry Routing Engine (EWMA + hysteresis lock)

Routes computation between Local Cerebellum and Cloud Cortex based on
exponentially smoothed memory telemetry, with a hard 90-second hysteresis
lock after any breach-triggered unload. Kills the Phase I VRAM thrashing loop.

Android adaptation: no discrete VRAM — models run in unified memory. Telemetry
source is /proc/meminfo MemAvailable (MB), thresholds are percentage-based
(BREACH < 10% of total RAM, MARGINAL < 25%).

Spec: Metaconscious Singularity Node — Sanctuary 2.0 subsystem.

States:
  CLEAR    → target local cerebellum
  MARGINAL → hybrid (local pre-checks, cloud escalation ready)
  BREACH   → full cloud cortex escalation, 90s unload lock
  BYPASS   → force local regardless of telemetry (hardware-bound tasks)
"""
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional, Tuple


class SanctuaryState(Enum):
    CLEAR = "clear"
    MARGINAL = "marginal"
    BREACH = "breach"
    BYPASS = "bypass"


def read_meminfo_mb() -> Tuple[float, float]:
    """Return (mem_available_mb, mem_total_mb) from /proc/meminfo."""
    total = avail = 0.0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = float(line.split()[1]) / 1024.0  # kB -> MB
                elif line.startswith("MemAvailable:"):
                    avail = float(line.split()[1]) / 1024.0
    except FileNotFoundError:
        return 4096.0, 8192.0  # fallback for non-Linux
    return avail, total


@dataclass
class SanctuaryRouter:
    alpha: float = 0.3               # EWMA smoothing factor
    window_seconds: float = 15.0     # rolling observation window
    breach_frac: float = 0.10        # BREACH when available < 10% of total
    marginal_frac: float = 0.25      # MARGINAL when available < 25% of total
    hysteresis_lock_seconds: float = 90.0
    telemetry_fn: object = read_meminfo_mb
    _samples: Deque = field(default_factory=lambda: deque(maxlen=300))
    _ewma: Optional[float] = None
    _last_unload_time: float = 0.0
    _last_breach_time: float = 0.0
    _current_state: SanctuaryState = SanctuaryState.CLEAR
    _bypass: bool = False

    # ─── Telemetry ───

    def _fetch_avail_gb(self) -> float:
        avail_mb, total_mb = self.telemetry_fn()
        self._total_mb = total_mb
        return avail_mb / 1024.0

    # ─── Core routing decision ───

    def update(self, avail_gb: Optional[float] = None) -> SanctuaryState:
        """Feed a telemetry sample, return the routing state."""
        if self._bypass:
            self._current_state = SanctuaryState.BYPASS
            return self._current_state

        now = time.time()
        if avail_gb is None:
            avail_gb = self._fetch_avail_gb()
        self._samples.append((now, avail_gb))

        # Prune samples outside the rolling window
        cutoff = now - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        if not self._samples:
            return self._current_state

        # EWMA over the window
        if self._ewma is None:
            self._ewma = self._samples[0][1]
        for _, v in list(self._samples)[1:]:
            self._ewma = self.alpha * v + (1 - self.alpha) * self._ewma

        total_gb = getattr(self, "_total_mb", 8192.0) / 1024.0
        breach_gb = total_gb * self.breach_frac
        marginal_gb = total_gb * self.marginal_frac

        # Hysteresis lock: after a breach unload, stay breached until lock expires
        in_lock = (now - self._last_unload_time) < self.hysteresis_lock_seconds
        if in_lock and self._current_state in (SanctuaryState.BREACH,
                                               SanctuaryState.MARGINAL):
            return self._current_state  # Frozen — no reload, no flapping

        # State transition
        if self._ewma < breach_gb:
            new_state = SanctuaryState.BREACH
            if self._current_state != SanctuaryState.BREACH:
                self._last_unload_time = now  # Start lock on entering breach
                self._last_breach_time = now
        elif self._ewma < marginal_gb:
            new_state = SanctuaryState.MARGINAL
        else:
            new_state = SanctuaryState.CLEAR

        self._current_state = new_state
        return new_state

    # ─── Overrides ───

    def force_local_bypass(self, enabled: bool = True) -> None:
        """Hardware-bound task: force local compute regardless of telemetry."""
        self._bypass = enabled
        if enabled:
            self._current_state = SanctuaryState.BYPASS

    def release_lock(self) -> None:
        """Manual lock release (for testing / ops)."""
        self._last_unload_time = 0.0

    # ─── Introspection ───

    def snapshot(self) -> dict:
        return {
            "state": self._current_state.value,
            "ewma_gb": round(self._ewma or 0.0, 2),
            "samples": len(self._samples),
            "lock_remaining": max(0.0, self.hysteresis_lock_seconds -
                                  (time.time() - self._last_unload_time)),
            "total_gb": round(getattr(self, "_total_mb", 0.0) / 1024.0, 1),
            "alpha": self.alpha,
            "window_seconds": self.window_seconds,
        }

    # ─── Decider helper for agents ───

    def decide(self) -> str:
        """Human/agent-readable routing verdict."""
        s = self._current_state
        if s == SanctuaryState.BYPASS:
            return "local-bypass: force local (hardware-bound task)"
        if s == SanctuaryState.BREACH:
            rem = self.snapshot()["lock_remaining"]
            return (f"cloud-escalate: memory breached, unload lock {rem:.0f}s "
                    f"remaining — local weights must stay unloaded")
        if s == SanctuaryState.MARGINAL:
            return "hybrid: local pre-checks + cloud-ready"
        return "local: target cerebellum (memory clear)"


if __name__ == "__main__":
    # ─── Test 1: synthetic thrashing → breach → lock holds → no reload ───
    router = SanctuaryRouter()
    router._total_mb = 8192.0  # 8GB device

    # Simulate 60s of low memory (0.5GB free < 10% of 8GB = 0.8GB)
    for _ in range(60):
        state = router.update(0.5)
        time.sleep(0.001)
    assert router._current_state == SanctuaryState.BREACH, \
        f"Expected BREACH, got {router._current_state}"
    breach_lock_time = router._last_unload_time

    # Memory frees up, but lock must prevent reload (the Phase I thrash bug)
    for _ in range(30):
        state = router.update(8.0)
        time.sleep(0.001)
    elapsed = time.time() - breach_lock_time
    assert state == SanctuaryState.BREACH, \
        f"Lock failed — reloaded during hysteresis! state={state}"
    assert elapsed < router.hysteresis_lock_seconds
    print(f"[sanctuary] PASS — breach lock holds ({elapsed:.1f}s < 90s), no thrashing")

    # ─── Test 2: lock expiry → recovers to CLEAR ───
    router._last_unload_time = time.time() - 95.0  # fast-forward past lock
    state = router.update(8.0)
    assert state == SanctuaryState.CLEAR, f"Expected CLEAR after lock expiry, got {state}"
    print("[sanctuary] PASS — recovers to CLEAR after lock expires")

    # ─── Test 3: EWMA ignores transient spikes (no instant BREACH) ───
    router2 = SanctuaryRouter()
    router2._total_mb = 8192.0
    router2.update(7.9)  # warm up
    router2.update(0.3)  # single transient spike
    router2.update(7.8)
    assert router2._current_state != SanctuaryState.BREACH, \
        "Transient spike caused instant breach — EWMA window not smoothing!"
    print("[sanctuary] PASS — EWMA damps transient spikes")

    # ─── Test 4: BYPASS overrides everything ───
    router2.force_local_bypass(True)
    router2.update(0.1)
    assert router2._current_state == SanctuaryState.BYPASS
    router2.force_local_bypass(False)
    print("[sanctuary] PASS — LOCAL_BYPASS override")

    # ─── Test 5: live telemetry read on this device ───
    avail_mb, total_mb = read_meminfo_mb()
    print(f"[sanctuary] live telemetry: {avail_mb:.0f}MB free / {total_mb:.0f}MB total")
    live = SanctuaryRouter()
    s = live.update()
    print(f"[sanctuary] live state: {s.value} — {live.decide()}")
    print(f"[sanctuary] snapshot: {live.snapshot()}")
