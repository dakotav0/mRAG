"""
Decay utilities for mRAG — PIDX-compatible salience decay.

PIDX clock:  continuous days-based exponential, λ=0.0500/day for NpcMemory.
mRAG clock:  discrete step-based exponential, rate=0.05/step.

These are parallel clocks, not the same unit. PidxSyncPacket.decay_delta
carries the PIDX-native float delta (days); convert_pidx_delta() translates
it into mRAG step counts for EngRamManager.tick_adapter(). Do not unify
the units — keep them decoupled so PIDX can change its tick granularity
without touching mRAG's table structure.

The step-to-day equivalence is a soft calibration:
    1 mRAG step ≈ 1 PIDX day at λ=0.05 / rate=0.05

This holds at small delta values. At large deltas the discrete approximation
diverges from the continuous formula — that's acceptable; mRAG's decay is
intentionally coarser than PIDX's.
"""

from __future__ import annotations

import math

# ── mRAG decay constants ──────────────────────────────────────────────────────
DEFAULT_DECAY_RATE: float = 0.05   # per step; matches EngRamPayload default
EVICTION_THRESHOLD: float = 0.15   # below this, memory is KV-cache evictable

# ── PIDX NpcMemory λ (days⁻¹) — used only for delta conversion ───────────────
_PIDX_NPC_MEMORY_LAMBDA: float = 0.05


def decayed_salience(salience: float, age: int,
                     rate: float = DEFAULT_DECAY_RATE) -> float:
    """
    Discrete exponential decay: salience × (1 − rate)^age.

    Mirrors EngRamPayload.decayed_salience() as a standalone function
    so the router and bridge can call it without an EngRamPayload instance.
    """
    return salience * (1.0 - rate) ** age


def is_evictable(salience: float, age: int,
                 rate: float = DEFAULT_DECAY_RATE,
                 threshold: float = EVICTION_THRESHOLD) -> bool:
    """Return True when decayed salience falls below the eviction threshold."""
    return decayed_salience(salience, age, rate) < threshold


def steps_until_eviction(salience: float,
                          rate: float = DEFAULT_DECAY_RATE,
                          threshold: float = EVICTION_THRESHOLD) -> int:
    """
    Return how many decay steps until a memory crosses the eviction threshold.

    Uses the closed-form solution:
        n = ceil(log(threshold / salience) / log(1 - rate))

    Returns 0 if already evictable, math.inf (as sys.maxsize) if salience
    is 0 or the threshold is unreachable.
    """
    if salience <= 0.0:
        return 0
    if salience < threshold:
        return 0
    # log(threshold / salience) is negative; log(1 - rate) is negative → positive ratio
    return math.ceil(math.log(threshold / salience) / math.log(1.0 - rate))


def convert_pidx_delta(decay_delta: float) -> int:
    """
    Convert a PIDX decay_delta (days, continuous) to mRAG step count (discrete).

    Formula: steps = round(decay_delta × λ_pidx / rate_mrag)
    At λ=0.05 and rate=0.05 this simplifies to steps = round(decay_delta).

    Clamps to [0, ∞). A negative delta (salience_boost direction) returns 0 —
    boosting is handled separately via EngRamManager.write_memory upsert.
    """
    if decay_delta <= 0.0:
        return 0
    ratio = _PIDX_NPC_MEMORY_LAMBDA / DEFAULT_DECAY_RATE
    return max(0, round(decay_delta * ratio))


if __name__ == "__main__":
    # Decay math
    assert abs(decayed_salience(0.9, 2) - 0.9 * 0.95**2) < 1e-12
    assert abs(decayed_salience(0.3, 40) - 0.3 * 0.95**40) < 1e-12

    # Eviction checks matching mock_packets.json fixtures
    assert not is_evictable(0.9, 2)               # trading_positive: survives
    assert not is_evictable(0.95, 1)              # combat_negative: survives
    assert is_evictable(0.3, 40)                  # decay_eviction: evicted

    # steps_until_eviction
    n = steps_until_eviction(0.9)
    # After n steps: 0.9 * 0.95^n < 0.15 → n ≈ 36
    assert is_evictable(0.9, n)
    assert not is_evictable(0.9, n - 1)

    n_low = steps_until_eviction(0.3)
    assert is_evictable(0.3, n_low)
    assert not is_evictable(0.3, n_low - 1)

    assert steps_until_eviction(0.0) == 0         # already gone
    assert steps_until_eviction(0.1) == 0         # below threshold at age 0

    # PIDX delta conversion: at equal λ and rate, 1 day ≈ 1 step
    assert convert_pidx_delta(2.0) == 2
    assert convert_pidx_delta(0.0) == 0
    assert convert_pidx_delta(-5.0) == 0          # boost direction → no aging

    print("decay.py OK")
