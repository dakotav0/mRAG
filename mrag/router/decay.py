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

from abc import ABC, abstractmethod
import math

# ── mRAG decay constants ──────────────────────────────────────────────────────
DEFAULT_DECAY_RATE: float = 0.05   # per step; matches EngRamPayload default
EVICTION_THRESHOLD: float = 0.15   # below this, memory is KV-cache evictable

# ── PIDX NpcMemory λ (days⁻¹) — used only for delta conversion ───────────────
_PIDX_NPC_MEMORY_LAMBDA: float = 0.05
_DECAY_LOOKUP = [0.95 ** i for i in range(1000)]


class DecayPolicy(ABC):
    """
    Abstract base class for all engram decay policies.
    """

    @abstractmethod
    def decayed_salience(self, salience: float, age: int) -> float:
        """Calculate the salience level of a memory after a given age."""
        pass

    @abstractmethod
    def is_evictable(self, salience: float, age: int, threshold: float = EVICTION_THRESHOLD) -> bool:
        """Return True when the salience of a memory drops below the eviction threshold."""
        pass

    @abstractmethod
    def steps_until_eviction(self, salience: float, threshold: float = EVICTION_THRESHOLD) -> int | float:
        """Return the number of simulated steps until a memory is evictable."""
        pass


class ExponentialDecay(DecayPolicy):
    """
    Discrete exponential decay: salience × (1 − rate)^age.
    """

    def __init__(self, rate: float = DEFAULT_DECAY_RATE) -> None:
        self.rate = rate

    def decayed_salience(self, salience: float, age: int) -> float:
        if self.rate == 0.05:
            if age < 1000:
                return salience * _DECAY_LOOKUP[age]
            return 0.0
        return salience * (1.0 - self.rate) ** age

    def is_evictable(self, salience: float, age: int, threshold: float = EVICTION_THRESHOLD) -> bool:
        return self.decayed_salience(salience, age) < threshold

    def steps_until_eviction(self, salience: float, threshold: float = EVICTION_THRESHOLD) -> int:
        if salience <= 0.0 or salience < threshold:
            return 0
        return math.ceil(math.log(threshold / salience) / math.log(1.0 - self.rate))


class LinearDecay(DecayPolicy):
    """
    Linear decay: max(0.0, salience - age × rate).
    """

    def __init__(self, rate: float) -> None:
        self.rate = rate

    def decayed_salience(self, salience: float, age: int) -> float:
        return max(0.0, salience - (age * self.rate))

    def is_evictable(self, salience: float, age: int, threshold: float = EVICTION_THRESHOLD) -> bool:
        return self.decayed_salience(salience, age) < threshold

    def steps_until_eviction(self, salience: float, threshold: float = EVICTION_THRESHOLD) -> int:
        if salience <= 0.0 or salience < threshold:
            return 0
        return math.ceil((salience - threshold) / self.rate)


class NoDecay(DecayPolicy):
    """
    No decay policy: salience remains constant over time steps.
    """

    def decayed_salience(self, salience: float, age: int) -> float:
        return salience

    def is_evictable(self, salience: float, age: int, threshold: float = EVICTION_THRESHOLD) -> bool:
        return salience < threshold

    def steps_until_eviction(self, salience: float, threshold: float = EVICTION_THRESHOLD) -> int | float:
        if salience < threshold:
            return 0
        return math.inf


# ── Backwards-Compatible Functional Shims ────────────────────────────────────

_DEFAULT_POLICY = ExponentialDecay(DEFAULT_DECAY_RATE)


def decayed_salience(salience: float, age: int,
                     rate: float = DEFAULT_DECAY_RATE) -> float:
    """
    Discrete exponential decay: salience × (1 − rate)^age.
    """
    if rate == DEFAULT_DECAY_RATE:
        return _DEFAULT_POLICY.decayed_salience(salience, age)
    return ExponentialDecay(rate).decayed_salience(salience, age)


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
    """
    if rate == DEFAULT_DECAY_RATE:
        return _DEFAULT_POLICY.steps_until_eviction(salience, threshold)
    return ExponentialDecay(rate).steps_until_eviction(salience, threshold)


def convert_pidx_delta(decay_delta: float) -> int:
    """
    Convert a PIDX decay_delta (days, continuous) to mRAG step count (discrete).
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
    assert not is_evictable(0.9, 2)               # survives
    assert not is_evictable(0.95, 1)              # survives
    assert is_evictable(0.3, 40)                  # evicted

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

    # LinearDecay specific verification
    linear = LinearDecay(0.01)
    assert linear.decayed_salience(0.5, 10) == 0.4
    assert linear.is_evictable(0.5, 40, threshold=0.15)  # 0.5 - 0.4 = 0.1 < 0.15
    assert not linear.is_evictable(0.5, 30, threshold=0.15)  # 0.5 - 0.3 = 0.2 >= 0.15
    assert linear.steps_until_eviction(0.5, threshold=0.15) == 35  # (0.5 - 0.15)/0.01 = 35

    print("decay.py OK")
