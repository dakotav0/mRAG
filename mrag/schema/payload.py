from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EngRamPayload:
    """
    The structured object returned on an Engram hash hit.

    text:     The memory string injected into the prompt context.
    salience: [0.0, 1.0] — how important this memory is.
              High salience resists PIDX exponential decay.
              Threshold for KV cache eviction: salience < 0.15.
    affect:   [-1.0, 1.0] — emotional valence.
              Positive → warm/friendly adapter family.
              Negative → hostile/cautious adapter family.
              0.0 → neutral/professional adapter.
    source:   Which adapter table this memory belongs to.
    age:      Simulated time steps since memory was written.
              Used by decay engine; not injected into prompt.

    Decay note: PIDX uses continuous days-based λ=0.05/day for NpcMemory.
    mRAG uses discrete steps with rate=0.05/step. These are parallel clocks —
    PidxSyncPacket.decay_delta bridges them; do not unify the units here.
    """

    text:     str
    salience: float           # 0.0–1.0
    affect:   float           # -1.0 to +1.0
    source:   str             # e.g. "blacksmith", "merchant"
    age:      int   = 0       # simulated time steps
    tags:     list  = field(default_factory=list)

    def decayed_salience(self, decay_rate: float = 0.05) -> float:
        """PIDX-compatible exponential decay over discrete time steps."""
        return self.salience * (1.0 - decay_rate) ** self.age

    def is_evictable(self, threshold: float = 0.15) -> bool:
        """True when decayed salience falls below the KV cache eviction threshold."""
        return self.decayed_salience() < threshold


if __name__ == "__main__":
    # Smoke tests against mock_packets.json fixtures
    trading = EngRamPayload(text="Player bought a sword last session.",
                            salience=0.9, affect=0.8, source="blacksmith", age=2)
    assert not trading.is_evictable(), "trading_positive should not be evictable"
    assert abs(trading.decayed_salience() - 0.9 * 0.95 ** 2) < 1e-9

    decay_case = EngRamPayload(text="Player haggled poorly three weeks ago.",
                               salience=0.3, affect=-0.3, source="merchant", age=40)
    assert decay_case.is_evictable(), "decay_eviction fixture should be evictable"
    # 0.3 * 0.95^40 ≈ 0.040 < 0.15
    assert decay_case.decayed_salience() < 0.15

    print("payload.py OK")
