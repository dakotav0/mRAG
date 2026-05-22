"""
EngRamTable — per-adapter in-memory engram store.

Maps ngram_hash (hex str) → EngRamPayload. No SQLite knowledge here;
the backend handles persistence. Keep this a pure data structure so it
can be re-implemented as a PyO3 extension later without changing call sites.
"""

from __future__ import annotations

import heapq
from typing import Iterator, Optional

from mrag.schema.payload import EngRamPayload
from mrag.router.decay import DecayPolicy, ExponentialDecay


class EngRamTable:
    """
    In-memory engram store for one adapter.

    Keys are hex digests produced by NgramHasher.lookup_key().
    Values are EngRamPayload instances (or generic Engram instances).

    Designed to be serialized/deserialized by sqlite_backend and
    mounted/unmounted by EngRamManager. No I/O here.
    """

    def __init__(self, adapter_name: str, decay_policy: Optional[DecayPolicy] = None) -> None:
        self.adapter_name = adapter_name
        self._store: dict[str, EngRamPayload] = {}
        self._needs_eviction = True
        # Expose customizable decay policy, fallback to default 0.05 ExponentialDecay
        self.decay_policy = decay_policy if decay_policy is not None else ExponentialDecay(0.05)

    # ── Core access ──────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[EngRamPayload]:
        """Return payload for key, or None on miss."""
        return self._store.get(key)

    def put(self, key: str, payload: EngRamPayload) -> None:
        """Upsert a payload. Overwrites existing entry for the same key."""
        self._store[key] = payload
        self._needs_eviction = True

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    # ── Iteration ────────────────────────────────────────────────────────────

    def items(self) -> Iterator[tuple[str, EngRamPayload]]:
        return iter(self._store.items())

    # ── Decay / eviction ─────────────────────────────────────────────────────

    def tick(self, steps: int = 1) -> None:
        """Age all entries by `steps` simulated time steps."""
        if steps > 0:
            for payload in self._store.values():
                payload.age += steps
            self._needs_eviction = True

    def evict_stale(self, threshold: float = 0.15) -> int:
        """
        Remove entries whose decayed salience is below threshold using the decay policy.
        Returns count of evicted entries.
        """
        if not self._needs_eviction:
            return 0
        to_drop = [
            k for k, v in self._store.items()
            if self.decay_policy.is_evictable(v.salience, v.age, threshold)
        ]
        for k in to_drop:
            del self._store[k]
        self._needs_eviction = False
        return len(to_drop)

    def top_by_salience(self, n: int = 5) -> list[EngRamPayload]:
        """Return up to n payloads sorted by decayed_salience descending using the decay policy."""
        return heapq.nlargest(
            n,
            self._store.values(),
            key=lambda p: self.decay_policy.decayed_salience(p.salience, p.age),
        )


if __name__ == "__main__":
    from mrag.schema.payload import EngRamPayload
    from mrag.router.decay import LinearDecay

    t = EngRamTable("blacksmith")

    p1 = EngRamPayload(text="Player bought a sword.", salience=0.9, affect=0.8,
                       source="blacksmith", age=0)
    p2 = EngRamPayload(text="Old memory.", salience=0.3, affect=-0.1,
                       source="blacksmith", age=0)

    t.put("aaa", p1)
    t.put("bbb", p2)
    assert len(t) == 2
    assert t.get("aaa") is p1
    assert t.get("zzz") is None

    # Tick 20 steps:
    #   p1: 0.9 * 0.95^20 ≈ 0.322 — survives
    #   p2: 0.3 * 0.95^20 ≈ 0.107 — evictable
    t.tick(20)
    assert p2.age == 20
    evicted = t.evict_stale()
    assert evicted == 1
    assert len(t) == 1
    assert "bbb" not in t

    # top_by_salience returns remaining entry
    top = t.top_by_salience()
    assert top[0] is p1

    # Linear decay table test
    t_linear = EngRamTable("merchant", decay_policy=LinearDecay(0.01))
    p3 = EngRamPayload("linear memory", salience=0.5, age=0)
    t_linear.put("linear_key", p3)
    t_linear.tick(40)  # age -> 40
    # decayed_salience = max(0, 0.5 - 40 * 0.01) = 0.1
    # is_evictable(0.15 threshold) -> True
    assert t_linear.evict_stale() == 1
    assert len(t_linear) == 0

    print("engram_table.py OK")
