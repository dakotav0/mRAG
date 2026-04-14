"""
EngRamTable — per-adapter in-memory hash table.

Maps ngram_hash (hex str) → EngRamPayload. No SQLite knowledge here;
the backend handles persistence. Keep this a pure data structure so it
can be re-implemented as a PyO3 extension later without changing call sites.
"""

from __future__ import annotations

from typing import Iterator, Optional

from mrag.schema.payload import EngRamPayload


class EngRamTable:
    """
    In-memory engram store for one adapter.

    Keys are hex digests produced by NgramHasher.lookup_key().
    Values are EngRamPayload instances.

    Designed to be serialized/deserialized by sqlite_backend and
    mounted/unmounted by EngRamManager. No I/O here.
    """

    def __init__(self, adapter_name: str) -> None:
        self.adapter_name = adapter_name
        self._store: dict[str, EngRamPayload] = {}

    # ── Core access ──────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[EngRamPayload]:
        """Return payload for key, or None on miss."""
        return self._store.get(key)

    def put(self, key: str, payload: EngRamPayload) -> None:
        """Upsert a payload. Overwrites existing entry for the same key."""
        self._store[key] = payload

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
        for payload in self._store.values():
            payload.age += steps

    def evict_stale(self, threshold: float = 0.15) -> int:
        """
        Remove entries whose decayed salience is below threshold.
        Returns count of evicted entries.
        """
        to_drop = [k for k, v in self._store.items() if v.is_evictable(threshold)]
        for k in to_drop:
            del self._store[k]
        return len(to_drop)

    def top_by_salience(self, n: int = 5) -> list[EngRamPayload]:
        """Return up to n payloads sorted by decayed_salience descending."""
        ranked = sorted(
            self._store.values(),
            key=lambda p: p.decayed_salience(),
            reverse=True,
        )
        return ranked[:n]


if __name__ == "__main__":
    from mrag.schema.payload import EngRamPayload

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
    # (40 ticks would drop p1 below threshold too: 0.9 * 0.95^40 ≈ 0.116 < 0.15)
    t.tick(20)
    assert p2.age == 20
    evicted = t.evict_stale()
    assert evicted == 1
    assert len(t) == 1
    assert "bbb" not in t

    # top_by_salience returns remaining entry
    top = t.top_by_salience()
    assert top[0] is p1

    print("engram_table.py OK")
