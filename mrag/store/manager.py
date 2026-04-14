"""
EngRamManager — mount/unmount lifecycle for adapter tables.

Keeps at most `max_loaded` EngRamTable instances in system RAM at once.
Evicts the Least Recently Used table on overflow, serializing it back to
SQLite before freeing RAM.

Thread safety: single-threaded by design (one inference pipeline).
If you ever need concurrent access, add a threading.Lock around _tables.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

from mrag.schema.payload import EngRamPayload
from mrag.store.engram_table import EngRamTable
from mrag.store.sqlite_backend import SqliteBackend


class EngRamManager:
    """
    Keeps at most `max_loaded` adapter tables in system RAM at once.
    Evicts LRU table on overflow. Tables backed by SQLite on NVMe (or
    ":memory:" for tests).

    Parameters
    ----------
    tables_dir : str
        Directory for SQLite .db files, or ":memory:" for tests.
    max_loaded : int
        Maximum number of adapter tables held in RAM simultaneously.
        Default 3 matches the MIIN archetype count (Warrior, Mystic, etc.).
    """

    def __init__(self, tables_dir: str, max_loaded: int = 3) -> None:
        self._backend = SqliteBackend(tables_dir)
        self._max_loaded = max_loaded
        # OrderedDict: insertion/access order = LRU order (oldest first)
        self._tables: OrderedDict[str, EngRamTable] = OrderedDict()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _evict_lru(self) -> None:
        """Serialize and remove the least recently used table."""
        adapter_name, table = self._tables.popitem(last=False)
        self._backend.save(adapter_name, table)

    def _touch(self, adapter_name: str) -> None:
        """Mark adapter as most-recently-used."""
        self._tables.move_to_end(adapter_name)

    # ── Public API ───────────────────────────────────────────────────────────

    def mount(self, adapter_name: str) -> EngRamTable:
        """
        Return the in-RAM table for adapter_name.
        Loads from SQLite if not already mounted; evicts LRU if at capacity.
        """
        if adapter_name in self._tables:
            self._touch(adapter_name)
            return self._tables[adapter_name]

        if len(self._tables) >= self._max_loaded:
            self._evict_lru()

        table = self._backend.load(adapter_name)
        self._tables[adapter_name] = table
        return table

    def unmount(self, adapter_name: str) -> None:
        """
        Serialize adapter table back to SQLite and remove from RAM.
        No-op if the adapter is not currently mounted.
        """
        if adapter_name not in self._tables:
            return
        table = self._tables.pop(adapter_name)
        self._backend.save(adapter_name, table)

    def unmount_all(self) -> None:
        """Flush all mounted tables to SQLite. Call on clean shutdown."""
        for name in list(self._tables):
            self.unmount(name)

    def write_memory(self, adapter_name: str, key: str,
                     payload: EngRamPayload) -> None:
        """Upsert a memory into the named table (auto-mounts if needed)."""
        table = self.mount(adapter_name)
        table.put(key, payload)

    def get_memory(self, adapter_name: str,
                   key: str) -> Optional[EngRamPayload]:
        """
        Look up a memory by ngram hash key.
        Returns None on miss (table not mounted OR key not present).
        Auto-mounts the table.
        """
        table = self.mount(adapter_name)
        return table.get(key)

    def tick_adapter(self, adapter_name: str, steps: int = 1) -> None:
        """Age all entries in an adapter table by `steps` time steps."""
        table = self.mount(adapter_name)
        table.tick(steps)

    def evict_stale(self, adapter_name: str,
                    threshold: float = 0.15) -> int:
        """
        Remove sub-threshold engrams from an adapter table.
        Returns count of evicted entries.
        """
        table = self.mount(adapter_name)
        return table.evict_stale(threshold)

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def loaded_adapters(self) -> list[str]:
        """Names of currently mounted adapters, LRU-first."""
        return list(self._tables)

    def __len__(self) -> int:
        return len(self._tables)


if __name__ == "__main__":
    from mrag.schema.payload import EngRamPayload

    mgr = EngRamManager(":memory:", max_loaded=2)

    # Write memories into two adapters
    mgr.write_memory("blacksmith", "k1",
                     EngRamPayload("Sword sale.", 0.9, 0.8, "blacksmith"))
    mgr.write_memory("merchant", "k2",
                     EngRamPayload("Potion haggle.", 0.7, -0.2, "merchant"))

    assert set(mgr.loaded_adapters) == {"blacksmith", "merchant"}
    assert len(mgr) == 2

    # Third adapter triggers LRU eviction of blacksmith (oldest access)
    mgr.write_memory("guard", "k3",
                     EngRamPayload("Player trespassed.", 0.95, -0.9, "guard"))

    assert len(mgr) == 2
    assert "blacksmith" not in mgr.loaded_adapters  # evicted to SQLite

    # Remounting blacksmith loads it back from SQLite
    p = mgr.get_memory("blacksmith", "k1")
    assert p is not None and p.text == "Sword sale."

    # tick + evict round-trip
    mgr.write_memory("merchant", "old",
                     EngRamPayload("Ancient deal.", 0.3, 0.0, "merchant", age=39))
    mgr.tick_adapter("merchant", steps=1)  # age 39 → 40
    evicted = mgr.evict_stale("merchant")
    assert evicted == 1  # 0.3 * 0.95^40 ≈ 0.040 < 0.15

    # Clean shutdown
    mgr.unmount_all()
    assert len(mgr) == 0

    print("manager.py OK")
