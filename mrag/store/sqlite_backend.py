"""
SQLite persistence layer for EngRamTable.

One database file per adapter under `tables_dir/`.
Schema is intentionally flat — one row per engram entry, tags stored as
a JSON array string. No ORM; plain sqlite3 for zero extra dependencies.

The hot path (hash lookup) never touches SQLite — that's all in-memory
via EngRamTable. SQLite is only hit on mount (load) and unmount (save).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from mrag.schema.payload import EngRamPayload
from mrag.store.engram_table import EngRamTable
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from mrag.router.decay import DecayPolicy


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS engrams (
    hash     TEXT PRIMARY KEY,
    text     TEXT    NOT NULL,
    salience REAL    NOT NULL,
    affect   REAL    NOT NULL,
    source   TEXT    NOT NULL,
    age      INTEGER NOT NULL DEFAULT 0,
    tags     TEXT    NOT NULL DEFAULT '[]'
)
"""

_UPSERT = """
INSERT INTO engrams (hash, text, salience, affect, source, age, tags)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(hash) DO UPDATE SET
    text     = excluded.text,
    salience = excluded.salience,
    affect   = excluded.affect,
    source   = excluded.source,
    age      = excluded.age,
    tags     = excluded.tags
"""


class SqliteBackend:
    """
    Loads and saves EngRamTable ↔ SQLite.

    Parameters
    ----------
    tables_dir : str | Path
        Directory where per-adapter .db files live.
        Pass ":memory:" to use an in-memory database (tests only).
    """

    def __init__(self, tables_dir: str | Path) -> None:
        self._in_memory = str(tables_dir) == ":memory:"
        if not self._in_memory:
            self.tables_dir = Path(tables_dir)
            self.tables_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.tables_dir = None
        # Shared in-memory connection pool (one per adapter name, tests only)
        self._mem_conns: dict[str, sqlite3.Connection] = {}

    # ── Internal ─────────────────────────────────────────────────────────────

    def _db_path(self, adapter_name: str) -> str:
        if self._in_memory:
            return ":memory:"
        return str(self.tables_dir / f"{adapter_name}.db")

    def _connect(self, adapter_name: str) -> sqlite3.Connection:
        if self._in_memory:
            # Reuse the same connection so data persists across load/save calls.
            # check_same_thread=False: the prefetch coordinator mounts tables from
            # a background thread; mRAG is single-writer by design so this is safe.
            if adapter_name not in self._mem_conns:
                conn = sqlite3.connect(":memory:", check_same_thread=False)
                conn.row_factory = sqlite3.Row
                self._mem_conns[adapter_name] = conn
            return self._mem_conns[adapter_name]
        conn = sqlite3.connect(self._db_path(adapter_name))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(_CREATE_TABLE)
        conn.commit()

    # ── Public API ───────────────────────────────────────────────────────────

    def load(self, adapter_name: str, decay_policy: Optional[DecayPolicy] = None) -> EngRamTable:
        """
        Read all rows from the adapter's SQLite db into a fresh EngRamTable.
        Returns an empty table if the db doesn't exist yet.
        """
        table = EngRamTable(adapter_name, decay_policy=decay_policy)
        conn = self._connect(adapter_name)
        self._ensure_schema(conn)

        for row in conn.execute("SELECT * FROM engrams"):
            payload = EngRamPayload(
                text=row["text"],
                salience=row["salience"],
                affect=row["affect"],
                source=row["source"],
                age=row["age"],
                tags=json.loads(row["tags"]),
            )
            table.put(row["hash"], payload)

        if not self._in_memory:
            conn.close()
        return table

    def save(self, adapter_name: str, table: EngRamTable) -> None:
        """
        Persist the entire EngRamTable back to SQLite.
        Replaces all rows (full upsert — no partial writes).
        """
        conn = self._connect(adapter_name)
        self._ensure_schema(conn)

        rows = [
            (key, p.text, p.salience, p.affect, p.source, p.age, json.dumps(p.tags))
            for key, p in table.items()
        ]
        conn.executemany(_UPSERT, rows)
        conn.commit()

        if not self._in_memory:
            conn.close()

    def count(self, adapter_name: str) -> int:
        """Return the number of engram rows in an adapter's database."""
        conn = self._connect(adapter_name)
        self._ensure_schema(conn)
        row = conn.execute("SELECT COUNT(*) FROM engrams").fetchone()
        count = row[0] if row else 0
        if not self._in_memory:
            conn.close()
        return count

    def list_adapters(self) -> list[str]:
        """Return all adapter names that have .db files on disk."""
        if self._in_memory:
            return list(self._mem_conns.keys())
        names = []
        for path in sorted(self.tables_dir.glob("*.db")):
            names.append(path.stem)
        return names

    def delete(self, adapter_name: str) -> None:
        """Drop the adapter's database file (used when an NPC is fully retired)."""
        if self._in_memory:
            self._mem_conns.pop(adapter_name, None)
        else:
            path = Path(self._db_path(adapter_name))
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    from mrag.schema.payload import EngRamPayload
    from mrag.store.engram_table import EngRamTable

    backend = SqliteBackend(":memory:")

    # Build a table and save it
    t = EngRamTable("merchant")
    t.put("key1", EngRamPayload(text="Player haggled.", salience=0.7,
                                 affect=-0.3, source="merchant", age=5,
                                 tags=["trade"]))
    t.put("key2", EngRamPayload(text="Player bought potions.", salience=0.5,
                                 affect=0.2, source="merchant", age=2))
    backend.save("merchant", t)

    # Load it back
    t2 = backend.load("merchant")
    assert len(t2) == 2

    p = t2.get("key1")
    assert p is not None
    assert p.text == "Player haggled."
    assert p.salience == 0.7
    assert p.tags == ["trade"]
    assert p.age == 5

    # Upsert: update key1 salience, add key3
    t2.put("key1", EngRamPayload(text="Player haggled.", salience=0.4,
                                  affect=-0.3, source="merchant", age=10))
    t2.put("key3", EngRamPayload(text="New memory.", salience=0.9,
                                  affect=0.5, source="merchant", age=0))
    backend.save("merchant", t2)

    t3 = backend.load("merchant")
    assert len(t3) == 3
    assert t3.get("key1").salience == 0.4
    assert t3.get("key3").text == "New memory."

    # delete clears the in-memory db
    backend.delete("merchant")
    t4 = backend.load("merchant")
    assert len(t4) == 0

    print("sqlite_backend.py OK")
