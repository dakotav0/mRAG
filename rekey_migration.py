"""
Re-key migration: regenerate all engram hashes using the current NgramHasher.

The old _text_to_key used word-ordinal-sums; the current code uses character
trigrams via NgramHasher.tokenize_text() + lookup_key(). Every stored memory
needs its hash regenerated so queries can find it again.

Run:  cd ~/Maker/mRAG && uv run python rekey_migration.py

Dry-run by default (--apply to write changes).
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

# Add project to path so we can import mrag
sys.path.insert(0, str(Path(__file__).parent))
from mrag.hash.ngram_hasher import NgramHasher

TABLES_DIR = Path(__file__).parent / "data" / "tables"


def rekey_db(db_path: Path, hasher: NgramHasher, apply: bool = False) -> dict:
    """Re-key one adapter database. Returns stats dict."""
    name = db_path.stem
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    rows = list(conn.execute("SELECT hash, text, salience, affect, source, age, tags FROM engrams"))
    if not rows:
        conn.close()
        return {"adapter": name, "entries": 0, "changed": 0, "collisions": 0, "errors": 0}

    stats = {"adapter": name, "entries": len(rows), "changed": 0, "collisions": 0, "errors": 0}
    new_map: dict[str, dict] = {}  # new_hash → row data
    changes: list[tuple[str, str]] = []  # (old_hash, new_hash)

    for row in rows:
        old_hash = row["hash"]
        text = row["text"]
        try:
            pseudo_ids = hasher.tokenize_text(text)
            new_hash = hasher.lookup_key(pseudo_ids) if pseudo_ids else "empty"
        except Exception as e:
            print(f"  [{name}] ERROR hashing key {old_hash}: {e}")
            stats["errors"] += 1
            continue

        if new_hash != old_hash:
            stats["changed"] += 1

        # Collision detection: same new hash, different text
        if new_hash in new_map:
            existing = new_map[new_hash]
            if existing["text"] != text:
                stats["collisions"] += 1
                # Keep the higher-salience entry
                if row["salience"] > existing["salience"]:
                    print(f"  [{name}] COLLISION: replacing '{existing['text'][:60]}...' "
                          f"(sal={existing['salience']}) with higher-salience entry "
                          f"(sal={row['salience']}) for hash {new_hash}")
                    new_map[new_hash] = {
                        "text": text, "salience": row["salience"],
                        "affect": row["affect"], "source": row["source"],
                        "age": row["age"], "tags": row["tags"],
                    }
                else:
                    print(f"  [{name}] COLLISION: dropping lower-salience entry "
                          f"'{text[:60]}...' (sal={row['salience']}) for hash {new_hash}")
                continue

        new_map[new_hash] = {
            "text": text, "salience": row["salience"],
            "affect": row["affect"], "source": row["source"],
            "age": row["age"], "tags": row["tags"],
        }
        changes.append((old_hash, new_hash))

    if apply and (stats["changed"] > 0 or stats["collisions"] > 0):
        # Rewrite the table with new keys
        conn.execute("DELETE FROM engrams")
        for new_hash, data in new_map.items():
            conn.execute(
                "INSERT INTO engrams (hash, text, salience, affect, source, age, tags) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (new_hash, data["text"], data["salience"], data["affect"],
                 data["source"], data["age"], json.dumps(data.get("tags", []))),
            )
        conn.commit()
        print(f"  [{name}] WROTE {len(new_map)} entries (was {len(rows)})")

    conn.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Re-key mRAG adapter tables")
    parser.add_argument("--apply", action="store_true",
                        help="Actually write changes (default: dry-run)")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Only re-key this adapter (default: all)")
    args = parser.parse_args()

    hasher = NgramHasher()
    db_files = sorted(TABLES_DIR.glob("*.db"))

    if args.adapter:
        db_files = [TABLES_DIR / f"{args.adapter}.db"]
        if not db_files[0].exists():
            print(f"Adapter '{args.adapter}' not found at {db_files[0]}")
            sys.exit(1)

    if not db_files:
        print(f"No .db files found in {TABLES_DIR}")
        return

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== Re-key migration ({mode}) ===\n")

    total = {"entries": 0, "changed": 0, "collisions": 0, "errors": 0}
    for db_path in db_files:
        stats = rekey_db(db_path, hasher, apply=args.apply)
        for k in total:
            total[k] += stats[k]

    print(f"\n---")
    print(f"Total: {total['entries']} entries across {len(db_files)} tables")
    print(f"  Changed hashes: {total['changed']}")
    print(f"  Collisions:     {total['collisions']}")
    print(f"  Errors:         {total['errors']}")
    if not args.apply:
        print(f"\n  Run with --apply to write changes.")


if __name__ == "__main__":
    main()
