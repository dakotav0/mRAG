"""
mrag_mcp.py — MCP server wrapping the mRAG FastAPI service.

Transport: stdio (standard MCP)
Backend:   mRAG FastAPI at MRAG_URL (default http://localhost:7438) OR direct SQLite Bridge.

Tools
-----
query_memory   Read relevant memories for a text snippet
write_memory   Store a new memory in an adapter table
memory_stats   Live stats from the running mRAG service

Run via Claude Desktop — see claude_desktop_config.json.
Requires the mRAG FastAPI server to be running independently:
    cd /Users/kota/Maker/mRAG && uv run uvicorn api:app --port 7438
"""

from __future__ import annotations

import os
import sys
import httpx
from pathlib import Path
from fastmcp import FastMCP

MRAG_URL = os.getenv("MRAG_URL", "http://localhost:7438")
TIMEOUT  = 5.0

mcp = FastMCP("mRAG")

# ── Direct Bridge Mode setup ──────────────────────────────────────────────────

mrag_root = Path(__file__).parent.resolve()
if str(mrag_root) not in sys.path:
    sys.path.insert(0, str(mrag_root))

_local_bridge = None
_hasher = None
ContextTrigger = None
EngRamPayload = None

try:
    from mrag.bridge.interface import BridgeInterface
    from mrag.schema.bridge import ContextTrigger
    from mrag.schema.payload import EngRamPayload
    from mrag.hash.ngram_hasher import NgramHasher

    TABLES_DIR  = os.environ.get("MRAG_TABLES_DIR", str(mrag_root / "data" / "tables"))
    MAX_LOADED  = int(os.environ.get("MRAG_MAX_LOADED", "3"))
    TOP_N       = int(os.environ.get("MRAG_TOP_N", "5"))

    Path(TABLES_DIR).mkdir(parents=True, exist_ok=True)

    _local_bridge = BridgeInterface(tables_dir=TABLES_DIR, max_loaded=MAX_LOADED, top_n=TOP_N)
    _hasher = NgramHasher()
    print("[mRAG MCP] Direct SQLite Bridge Mode initialized.")
except Exception as e:
    print(f"[mRAG MCP] Direct Mode import failed: {e}. Falling back to HTTP loopback.")

# Use HTTP if MRAG_URL is explicitly set to non-default OR if local bridge is not available
USE_HTTP = (os.getenv("MRAG_URL") is not None) or (_local_bridge is None)

if not USE_HTTP and _local_bridge is not None:
    import atexit
    atexit.register(_local_bridge.shutdown)


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(path: str) -> dict:
    with httpx.Client(base_url=MRAG_URL, timeout=TIMEOUT) as client:
        r = client.get(path)
        r.raise_for_status()
        return r.json()


def _post(path: str, body: dict) -> dict:
    with httpx.Client(base_url=MRAG_URL, timeout=TIMEOUT) as client:
        r = client.post(path, json=body)
        r.raise_for_status()
        return r.json()


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def query_memory(
    text: str,
    adapter_hint: str = "unknown",
    adapter_hints: list[str] = [],
) -> str:
    """
    Retrieve memories relevant to a piece of text from the mRAG store.

    adapter_hint: persona or NPC name the memories belong to
                  (e.g. "blacksmith", "merchant", "dakota").
    adapter_hints: optional list of adapter names for cross-table query —
                  searches all listed adapters and merges by salience.
                  When empty, falls back to single adapter_hint query.
    Returns the top matching memory strings and affect metadata.
    """
    if not USE_HTTP and _local_bridge is not None and ContextTrigger is not None:
        try:
            pseudo_ids = _hasher.tokenize_text(text) if _hasher else []
            key = _hasher.lookup_key(pseudo_ids) if (_hasher and pseudo_ids) else "empty"
            
            if adapter_hints:
                response = _local_bridge.handle_cross_trigger(
                    adapter_hints=adapter_hints,
                    context_hash=key,
                    query_text=text,
                )
            else:
                response = _local_bridge.handle_trigger(ContextTrigger(
                    adapter_hint=adapter_hint,
                    context_hash=key,
                    prompt_preview=text[:128],
                ), query_text=text)

            tokens = response.memory_tokens
            if not tokens:
                hint = adapter_hints[0] if adapter_hints else adapter_hint
                return f"No memories found for adapter(s) '{hint}'."

            salience = response.salience_max
            affect = response.affect_mean
            label = response.adapter_label

            lines = [f"[mRAG (Local)] {label}  salience={salience:.2f}  affect={affect:+.2f}"]
            for i, t in enumerate(tokens, 1):
                lines.append(f"  {i}. {t}")
            return "\n".join(lines)
        except Exception as exc:
            # Fall through to HTTP
            pass

    # HTTP Fallback
    try:
        if adapter_hints:
            data = _post("/query_cross", {
                "text":          text,
                "adapter_hints": adapter_hints,
            })
        else:
            data = _post("/query_text", {
                "text":         text,
                "adapter_hint": adapter_hint,
            })
    except httpx.HTTPError as exc:
        return f"mRAG unavailable ({MRAG_URL}): {exc}"

    tokens = data.get("memory_tokens", [])
    if not tokens:
        hint = adapter_hints[0] if adapter_hints else adapter_hint
        return f"No memories found for adapter(s) '{hint}'."

    salience = data.get("salience_max", 0.0)
    affect   = data.get("affect_mean",  0.0)
    label    = data.get("adapter_label", "cross")

    lines = [f"[mRAG] {label}  salience={salience:.2f}  affect={affect:+.2f}"]
    for i, t in enumerate(tokens, 1):
        lines.append(f"  {i}. {t}")
    return "\n".join(lines)


@mcp.tool()
def write_memory(
    adapter_name: str,
    text:         str,
    salience:     float      = 0.8,
    affect:       float      = 0.0,
    tags:         list[str]  = [],
) -> str:
    """
    Store a memory in the mRAG store under an adapter/persona.

    adapter_name: persona or NPC name (e.g. "blacksmith", "dakota").
    text:         the memory string to store.
    salience:     importance [0.0–1.0]; high salience resists decay.
    affect:       emotional valence [-1.0–1.0]; 0 is neutral.
    tags:         optional category strings for salience boosting (e.g. ["craft", "trade"]).
    """
    if not USE_HTTP and _local_bridge is not None and EngRamPayload is not None:
        try:
            pseudo_ids = _hasher.tokenize_text(text) if _hasher else []
            key = _hasher.lookup_key(pseudo_ids) if (_hasher and pseudo_ids) else "empty"
            
            merged_tags = list(tags) if tags else []
            tag_adapter = f"adapter:{adapter_name}"
            if tag_adapter not in merged_tags:
                merged_tags.append(tag_adapter)

            payload = EngRamPayload(
                text=text,
                salience=salience,
                affect=affect,
                source=adapter_name,
                age=0,
                tags=merged_tags,
            )
            _local_bridge._manager.write_memory(adapter_name, key, payload)
            return f"Stored in '{adapter_name}' (key={key}) [Local Direct Mode]"
        except Exception as exc:
            # Fall through to HTTP
            pass

    # HTTP Fallback
    try:
        data = _post(f"/memory/{adapter_name}", {
            "text":     text,
            "salience": salience,
            "affect":   affect,
            "tags":     tags,
        })
    except httpx.HTTPError as exc:
        return f"Write failed: {exc}"

    return f"Stored in '{data.get('adapter', adapter_name)}' (key={data.get('key', '?')})"


@mcp.tool()
def memory_stats() -> str:
    """
    Live stats from the mRAG service: tables loaded, hit rate, eviction count.
    Use this to check whether the FastAPI server is reachable and healthy.
    """
    if not USE_HTTP and _local_bridge is not None:
        try:
            loaded = len(_local_bridge._manager)
            max_loaded = _local_bridge._manager._max_loaded
            ngram_n = _local_bridge._hasher.n if _local_bridge._hasher else 3
            return (
                f"[mRAG (Local)]\n"
                f"tables: {loaded}/{max_loaded}  "
                f"ngram_n: {ngram_n}"
            )
        except Exception as exc:
            # Fall through to HTTP
            pass

    # HTTP Fallback
    try:
        data = _get("/stats")
    except httpx.HTTPError as exc:
        return f"mRAG unavailable ({MRAG_URL}): {exc}"

    return (
        f"tables: {data['tables_loaded']}/{data['tables_max']}  "
        f"hit_rate: {data['hash_hit_rate']:.1%}  "
        f"evicted: {data['evicted']}  "
        f"ngram_n: {data['ngram_n']}"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
