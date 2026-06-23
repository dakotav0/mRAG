"""
mrag_mcp.py — MCP server wrapping the mRAG FastAPI service.

Transport: stdio (standard MCP)
Backend:   mRAG FastAPI at MRAG_URL (default http://localhost:7438) OR direct SQLite Bridge.

Tools
-----
query_memory     Read relevant memories for a text snippet
write_memory     Store a new memory in an adapter table
write_observation  [Legend] Store a typed Observation (Layer 0)
query_observations [Legend] Query and deserialize Observations
memory_stats     Live stats from the running mRAG service

Lingua Franca / Layer 0 Legend Types
-------------------------------------
This server now imports the legend module from the lingua-franca blueprint:
  ~/agents/blueprints/lingua-franca/legend.py

When available, it registers Observation-typed tools on top of the existing
raw-memory interface. The Observation type is the shared currency between
mRAG, PIDX, the bridge, and agent profiles (Layer 0 of the ensemble).

Run via Claude Desktop — see claude_desktop_config.json.
Requires the mRAG FastAPI server to be running independently:
    cd /Users/kota/Maker/mRAG && uv run uvicorn api:app --port 7438
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
import httpx
from pathlib import Path
from fastmcp import FastMCP

# ── Lingua Franca (Layer 0 Legend Types) ─────────────────────────────────────
legend_path = Path(__file__).parent.parent.parent / "agents" / "blueprints" / "lingua-franca"
if str(legend_path) not in sys.path:
    sys.path.insert(0, str(legend_path))

_legend_loaded = False
try:
    from legend import Observation, WeightShift, ThreadStatus, ConfidenceDecay
    _legend_loaded = True
    print(f"[mRAG MCP] Lingua Franca legend v{getattr(Observation, '__module__', '0.1.0')} loaded.")
except ImportError as e:
    print(f"[mRAG MCP] Legend not available ({e}). Observation-typed tools disabled.")

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


def _delete(path: str) -> dict:
    with httpx.Client(base_url=MRAG_URL, timeout=TIMEOUT) as client:
        r = client.delete(path)
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
    age:          int        = 0,
    tags:         list[str]  = [],
) -> str:
    """
    Store a memory in the mRAG store under an adapter/persona.

    adapter_name: persona or NPC name (e.g. "blacksmith", "dakota").
    text:         the memory string to store.
    salience:     importance [0.0–1.0]; high salience resists decay.
    affect:       emotional valence [-1.0–1.0]; 0 is neutral.
    age:          initial age in decay steps.  Set >0 for information that
                  describes past events — creates natural salience spread.
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
                age=age,
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
            "age":      age,
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


@mcp.tool()
def list_adapters() -> str:
    """
    List all adapter names in the mRAG store, with load status and entry counts.
    Returns on-disk adapters even when they are not currently loaded in RAM.
    """
    if not USE_HTTP and _local_bridge is not None:
        try:
            backend = _local_bridge._manager._backend
            on_disk = backend.list_adapters()
            loaded = set(_local_bridge._manager.loaded_adapters)
            all_names = sorted(set(on_disk) | loaded)
            lines = [f"[mRAG (Local)] {len(all_names)} adapters:"]
            for name in all_names:
                status = "loaded" if name in loaded else "on-disk"
                count = backend.count(name)
                lines.append(f"  {name} ({status}, {count} entries)")
            return "\n".join(lines)
        except Exception as exc:
            pass

    # HTTP Fallback
    try:
        data = _get("/adapters")
    except httpx.HTTPError as exc:
        return f"mRAG unavailable ({MRAG_URL}): {exc}"

    lines = [f"[mRAG] {data['total']} adapters:"]
    for a in data["adapters"]:
        status = "loaded" if a["loaded"] else "on-disk"
        lines.append(f"  {a['name']} ({status}, {a['entry_count']} entries)")
    return "\n".join(lines)


@mcp.tool()
def delete_memory(
    adapter_name: str,
    key: str,
) -> str:
    """
    Delete a single memory entry by adapter and key.
    No-op if the key does not exist.
    """
    if not USE_HTTP and _local_bridge is not None:
        try:
            table = _local_bridge._manager.mount(adapter_name)
            if key in table:
                del table._store[key]
                return f"Deleted from '{adapter_name}' (key={key})"
            return f"Not found in '{adapter_name}' (key={key})"
        except Exception as exc:
            pass

    # HTTP Fallback
    try:
        data = _delete(f"/memory/{adapter_name}/{key}")
    except httpx.HTTPError as exc:
        return f"Delete failed: {exc}"

    return f"{data['status']}: {data.get('key', key)}"


@mcp.tool()
def get_memory(
    adapter_name: str,
    key: str,
) -> str:
    """
    Direct keyed lookup of a single memory entry.
    Returns full payload (text, salience, affect, tags, age) or 404.
    """
    if not USE_HTTP and _local_bridge is not None:
        try:
            table = _local_bridge._manager.mount(adapter_name)
            payload = table.get(key)
            if payload is None:
                return f"Not found: '{adapter_name}' key={key}"
            return (
                f"[{adapter_name}] key={key}\n"
                f"  text: {payload.text[:200]}{'...' if len(payload.text) > 200 else ''}\n"
                f"  salience: {payload.salience:.2f}  affect: {payload.affect:+.2f}\n"
                f"  age: {payload.age}  tags: {payload.tags}"
            )
        except Exception as exc:
            pass

    # HTTP Fallback
    try:
        data = _get(f"/memory/{adapter_name}/{key}")
    except httpx.HTTPError as exc:
        return f"mRAG unavailable ({MRAG_URL}): {exc}"

    return (
        f"[{data['adapter']}] key={data['key']}\n"
        f"  text: {data['text'][:200]}{'...' if len(data['text']) > 200 else ''}\n"
        f"  salience: {data['salience']:.2f}  affect: {data['affect']:+.2f}\n"
        f"  age: {data['age']}  tags: {data['tags']}"
    )


@mcp.tool()
def update_memory(
    adapter_name: str,
    key: str,
    salience: float | None = None,
    affect: float | None = None,
    tags: list[str] | None = None,
    text: str | None = None,
) -> str:
    """
    Update an existing memory entry's salience, affect, tags, or text.
    Returns not_found if the key does not exist. Only provided fields are changed.
    Primary use: thread lifecycle → mRAG salience propagation.
    """
    if not USE_HTTP and _local_bridge is not None:
        try:
            table = _local_bridge._manager.mount(adapter_name)
            payload = table.get(key)
            if payload is None:
                return f"Not found: '{adapter_name}' key={key}"
            if salience is not None:
                payload.salience = max(0.0, min(1.0, salience))
            if affect is not None:
                payload.affect = max(-1.0, min(1.0, affect))
            if tags is not None:
                payload.tags = list(tags)
            if text is not None:
                payload.text = text
            return f"Updated '{adapter_name}' key={key} (salience={payload.salience:.2f}, affect={payload.affect:+.2f})"
        except Exception as exc:
            pass

    # HTTP Fallback
    try:
        data = _post(f"/memory/{adapter_name}/{key}/update", {
            "salience": salience,
            "affect": affect,
            "tags": tags,
            "text": text,
        })
    except httpx.HTTPError as exc:
        return f"Update failed: {exc}"

    return f"Updated '{data['adapter']}' key={data['key']}"


# ── Lingua Franca Legend-Typed Tools ──────────────────────────────────────────

if _legend_loaded:

    @mcp.tool()
    def write_observation(
        source: str,
        field: str,
        value: str,
        confidence: float = 0.5,
        salience: float = 0.8,
        thread_id: str | None = None,
        raw: str | None = None,
    ) -> str:
        """
        Store a legend-typed Observation in the mRAG store.

        The Observation is the Layer 0 currency of the ensemble — a structured
        record of a behavioral, cognitive, or preference observation. Every
        subsystem (PIDX, bridge, agent profiles) speaks this type.

        source:      origin identifier ("mrag" | "pidx" | "model" | "user")
        field:       dot-path category ("identity.core", "working.pattern")
        value:       the observation value (JSON-encoded string)
        confidence:  0.0–1.0 how certain the observation is
        salience:    0.0–1.0 importance; high salience resists decay
        thread_id:   optional thread this observation belongs to
        raw:         optional verbatim source text
        """
        obs = Observation(
            source=source,
            field=field,
            value=value,
            confidence=confidence,
            salience=salience,
            timestamp=datetime.utcnow().isoformat(),
            thread_id=thread_id,
            raw=raw,
        )
        serialized = json.dumps(obs.to_dict())
        tags = [f"type:observation", f"field:{field}", f"source:{source}"]
        if thread_id:
            tags.append(f"thread:{thread_id}")

        return str(_store_legend_entry(source, serialized, salience, tags))

    @mcp.tool()
    def query_observations(
        text: str,
        adapter_hint: str = "unknown",
        adapter_hints: list[str] = [],
        min_confidence: float = 0.0,
    ) -> str:
        """
        Query mRAG and return results deserialized as Observation objects.

        Filters for entries tagged as legend Observations and applies an
        optional confidence floor. Returns a formatted list with typed
        metadata (field, source, confidence, salience, thread_id).

        text:          query text for semantic retrieval
        adapter_hint:  single persona name (e.g. "ada", "dakota")
        adapter_hints: cross-table query (supersedes adapter_hint)
        min_confidence: minimum confidence threshold [0.0–1.0]
        """
        raw_result = _query_memory_internal(text, adapter_hint, adapter_hints)
        return _format_observation_results(raw_result, min_confidence)


def _store_legend_entry(adapter_name: str, serialized: str, salience: float, tags: list[str]) -> str:
    """Internal helper — stores an Observation JSON blob into mRAG.

    Returns the storage confirmation string from the underlying write_memory path.
    Works in Local Direct Mode or HTTP fallback.
    """
    # Local Direct Mode
    if not USE_HTTP and _local_bridge is not None and _hasher is not None and EngRamPayload is not None:
        try:
            pseudo_ids = _hasher.tokenize_text(serialized) if _hasher else []
            key = _hasher.lookup_key(pseudo_ids) if (_hasher and pseudo_ids) else "empty"

            merged_tags = list(tags) if tags else []
            tag_adapter = f"adapter:{adapter_name}"
            if tag_adapter not in merged_tags:
                merged_tags.append(tag_adapter)

            payload = EngRamPayload(
                text=serialized,
                salience=salience,
                affect=0.0,
                source=adapter_name,
                age=0,
                tags=merged_tags,
            )
            _local_bridge._manager.write_memory(adapter_name, key, payload)
            return f"Observation stored in '{adapter_name}' (key={key}) [Local Direct Mode]"
        except Exception:
            pass

    # HTTP Fallback
    try:
        data = _post(f"/memory/{adapter_name}", {
            "text": serialized,
            "salience": salience,
            "affect": 0.0,
            "tags": tags,
        })
    except httpx.HTTPError as exc:
        return f"Observation write failed: {exc}"

    return f"Observation stored in '{data.get('adapter', adapter_name)}' (key={data.get('key', '?')})"


def _query_memory_internal(text: str, adapter_hint: str, adapter_hints: list[str]) -> list[dict]:
    """Internal helper — query mRAG and return raw token dicts.

    Returns a list of dicts with 'text', 'salience_max', 'affect_mean' keys,
    or an empty list on failure.
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
                response = _local_bridge.handle_trigger(
                    ContextTrigger(
                        adapter_hint=adapter_hint,
                        context_hash=key,
                        prompt_preview=text[:128],
                    ),
                    query_text=text,
                )
            tokens = response.memory_tokens
            if not tokens:
                return []
            return [
                {"text": t, "salience_max": response.salience_max, "affect_mean": response.affect_mean}
                for t in tokens
            ]
        except Exception:
            pass

    try:
        if adapter_hints:
            data = _post("/query_cross", {
                "text": text,
                "adapter_hints": adapter_hints,
            })
        else:
            data = _post("/query_text", {
                "text": text,
                "adapter_hint": adapter_hint,
            })
    except httpx.HTTPError:
        return []

    tokens = data.get("memory_tokens", [])
    if not tokens:
        return []
    salience = data.get("salience_max", 0.0)
    affect = data.get("affect_mean", 0.0)
    return [{"text": t, "salience_max": salience, "affect_mean": affect} for t in tokens]


def _format_observation_results(tokens: list[dict], min_confidence: float) -> str:
    """Internal helper — parse raw tokens as Observations and format for display."""
    if not tokens:
        return "No observations found."

    lines = [f"[mRAG] Observation results (min_confidence={min_confidence:.2f}):"]
    obs_count = 0

    for i, entry in enumerate(tokens, 1):
        text = entry.get("text", "")
        salience = entry.get("salience_max", 0.0)
        affect = entry.get("affect_mean", 0.0)

        # Try to parse as serialized Observation
        try:
            data = json.loads(text)
            obs = Observation.from_dict(data)
            if obs.confidence < min_confidence:
                continue
            obs_count += 1
            lines.append(
                f"  {obs_count}. [{obs.field}] (src={obs.source}, "
                f"conf={obs.confidence:.2f}, sal={obs.salience:.2f})"
            )
            lines.append(f"     value: {str(obs.value)[:120]}")
            if obs.thread_id:
                lines.append(f"     thread: {obs.thread_id}")
            if obs.raw:
                lines.append(f"     raw: {obs.raw[:100]}")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Not an Observation — show as raw memory with confidence placeholder
            conf = max(0.0, min(1.0, salience))  # estimate
            if conf < min_confidence:
                continue
            obs_count += 1
            lines.append(f"  {obs_count}. [raw] (salience={salience:.2f}, affect={affect:+.2f})")
            lines.append(f"     {text[:200]}")

    if obs_count == 0:
        return "No observations meet the confidence threshold."

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
