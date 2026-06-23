"""
mRAG FastAPI server — HTTP surface for agent pipelines and the mRAG MCP.

Endpoints
---------
POST /query           ContextTrigger  → EngRamResponse  (main query path)
POST /query_text      TextQueryRequest → EngRamResponse  (model-agnostic; hash server-side)
GET  /stats           → MragStats     (runtime stats)
POST /memory/{adapter_name}           write adapter memory
DELETE /memory/{adapter_name}/{key}   evict one entry
POST /memory/{adapter_name}/tick      age all entries N steps
POST /sync            PidxSyncPacket  → apply decay/boost
GET  /health          liveness check

Run:
    uv run uvicorn api:app --port 7438 --reload

Env vars:
    MRAG_TABLES_DIR  Path to SQLite .db directory (default: ./data/tables)
    MRAG_MAX_LOADED  Max adapter tables in RAM (default: 3)
    MRAG_TOP_N       Memories per EngRamResponse (default: 5)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from mrag.bridge.interface import BridgeInterface
from mrag.schema.bridge import ContextTrigger, PidxSyncPacket
from mrag.schema.payload import EngRamPayload
from mrag.router.affect_router import AFFECT_BANDS
from mrag.router.decay import EVICTION_THRESHOLD
from mrag.hash.ngram_hasher import NgramHasher

# ── Config ───────────────────────────────────────────────────────────────────

TABLES_DIR  = os.environ.get("MRAG_TABLES_DIR", str(Path(__file__).parent / "data" / "tables"))
MAX_LOADED  = int(os.environ.get("MRAG_MAX_LOADED", "3"))
TOP_N       = int(os.environ.get("MRAG_TOP_N", "5"))

Path(TABLES_DIR).mkdir(parents=True, exist_ok=True)

# One bridge per process — shared EngRamManager LRU pool
_bridge = BridgeInterface(tables_dir=TABLES_DIR, max_loaded=MAX_LOADED, top_n=TOP_N)
_hasher = NgramHasher()

# ── Auto-tick (time-based decay) ─────────────────────────────────────────────

# 1 step ≈ N seconds of wall-clock time. At 86400s (24h) per step and λ=0.05,
# a memory at 0.8 salience survives ~33 days before hitting the 0.15 threshold.
# Adjust TICK_INTERVAL to make decay faster (lower) or slower (higher).
TICK_INTERVAL_SECONDS: float = 86_400.0  # seconds per decay step (24h)

_last_tick_mono: float = time.monotonic()  # monotonic clock, survives system sleep


def _auto_tick() -> None:
    """Apply time-based decay to all mounted adapters since last call.

    Calculates elapsed wall time, converts to whole decay steps, and ticks
    every adapter currently in RAM.  This keeps salience drifting in real
    time without requiring an external cron job.

    Each adapter uses its own eviction threshold (identity adapters decay
    slower than session adapters).

    Called at the top of every query endpoint.  Decay naturally slows as
    entries cross the eviction threshold — nothing forces them out; they
    just stop appearing in results because their decayed salience is below
    the threshold.
    """
    global _last_tick_mono
    now = time.monotonic()
    elapsed = now - _last_tick_mono
    if elapsed < TICK_INTERVAL_SECONDS:
        return  # not enough time has passed

    steps = int(elapsed / TICK_INTERVAL_SECONDS)
    if steps < 1:
        return

    # Tick every adapter currently in RAM
    adapters = list(_bridge._manager.loaded_adapters)
    for name in adapters:
        try:
            _bridge._manager.tick_adapter(name, steps)
        except Exception:
            pass  # skip adapters that fail to mount

    # Eviction: prune stale entries using per-adapter thresholds
    for name in adapters:
        try:
            threshold = _get_eviction_threshold(name)
            _bridge._manager.evict_stale(name, threshold)
        except Exception:
            pass

    _last_tick_mono = now


# ── Per-adapter eviction threshold overrides ────────────────────────────
# Adapters not listed use EVICTION_THRESHOLD (0.15).  Identity-bearing
# adapters use a lower threshold so core observations persist longer.
PER_ADAPTER_THRESHOLD: dict[str, float] = {
    "ada":       0.05,   # identity — very slow decay
    "dakota":    0.05,   # identity — very slow decay
    "naomi":     0.05,   # identity — very slow decay
    "Elia":      0.05,   # identity — very slow decay
    "elia":      0.05,   # identity — very slow decay
    "nested-learning": 0.08,  # knowledge — slow decay
    "research":        0.08,  # knowledge — slow decay
}


def _get_eviction_threshold(adapter_name: str) -> float:
    """Return the eviction threshold for the given adapter.

    Falls back to the global EVICTION_THRESHOLD if no per-adapter override
    is configured.
    """
    return PER_ADAPTER_THRESHOLD.get(adapter_name, EVICTION_THRESHOLD)


app = FastAPI(title="mRAG", version="0.1.0")

# ── Request/Response models ───────────────────────────────────────────────────

class TextQueryRequest(BaseModel):
    """Query by raw text — hash is computed server-side via _text_to_key."""
    text:           str
    adapter_hint:   str = "unknown"
    prompt_preview: Optional[str] = None


class CrossQueryRequest(BaseModel):
    """Query across multiple adapters — merges results by salience."""
    text:          str
    adapter_hints: list[str]
    prompt_preview: Optional[str] = None


class MemoryWriteRequest(BaseModel):
    """Write a single memory into an adapter table.

    key is optional — if omitted, derived from text via word-level pseudo-tokenisation
    so callers can write memories without knowing token IDs.

    age is optional — set to 0 (fresh memory) by default.  When writing data
    that describes past events, set age to an estimate of how many steps old
    the information already is.  This creates natural salience spread:
    older information starts lower, fresh info starts higher.
    """
    text:     str
    salience: float = 0.8
    affect:   float = 0.0
    age:      int = 0     # initial age in steps — stagger this for spread
    tags:     list[str] = []
    key:      Optional[str] = None


class TickRequest(BaseModel):
    steps: int = 1
    threshold: Optional[float] = None  # override eviction threshold per tick


class UpdateMemoryRequest(BaseModel):
    """Partial update — all fields optional, only provided fields change."""
    salience: Optional[float] = None
    affect:   Optional[float] = None
    tags:     Optional[list[str]] = None
    text:     Optional[str] = None


class BulkTagSalienceRequest(BaseModel):
    """Bulk salience update for all entries with a given tag."""
    salience: float


class StatsResponse(BaseModel):
    tables_loaded:  int
    tables_max:     int
    hash_hit_rate:  float       # placeholder — tracked by the caller's local cache
    evicted:        int
    ngram_n:        int
    seed:           int
    affect_bands:   dict[str, list[float]]


# ── Session-scoped hit tracking (approximate — resets on restart) ──────────

_hit_count   = 0
_query_count = 0
_evicted_total = 0

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/adapters")
def list_adapters():
    """List all adapter names (on-disk + loaded) with metadata."""
    on_disk = _bridge._manager._backend.list_adapters()
    loaded = set(_bridge._manager.loaded_adapters)
    result = []
    for name in sorted(set(on_disk) | loaded):
        info = {
            "name": name,
            "loaded": name in loaded,
            "on_disk": name in on_disk,
        }
        try:
            info["entry_count"] = _bridge._manager._backend.count(name)
        except Exception:
            info["entry_count"] = 0
        result.append(info)
    return {"adapters": result, "total": len(result)}


@app.get("/health")
def health():
    return {"status": "ok", "loaded": _bridge._manager.loaded_adapters}


@app.post("/query")
def query(trigger: ContextTrigger):
    """Main query path. Returns memory context for the given adapter trigger."""
    global _hit_count, _query_count, _evicted_total
    _query_count += 1

    _auto_tick()  # time-based decay before every query

    response = _bridge.handle_trigger(trigger)
    _evicted_total += response.evicted_count

    if response.salience_max > 0.0:
        _hit_count += 1

    return response


@app.post("/query_text")
def query_text(req: TextQueryRequest):
    """Model-agnostic query: callers pass raw text, hash is computed here."""
    global _hit_count, _query_count, _evicted_total
    _query_count += 1

    _auto_tick()  # time-based decay before every query

    key = _text_to_key(req.text)
    response = _bridge.handle_trigger(ContextTrigger(
        adapter_hint=req.adapter_hint,
        context_hash=key,
        prompt_preview=(req.prompt_preview or req.text)[:128],
    ), query_text=req.text)
    _evicted_total += response.evicted_count
    if response.salience_max > 0.0:
        _hit_count += 1
    return response


@app.post("/query_cross")
def query_cross(req: CrossQueryRequest):
    """Cross-table query: searches multiple adapters, merges by salience + similarity."""
    global _hit_count, _query_count, _evicted_total
    _query_count += 1

    _auto_tick()  # time-based decay before every query

    key = _text_to_key(req.text)
    response = _bridge.handle_cross_trigger(
        adapter_hints=req.adapter_hints,
        context_hash=key,
        query_text=req.text,
    )
    _evicted_total += response.evicted_count

    if response.salience_max > 0.0:
        _hit_count += 1

    return response


@app.get("/stats", response_model=StatsResponse)
def stats():
    """Runtime stats for the memory broker."""
    hit_rate = (_hit_count / _query_count) if _query_count else 0.0
    bands = {b.name: [b.low, b.high] for b in AFFECT_BANDS}
    return StatsResponse(
        tables_loaded=len(_bridge._manager),
        tables_max=MAX_LOADED,
        hash_hit_rate=hit_rate,
        evicted=_evicted_total,
        ngram_n=_hasher.n,
        seed=_hasher.seed,
        affect_bands=bands,
    )


@app.post("/memory/{adapter_name}")
def write_memory(adapter_name: str, req: MemoryWriteRequest):
    """Write an interaction memory to an adapter table.

    Auto-tags every write with the adapter name so decay passes can be
    context-aware (e.g., PIDX can boost "drift" tags separately from
    "session" tags). Caller-provided tags are merged in.
    """
    key = req.key or _text_to_key(req.text)
    tags = list(req.tags) if req.tags else []
    # Auto-tag with adapter for context-aware decay
    tag_adapter = f"adapter:{adapter_name}"
    if tag_adapter not in tags:
        tags.append(tag_adapter)
    payload = EngRamPayload(
        text=req.text,
        salience=req.salience,
        affect=req.affect,
        source=adapter_name,
        age=req.age,
        tags=tags,
    )
    _bridge._manager.write_memory(adapter_name, key, payload)
    return {"status": "ok", "adapter": adapter_name, "key": key, "tags": tags}


@app.get("/memory/{adapter_name}/{key}")
def get_memory(adapter_name: str, key: str):
    """Direct key lookup — returns the full payload or 404."""
    table = _bridge._manager.mount(adapter_name)
    payload = table.get(key)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"key {key} not found in adapter {adapter_name}")
    return {
        "adapter": adapter_name,
        "key": key,
        "text": payload.text,
        "salience": payload.salience,
        "affect": payload.affect,
        "source": payload.source,
        "age": payload.age,
        "tags": payload.tags,
        "decayed_salience": payload.decayed_salience(),
    }


@app.delete("/memory/{adapter_name}/{key}")
def delete_memory(adapter_name: str, key: str):
    """Explicitly evict one entry. No-op if key does not exist."""
    table = _bridge._manager.mount(adapter_name)
    if key in table:
        del table._store[key]
        return {"status": "evicted", "key": key}
    return {"status": "not_found", "key": key}


@app.post("/memory/{adapter_name}/{key}/update")
def update_memory(adapter_name: str, key: str, req: UpdateMemoryRequest):
    """Partial update — only provided fields change. Thread lifecycle → salience hook."""
    table = _bridge._manager.mount(adapter_name)
    payload = table.get(key)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"key {key} not found in adapter {adapter_name}")
    if req.salience is not None:
        payload.salience = max(0.0, min(1.0, req.salience))
    if req.affect is not None:
        payload.affect = max(-1.0, min(1.0, req.affect))
    if req.tags is not None:
        payload.tags = list(req.tags)
    if req.text is not None:
        payload.text = req.text
    return {
        "status": "updated",
        "adapter": adapter_name,
        "key": key,
        "salience": payload.salience,
        "affect": payload.affect,
        "tags": payload.tags,
    }


@app.post("/memory/{adapter_name}/tick")
def tick_memory(adapter_name: str, req: TickRequest):
    """Age all entries in an adapter table by N steps. Typically called at session end.

    Uses the per-adapter eviction threshold by default.  Pass `threshold` in
    the request body to override for this tick.
    """
    _bridge._manager.tick_adapter(adapter_name, req.steps)
    threshold = req.threshold if req.threshold is not None else _get_eviction_threshold(adapter_name)
    evicted = _bridge._manager.evict_stale(adapter_name, threshold)
    return {"status": "ok", "steps": req.steps, "evicted": evicted}


@app.post("/memory/{adapter_name}/tag/{tag}")
def update_by_tag(adapter_name: str, tag: str, req: BulkTagSalienceRequest):
    """Bulk salience update for all entries tagged with <tag>.
    
    Thread lifecycle hook: when a thread transitions to latent or closed,
    all mRAG memories tagged with that thread slug get their salience
    adjusted. No-op if no entries match the tag.
    """
    table = _bridge._manager.mount(adapter_name)
    entries = table.query_by_tag(tag)
    updated = 0
    for key, payload in entries:
        payload.salience = max(0.0, min(1.0, req.salience))
        updated += 1
    return {"status": "ok", "adapter": adapter_name, "tag": tag, "updated": updated}


@app.post("/sync")
def sync(packet: PidxSyncPacket):
    """PIDX decay tick + salience boost. Called when PIDX pushes identity updates."""
    _bridge.handle_pidx_sync(packet)
    return {"status": "ok"}


# ── Shutdown ──────────────────────────────────────────────────────────────────

import atexit
atexit.register(_bridge.shutdown)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _text_to_key(text: str) -> str:
    """
    Derive a lookup key from raw text via character trigrams → N-gram hasher.

    Uses NgramHasher.tokenize_text() to produce pseudo-token IDs, then
    lookup_key() to find the dominant N-gram hash. Character trigrams are
    vocabulary-independent and deterministic — same text always maps to the
    same key regardless of tokenizer or platform.
    """
    pseudo_ids = _hasher.tokenize_text(text)
    return _hasher.lookup_key(pseudo_ids) if pseudo_ids else "empty"
