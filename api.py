"""
mRAG FastAPI server — HTTP surface for Garden-v2, MIIN-kt, and the mRAG MCP.

Endpoints
---------
POST /query           ContextTrigger  → EngRamResponse  (Garden hot path)
POST /query_text      TextQueryRequest → EngRamResponse  (model-agnostic; hash server-side)
GET  /stats           → MragStats     (Garden control plane panel)
POST /memory/{adapter_name}           write NPC memory   (MIIN harvest)
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
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from mrag.bridge.interface import BridgeInterface
from mrag.schema.bridge import ContextTrigger, PidxSyncPacket
from mrag.schema.payload import EngRamPayload
from mrag.router.affect_router import AFFECT_BANDS
from mrag.hash.ngram_hasher import NgramHasher

# ── Config ───────────────────────────────────────────────────────────────────

TABLES_DIR  = os.environ.get("MRAG_TABLES_DIR", str(Path(__file__).parent / "data" / "tables"))
MAX_LOADED  = int(os.environ.get("MRAG_MAX_LOADED", "3"))
TOP_N       = int(os.environ.get("MRAG_TOP_N", "5"))

Path(TABLES_DIR).mkdir(parents=True, exist_ok=True)

# One bridge per process — shared EngRamManager LRU pool
_bridge = BridgeInterface(tables_dir=TABLES_DIR, max_loaded=MAX_LOADED, top_n=TOP_N)
_hasher = NgramHasher()

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
    """Write a single NPC memory into an adapter table.

    key is optional — if omitted, derived from text via word-level pseudo-tokenisation
    so MIIN can write memories without knowing token IDs.
    """
    text:     str
    salience: float = 0.8
    affect:   float = 0.0
    tags:     list[str] = []
    key:      Optional[str] = None


class TickRequest(BaseModel):
    steps: int = 1


class StatsResponse(BaseModel):
    tables_loaded:  int
    tables_max:     int
    hash_hit_rate:  float       # placeholder — tracked by Garden's local cache
    evicted:        int
    ngram_n:        int
    seed:           int
    affect_bands:   dict[str, list[float]]


# ── Session-scoped hit tracking (approximate — resets on restart) ──────────

_hit_count   = 0
_query_count = 0
_evicted_total = 0

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "loaded": _bridge._manager.loaded_adapters}


@app.post("/query")
def query(trigger: ContextTrigger):
    """Main Garden hot path. Called before mRNA Layer 25 fires."""
    global _hit_count, _query_count, _evicted_total
    _query_count += 1

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
    """Garden control plane panel data."""
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
    """MIIN harvest loop — write an NPC interaction memory to an adapter table.

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
        age=0,
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


@app.post("/memory/{adapter_name}/tick")
def tick_memory(adapter_name: str, req: TickRequest):
    """Age all entries in an adapter table by N steps. Called by MIIN session end."""
    _bridge._manager.tick_adapter(adapter_name, req.steps)
    evicted = _bridge._manager.evict_stale(adapter_name)
    return {"status": "ok", "steps": req.steps, "evicted": evicted}


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
