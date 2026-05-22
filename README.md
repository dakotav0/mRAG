# 🧠 mRAG: Working Memory at the Frontier

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

`mRAG` is a decoupled, ultra-high-performance, local-first associative engram caching framework. Designed for sub-millisecond retrieval in real-time LLM agent pipelines, it brokers active working memory between identity engines (such as `PIDX`) and execution layers (such as `mRNA`).

Originally developed as part of a gaming intelligence network, this library has been fully decoupled, generalized, and optimized as a standalone open-source repository.

---

## ❓ Why mRAG?

Most memory systems for LLM agents do one of two things: vector search over embeddings (slow, needs an embedding model) or full-text keyword match (brittle, no semantic layering). Both add latency to the critical path — the moment between a context trigger and the agent's next token.

mRAG takes a different bet: **associative memory with configurable decay**. Instead of searching, it *routes*. A hot in-memory hash index resolves lookups in under a millisecond. Memories decay over time unless they're reinforced — so stale information fades naturally, and what matters stays at the surface.

It was built for real-time agent pipelines where every millisecond counts. It doesn't replace vector search — it sits upstream of it, keeping working memory fast and letting long-term retrieval happen out of band.

**Use cases:**
* **Agent identity and continuity** (paired with [PIDX](https://github.com/<org>/pidx))
* **Real-time NPC state** in game engines
* **Session context** that decays dynamically between conversations
* **Anywhere you need "what's relevant right now"** faster than an embedding lookup

---

## 📦 Installation

```bash
# Recommended: uv
uv add git+https://github.com/<org>/mrag.git

# pip
pip install git+https://github.com/<org>/mrag.git

# From source
git clone https://github.com/<org>/mrag.git
cd mrag
uv sync
```

**Requirements:** Python ≥ 3.10, SQLite3. No GPU, no API keys, no external services required.

---

## ⚖️ Key Features

* **🧬 Generic Engram Payloads (`Engram[T]`)**: Use any custom data structure (dictionaries, vectors, serialized objects, or trees) as your memory payload. Includes a 100% backward-compatible string-based `EngRamPayload` for drop-in replacement in legacy pipelines.
* **⏱️ Pluggable Decay Policies**: Implement custom memory eviction strategies via the `DecayPolicy` protocol. Out-of-the-box policies include:
  * `ExponentialDecay`: Discrete step-based decay with $O(1)$ lookup mathematical optimization.
  * `LinearDecay`: Fixed subtraction per time step, clamping at 0.0.
  * `NoDecay`: Infinite memory persistence.
* **🎛️ Configurable Affect Router**: Route retrieval metrics into emotional state or context-switching labels using dependency-injected valence bands and customizable string formatting callbacks (e.g. `concept::band` instead of `concept_band`).
* **⚡ Sub-Millisecond Retrieval**: Built around in-memory hash indexes with asynchronous prefetching capabilities and global Least Recently Used (LRU) memory table management backed by localized SQLite databases.

---

## 🏗️ Architecture

```
                       ┌─────────────────────────┐
                       │  mRNA Layer 14 SAE Fire │
                       └────────────┬────────────┘
                                    │ (ContextTrigger)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ mRAG BridgeInterface                                                   │
│                                                                        │
│  1. Mount Table (LRU Global Eviction) ──► SqliteBackend (.db File)     │
│  2. Evict Stale Memories               ──► Injected DecayPolicy        │
│  3. Hot Hash Lookup / Miss Re-ranking  ──► NgramHasher                 │
│  4. Formulate Emotional Valence Label  ──► Configurable AffectRouter   │
│                                                                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (EngRamResponse)
                                    ▼
                       ┌─────────────────────────┐
                       │  mRNA Layer 25 Injection│
                       └─────────────────────────┘
```

---

## 🔗 Bundled with PIDX

mRAG ships as a standalone library and as the memory backend for [PIDX](https://github.com/<org>/pidx), the Personality Index for LLM agents. Together they provide:

| Layer | Tool | Role |
|---|---|---|
| **Identity** | [PIDX](https://github.com/<org>/pidx) | Structured observations, confidence tracking, profile diffing |
| **Memory** | [mRAG](https://github.com/<org>/mrag) | Associative retrieval, decay, affect routing |
| **Bridge** | `.bridge.json` | Cross-agent session packets |

See the [PIDX repository](https://github.com/<org>/pidx) for the full integration guide.

---

## 🚀 Quick Start

### 1. Generic Payloads

```python
from mrag.schema.payload import Engram

# Store dictionary payloads instead of raw strings
custom_engram = Engram[dict](
    value={"health": 90, "mana": 45},
    salience=0.9,
    affect=0.1,
    source="player_stats"
)

print(custom_engram.value["health"])  # -> 90
```

### 2. Custom Decay & Eviction

```python
from mrag.router.decay import LinearDecay
from mrag.store.engram_table import EngRamTable
from mrag.schema.payload import EngRamPayload

# Evict memories linearly over time
table = EngRamTable("alchemist", decay_policy=LinearDecay(rate=0.05))

table.put("key_1", EngRamPayload("Player stole a potion.", salience=0.3))

table.tick(4)  # salience decays to 0.3 - (4 * 0.05) = 0.1
table.evict_stale(threshold=0.15)  # key_1 is evicted!
```

### 3. Configurable Affect Router

```python
from mrag.router.affect_router import AffectRouter, _Band

# Configure custom emotional bands and formatting callbacks
custom_bands = [
    _Band(low=0.2, high=1.01, name="friendly"),
    _Band(low=-0.2, high=0.2, name="neutral"),
    _Band(low=-1.01, high=-0.2, name="hostile")
]

router = AffectRouter(
    bands=custom_bands,
    format_fn=lambda hint, band: f"adapter::{hint}::{band}"
)

label = router.route(affect=-0.5, adapter_hint="guard")
print(label)  # -> "adapter::guard::hostile"
```

---

## 🧪 Testing

We maintain a 100% local, dependency-free test harness asserting exact decay math, latency constraints, and routing boundaries.

Run tests using the standard `pytest` runner:

```bash
uv run pytest tests/
```

Individual module-level smoke tests are available and can be run directly:

```bash
uv run python mrag/schema/payload.py
uv run python mrag/router/decay.py
uv run python mrag/router/affect_router.py
uv run python mrag/store/sqlite_backend.py
```

---

## 📄 Project Status & License

mRAG is stable and in active use. The API surface (`Engram`, `DecayPolicy`, `AffectRouter`, `BridgeInterface`) is fully versioned and backward-compatible. Contributions and custom adapter builds are welcome.

This project is licensed under the Apache License 2.0.
