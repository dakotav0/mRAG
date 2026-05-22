# mRAG — Memory Broker Toolkit

> **Role in the stack:** Sits between PIDX (identity) and mRNA (execution).
> Receives a context hash from mRNA's Layer 14 SAE, pulls the relevant
> EngRamPayload from CPU RAM, and returns an adapter label + injected
> memory tokens before Layer 25 fires.
>
> All three repos communicate exclusively via **BridgePacket JSON**.
> mRAG has zero knowledge of llama.cpp internals. mRNA has zero knowledge of
> SQLite. PIDX has zero knowledge of adapters. That's the contract.

---

## Directory Tree

```
mrag/
├── pyproject.toml
├── README.md
│
├── mrag/
│   ├── __init__.py
│   │
│   ├── schema/
│   │   ├── __init__.py
│   │   ├── payload.py          # EngRamPayload dataclass (text, salience, affect)
│   │   └── bridge.py           # BridgePacket in/out schemas (pydantic)
│   │
│   ├── store/
│   │   ├── __init__.py
│   │   ├── engram_table.py     # Per-adapter hash table: ngram_hash → EngRamPayload
│   │   ├── sqlite_backend.py   # Persistent backing store (SQLite per adapter)
│   │   └── manager.py          # Mount/unmount tables into system RAM on demand
│   │
│   ├── hash/
│   │   ├── __init__.py
│   │   └── ngram_hasher.py     # Configurable N-gram hashing (N, token level, seed)
│   │
│   ├── router/
│   │   ├── __init__.py
│   │   ├── affect_router.py    # affect float → adapter label string
│   │   └── decay.py            # PIDX-compatible salience decay (exponential)
│   │
│   ├── prefetch/
│   │   ├── __init__.py
│   │   └── coordinator.py      # Async prefetch: schedule table load before Layer 25
│   │
│   └── bridge/
│       ├── __init__.py
│       └── interface.py        # JSON BridgePacket I/O — the only public surface
│
├── tests/
│   ├── fixtures/
│   │   └── mock_packets.json   # Sample BridgePackets for harness (no external dependencies needed)
│   ├── test_route_accuracy.py  # Does salience+affect combo → correct adapter label?
│   ├── test_decay.py           # Do low-salience memories drop out over time steps?
│   └── test_latency.py         # End-to-end < 500ms budget on RTX 4070 Super config
│
└── data/
    └── memory_tables/          # Per-NPC SQLite dbs (gitignored, generated at runtime)
        ├── blacksmith.db
        └── merchant.db
```

---

## Core Schemas

### `mrag/schema/payload.py`

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EngRamPayload:
    """
    The structured object returned on an Engram hash hit.

    text:     The memory string injected into the prompt context.
    salience: [0.0, 1.0] — how important this memory is.
              High salience resists PIDX exponential decay.
              Threshold for KV cache eviction: salience < 0.15.
    affect:   [-1.0, 1.0] — emotional valence.
              Positive → warm/friendly adapter family.
              Negative → hostile/cautious adapter family.
              0.0 → neutral/professional adapter.
    source:   Which adapter table this memory belongs to.
    age:      Simulated time steps since memory was written.
              Used by decay engine; not injected into prompt.
    """
    text:     str
    salience: float           # 0.0–1.0
    affect:   float           # -1.0 to +1.0
    source:   str             # e.g. "blacksmith", "merchant"
    age:      int   = 0       # simulated time steps
    tags:     list  = field(default_factory=list)

    def decayed_salience(self, decay_rate: float = 0.05) -> float:
        """PIDX-compatible exponential decay."""
        return self.salience * (1.0 - decay_rate) ** self.age

    def is_evictable(self, threshold: float = 0.15) -> bool:
        return self.decayed_salience() < threshold
```

---

### `mrag/schema/bridge.py`

```python
from pydantic import BaseModel
from typing import Optional, Literal

# ── Inbound: mRNA Layer 14 fires this when it detects a semantic concept ──

class ContextTrigger(BaseModel):
    """Sent by mRNA Layer 14 SAE → mRAG."""
    adapter_hint:   str           # SAE latent label, e.g. "trading", "combat"
    context_hash:   str           # Hex digest of N-gram hash of the prompt
    prompt_preview: str           # First 128 chars for debugging (never injected)
    layer:          int = 14

# ── Outbound: mRAG returns this to mRNA before Layer 25 ──

class EngRamResponse(BaseModel):
    """Sent by mRAG → mRNA Layer 25."""
    adapter_label:   str          # e.g. "warm_merchant", "hostile_guard"
    memory_tokens:   list[str]    # Short facts to prepend to prompt context
    salience_max:    float        # Highest salience among retrieved memories
    affect_mean:     float        # Mean affect across retrieved memories
    evicted_count:   int = 0      # How many memories were below eviction threshold

# ── PIDX sync: PIDX pushes identity/decay updates to mRAG ──

class PidxSyncPacket(BaseModel):
    """Sent by PIDX → mRAG on identity change or decay tick."""
    npc_id:         str
    adapter_name:   str
    decay_delta:    float         # How much to age all memories for this NPC
    salience_boost: Optional[dict[str, float]] = None  # tag → salience modifier
```

---

## Module Responsibilities

### `mrag/hash/ngram_hasher.py`

The one decision that needs locking down before implementation: **what level of N-gram?**

| Level | Pros | Cons |
|---|---|---|
| Token-level N=3 | Matches the model's vocabulary; collision-resistant | Requires tokenizer at hash time |
| BPE unigrams | Fastest; no tokenizer needed | High collision rate on short prompts |
| Character trigrams | Portable; no tokenizer dep | ~3x more entries per prompt |

**Recommended starting point:** token-level N=3 using the same tokenizer as the base model. Lock `N` as a config constant, not a runtime parameter — it determines table structure and can't change after tables are built.

```python
class NgramHasher:
    def __init__(self, n: int = 3, seed: int = 42):
        self.n = n
        self.seed = seed

    def hash_context(self, token_ids: list[int]) -> list[str]:
        """Returns list of hex digests, one per N-gram."""
        ...

    def lookup_key(self, token_ids: list[int]) -> str:
        """Returns the single best lookup key for this context."""
        # Aggregate N-gram hashes → single canonical key
        # Option A: XOR-fold all hashes → fast but lossy
        # Option B: use the highest-frequency N-gram → semantically stable
        ...
```

---

### `mrag/store/manager.py`

Handles the mount/unmount lifecycle so only the active NPC's tables occupy RAM.

```python
class EngRamManager:
    """
    Keeps at most `max_loaded` adapter tables in system RAM at once.
    Evicts LRU table on overflow. Tables backed by SQLite on NVMe.
    """
    def __init__(self, tables_dir: str, max_loaded: int = 3):
        ...

    def mount(self, adapter_name: str) -> EngRamTable:
        """Load adapter table into RAM, evict LRU if needed."""
        ...

    def unmount(self, adapter_name: str) -> None:
        """Serialize current state back to SQLite and free RAM."""
        ...

    def write_memory(self, adapter_name: str, payload: EngRamPayload) -> None:
        """Upsert a memory into the named table (auto-mounts if needed)."""
        ...
```

---

### `mrag/router/affect_router.py`

Converts the `affect_mean` from retrieved memories into an adapter label. This is a **lookup table, not a neural network** — the base model stays frozen.

```python
# Affect bands → adapter family mapping
# These map directly to .mrna adapter filenames
AFFECT_BANDS = [
    (+0.6,  1.0,  "warm"),       # nostalgic, friendly, generous
    (+0.2,  0.6,  "cordial"),    # neutral-positive, professional
    (-0.2,  0.2,  "neutral"),    # no strong valence
    (-0.6, -0.2,  "guarded"),    # wary, clipped, transactional
    (-1.0, -0.6,  "hostile"),    # betrayal, anger, threatening
]

class AffectRouter:
    def route(self, affect: float, adapter_hint: str) -> str:
        """
        Returns a composite adapter label:
        e.g. affect=+0.7, hint="blacksmith" → "blacksmith_warm"
        """
        band = self._band_for(affect)
        return f"{adapter_hint}_{band}"
```

---

## Test Harness

Tests run against **mocked BridgePackets only** — no external model dependencies, no GPU.

### `tests/fixtures/mock_packets.json`

```json
[
  {
    "id": "trading_positive",
    "trigger": {
      "adapter_hint": "blacksmith",
      "context_hash": "a3f9b2c1",
      "prompt_preview": "Player: I want to buy an iron sword."
    },
    "memories": [
      { "text": "Player bought a sword last session.",
        "salience": 0.9, "affect": 0.8, "source": "blacksmith", "age": 2 }
    ],
    "expected_adapter": "blacksmith_warm"
  },
  {
    "id": "combat_negative",
    "trigger": {
      "adapter_hint": "guard",
      "context_hash": "f1e4d2a8",
      "prompt_preview": "Player attacked the market stall."
    },
    "memories": [
      { "text": "Player attacked the market stall yesterday.",
        "salience": 0.95, "affect": -0.85, "source": "guard", "age": 1 }
    ],
    "expected_adapter": "guard_hostile"
  },
  {
    "id": "decay_eviction",
    "trigger": {
      "adapter_hint": "merchant",
      "context_hash": "b8c3a1d5",
      "prompt_preview": "Player: Do you have any potions?"
    },
    "memories": [
      { "text": "Player haggled poorly three weeks ago.",
        "salience": 0.3, "affect": -0.3, "source": "merchant", "age": 40 }
    ],
    "expected_adapter": "merchant_neutral",
    "expected_evicted": 1
  }
]
```

### What each test validates

| Test | Validates | Pass condition |
|---|---|---|
| `test_route_accuracy` | `affect + salience → adapter_label` | 100% match on all fixtures |
| `test_decay` | Low-salience memories evict over time | `is_evictable()` fires at correct age |
| `test_latency` | End-to-end hash + lookup + route | < 500ms on CPU RAM lookup |

---

## Implementation Sequence (Pre-Mac)

Since you're waiting on hardware, this order lets you build and validate everything
without needing GPU or local model access:

1. **`schema/`** — Lock `EngRamPayload` and `BridgePacket` types. These are the
   contract. Everything else builds around them.

2. **`hash/ngram_hasher.py`** — Decide on token-level N=3. Write the hasher.
   Validate collision rate on a sample of NPC dialogue strings.

3. **`store/`** — Build `EngRamTable` (in-memory dict) + `sqlite_backend`
   (persistence). Build `manager.py` LRU mount/unmount logic.

4. **`router/affect_router.py`** — Hardcode the affect bands. Wire to table lookup.

5. **`tests/`** — Feed mock packets through the full pipeline. All three test
   categories should pass before wiring to a live model or inference backend.

6. **`prefetch/coordinator.py`** — Last, because it depends on mRNA's actual
   layer-timing signals. Mock it in tests; wire the real PCIe prefetch
   once the Mac arrives and you can run llama-server.

---

## Notes on Future Rust Extraction

The hot path (hash lookup + RAM fetch) is the only thing that needs Rust later.
Keep `NgramHasher` and `EngRamTable` as pure functions with no side effects so
they can be re-implemented as a PyO3 extension without changing any call sites.

Everything else (manager, router, bridge I/O) can stay Python indefinitely —
it runs once per inference, not per token.
