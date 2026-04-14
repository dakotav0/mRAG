"""
mRAG integration demo — full pipeline without GPU or Minecraft.

Simulates what mRNA does between Layer 14 and Layer 25:
  1. Tokenize a prompt (pseudo-tokenizer: word → stable int, no HF dep)
  2. Compute context_hash via NgramHasher
  3. Fire ContextTrigger at BridgeInterface
  4. Receive EngRamResponse (adapter_label + memory_tokens)
  5. Show the augmented prompt mRNA would forward to llama-server

When you wire real llama-server later, replace pseudo_tokenize() with
the actual HF tokenizer and feed real token IDs into NgramHasher.
The rest of the pipeline is identical.

Run:
    python scripts/demo_inference.py
    python scripts/demo_inference.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from mrag.bridge.interface import BridgeInterface
from mrag.hash.ngram_hasher import NgramHasher
from mrag.prefetch.coordinator import PrefetchCoordinator
from mrag.schema.bridge import ContextTrigger
from mrag.schema.payload import EngRamPayload

# ── Pseudo-tokenizer ──────────────────────────────────────────────────────────
# Maps each whitespace-split word to a stable integer in [0, 32000).
# Deterministic within a process (no random seed). Matches no real vocabulary —
# replace with HF tokenizer(prompt).input_ids when wiring real mRNA.

_VOCAB_SIZE = 32_000


def pseudo_tokenize(text: str) -> list[int]:
    """Word-level pseudo-tokenizer: stable int per word, BPE vocab range."""
    words = text.lower().split()
    # Use Python's built-in hash with a fixed seed via explicit formula
    # to avoid PYTHONHASHSEED instability across runs
    return [
        (sum(ord(c) * (31 ** i) for i, c in enumerate(w)) % _VOCAB_SIZE)
        for w in words
    ]


# ── NPC memory corpus ─────────────────────────────────────────────────────────
# Realistic memories for three NPCs. Each is stored under the context_hash
# that the corresponding trigger prompt produces — so lookups succeed.

class NPCMemory(NamedTuple):
    adapter:  str
    text:     str
    salience: float
    affect:   float
    age:      int
    tags:     list[str]


NPC_MEMORIES: dict[str, list[NPCMemory]] = {
    "blacksmith": [
        NPCMemory("blacksmith",
                  "Player commissioned a masterwork sword three sessions ago.",
                  salience=0.92, affect=0.75, age=3, tags=["craft", "commission"]),
        NPCMemory("blacksmith",
                  "Player paid in full without haggling — reliable customer.",
                  salience=0.85, affect=0.80, age=5, tags=["trade", "trust"]),
        NPCMemory("blacksmith",
                  "Player asked about the northern siege — blacksmith knows nothing.",
                  salience=0.40, affect=0.10, age=12, tags=["lore"]),
    ],
    "guard": [
        NPCMemory("guard",
                  "Player was caught loitering near the treasury at night.",
                  salience=0.95, affect=-0.70, age=2, tags=["crime", "alert"]),
        NPCMemory("guard",
                  "Player previously helped escort the trade caravan safely.",
                  salience=0.65, affect=0.45, age=20, tags=["reputation"]),
    ],
    "merchant": [
        NPCMemory("merchant",
                  "Player haggled aggressively and walked out without buying.",
                  salience=0.55, affect=-0.40, age=8, tags=["trade"]),
        NPCMemory("merchant",
                  "Player bought rare spices in bulk — good business.",
                  salience=0.72, affect=0.55, age=15, tags=["trade", "profit"]),
    ],
}

# ── Scenarios ─────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "id":           "sword_inquiry",
        "npc":          "blacksmith",
        "prompt":       "Player: I need a blade forged for the coming war. Can you help me?",
        "description":  "Player commissioning combat gear from a friendly blacksmith",
    },
    {
        "id":           "night_patrol",
        "npc":          "guard",
        "prompt":       "Player: I was just passing through. I meant no harm.",
        "description":  "Player caught near restricted area — guard remembers prior incident",
    },
    {
        "id":           "potion_request",
        "npc":          "merchant",
        "prompt":       "Player: Do you have healing potions? I'll pay fair price.",
        "description":  "Player buying from merchant with mixed trade history",
    },
    {
        "id":           "cold_npc",
        "npc":          "innkeeper",
        "prompt":       "Player: A room for the night, please.",
        "description":  "NPC with no prior history — cold table, neutral fallback",
    },
]


# ── Augmented prompt builder ──────────────────────────────────────────────────

def build_augmented_prompt(memory_tokens: list[str], adapter_label: str,
                            original_prompt: str) -> str:
    """
    Mirrors what mRNA Layer 25 would prepend before forwarding to llama-server.
    In real mRNA this is injected as soft tokens, not literal text.
    Shown here as text for readability.
    """
    lines = [f"[adapter: {adapter_label}]"]
    if memory_tokens:
        lines.append("### Memory Context")
        for tok in memory_tokens:
            lines.append(f"- {tok}")
        lines.append("")
    lines.append(original_prompt)
    return "\n".join(lines)


# ── Main demo ─────────────────────────────────────────────────────────────────

def run_demo(verbose: bool = False) -> None:
    hasher  = NgramHasher()
    manager_dir = ":memory:"

    # Set up bridge with prefetch coordinator
    bridge_inst = BridgeInterface(manager_dir, max_loaded=4)
    coordinator = PrefetchCoordinator(bridge_inst._manager, lookahead_ms=0)
    bridge_inst = BridgeInterface(
        manager_dir, max_loaded=4,
        prefetch_hint=coordinator.as_prefetch_hook()
    )

    # ── Populate memories into NPC tables ────────────────────────────────────
    # Each memory gets its own key (hash of its text) so all survive in the
    # table. The trigger's context_hash is a direct lookup for exact past
    # matches; route_from_table() scans ALL entries for the relationship history.
    print("Populating NPC memory tables...")
    for npc, memories in NPC_MEMORIES.items():
        for i, mem in enumerate(memories):
            # Stable per-memory key: hash of the memory text itself
            mem_key = hasher.lookup_key(pseudo_tokenize(mem.text))
            # Append index to guarantee uniqueness if two texts hash identically
            mem_key = f"{mem_key}_{i}"
            bridge_inst._manager.write_memory(
                npc, mem_key,
                EngRamPayload(
                    text=mem.text, salience=mem.salience, affect=mem.affect,
                    source=mem.adapter, age=mem.age, tags=mem.tags,
                ),
            )
        if verbose and memories:
            print(f"  {npc}: {len(memories)} memories in table")

    print(f"\n{'─'*62}")
    print(f"{'mRAG Integration Demo':^62}")
    print(f"{'─'*62}\n")

    total_ms = []

    for scenario in SCENARIOS:
        npc      = scenario["npc"]
        prompt   = scenario["prompt"]
        token_ids = pseudo_tokenize(prompt)
        ctx_hash  = hasher.lookup_key(token_ids)

        trigger = ContextTrigger(
            adapter_hint=npc,
            context_hash=ctx_hash,
            prompt_preview=prompt[:128],
        )

        t0 = time.perf_counter()
        response = bridge_inst.handle_trigger(trigger)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_ms.append(elapsed_ms)

        augmented = build_augmented_prompt(
            response.memory_tokens, response.adapter_label, prompt
        )

        print(f"Scenario: {scenario['id']}")
        print(f"  NPC:          {npc}")
        print(f"  Context hash: {ctx_hash[:12]}…")
        print(f"  Description:  {scenario['description']}")
        print(f"  → adapter:    {response.adapter_label}")
        print(f"  → salience:   {response.salience_max:.3f}")
        print(f"  → affect:     {response.affect_mean:+.3f}")
        print(f"  → evicted:    {response.evicted_count}")
        print(f"  → latency:    {elapsed_ms:.3f}ms")
        if response.memory_tokens:
            print(f"  → memories:   {len(response.memory_tokens)} injected")
        else:
            print(f"  → memories:   (none — cold table)")
        print()

        if verbose:
            print("  Augmented prompt (what llama-server receives):")
            for line in augmented.splitlines():
                print(f"    {line}")
            print()

    coordinator.wait_all(timeout=1.0)
    bridge_inst.shutdown()

    avg_ms = sum(total_ms) / len(total_ms)
    print(f"{'─'*62}")
    print(f"  {len(SCENARIOS)} scenarios  |  avg {avg_ms:.3f}ms  |  max {max(total_ms):.3f}ms")
    print(f"  All within 500ms budget: {all(ms < 500 for ms in total_ms)}")
    print(f"{'─'*62}")


# ── PIDX sync demo ────────────────────────────────────────────────────────────

def run_pidx_sync_demo(verbose: bool = False) -> None:
    """
    Shows how a PIDX decay tick changes routing after enough time steps.
    guard memory starts negative but not hostile; after aging it may cross bands.
    """
    print(f"\n{'─'*62}")
    print(f"{'PIDX Sync Demo — decay tick effect':^62}")
    print(f"{'─'*62}\n")

    hasher = NgramHasher()
    bridge = BridgeInterface(":memory:")

    prompt    = "Player: I was just passing through. I meant no harm."
    token_ids = pseudo_tokenize(prompt)
    ctx_hash  = hasher.lookup_key(token_ids)

    # Insert a guard memory that starts at affect=-0.4 (guarded band)
    bridge._manager.write_memory(
        "guard", ctx_hash,
        EngRamPayload("Player loitered near treasury at midnight.",
                      salience=0.9, affect=-0.4, source="guard", age=0,
                      tags=["alert"]),
    )

    trigger = ContextTrigger(adapter_hint="guard", context_hash=ctx_hash,
                              prompt_preview=prompt[:128])

    r1 = bridge.handle_trigger(trigger)
    print(f"  Before PIDX sync:  {r1.adapter_label}  (salience={r1.salience_max:.3f})")

    # PIDX sends a decay tick: 30 simulated days → 30 mRAG steps
    import json as _json
    bridge.handle_pidx_sync_json(_json.dumps({
        "npc_id": "guard_01", "adapter_name": "guard",
        "decay_delta": 30.0, "salience_boost": None,
    }))

    r2 = bridge.handle_trigger(trigger)
    print(f"  After 30-step tick: {r2.adapter_label}  "
          f"(salience={r2.salience_max:.3f}, evicted={r2.evicted_count})")

    bridge.shutdown()
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mRAG integration demo")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print augmented prompts and extra detail")
    args = parser.parse_args()

    run_demo(args.verbose)
    run_pidx_sync_demo(args.verbose)
