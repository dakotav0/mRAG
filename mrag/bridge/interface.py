"""
BridgeInterface — the only public surface of mRAG.

Receives a ContextTrigger JSON from mRNA Layer 14, pulls the relevant
EngRamPayload(s) from the mounted adapter table, and returns an
EngRamResponse JSON before Layer 25 fires.

Also handles PidxSyncPacket — PIDX pushes identity/decay updates here,
which are applied to the appropriate adapter table via EngRamManager.

Contract:
  - mRAG has zero knowledge of llama.cpp internals.
  - mRNA has zero knowledge of SQLite.
  - PIDX has zero knowledge of adapters.
  - All three communicate exclusively via BridgePacket JSON.

Prefetch coordinator is NOT wired here yet (Phase 6 — requires real
mRNA layer-timing signals). The interface accepts an optional
`prefetch_hint` callback so it can be slotted in without changing
call sites.
"""

from __future__ import annotations

from typing import Callable, Optional

from mrag.schema.bridge import ContextTrigger, EngRamResponse, PidxSyncPacket
from mrag.store.manager import EngRamManager
from mrag.router.affect_router import AffectRouter
from mrag.router.decay import convert_pidx_delta


class BridgeInterface:
    """
    Stateful broker between mRNA and PIDX.

    One instance per server process. The EngRamManager is shared so all
    adapters draw from the same RAM pool and LRU eviction works globally.

    Parameters
    ----------
    tables_dir : str
        Passed through to EngRamManager / SqliteBackend.
        Use ":memory:" for tests.
    max_loaded : int
        Max adapter tables in RAM simultaneously (default 3).
    top_n : int
        How many top-salience memories to include in each EngRamResponse.
    prefetch_hint : callable, optional
        Called with (adapter_name,) after routing to schedule an async
        prefetch. Slot in prefetch/coordinator.py here when ready.
    """

    def __init__(
        self,
        tables_dir:     str,
        max_loaded:     int = 3,
        top_n:          int = 5,
        prefetch_hint:  Optional[Callable[[str], None]] = None,
    ) -> None:
        self._manager       = EngRamManager(tables_dir, max_loaded)
        self._router        = AffectRouter()
        self._top_n         = top_n
        self._prefetch_hint = prefetch_hint

    # ── Core path: mRNA Layer 14 → mRAG → mRNA Layer 25 ─────────────────────

    def handle_trigger(self, trigger: ContextTrigger) -> EngRamResponse:
        """
        Main entry point called by mRNA after Layer 14 SAE fires.

        1. Mount the adapter table for trigger.adapter_hint.
        2. Look up trigger.context_hash in the table.
        3. Derive adapter_label, salience_max, affect_mean from top-N memories.
        4. Evict any stale entries discovered during this pass.
        5. Return EngRamResponse for mRNA Layer 25.
        """
        adapter_hint = trigger.adapter_hint
        table = self._manager.mount(adapter_hint)

        # Direct hash lookup first — O(1), the hot path
        hit = table.get(trigger.context_hash)
        if hit is not None and not hit.is_evictable():
            # Single-entry fast path: hash matched a live memory
            label, sal, aff = self._router.route_from_table(adapter_hint, table,
                                                             self._top_n)
            evicted = table.evict_stale()
            if self._prefetch_hint:
                self._prefetch_hint(label)
            return EngRamResponse(
                adapter_label=label,
                memory_tokens=self._collect_tokens(table),
                salience_max=sal,
                affect_mean=aff,
                evicted_count=evicted,
            )

        # Miss or stale hit — still route from whatever is in the table
        evicted = table.evict_stale()
        label, sal, aff = self._router.route_from_table(adapter_hint, table,
                                                         self._top_n)
        if self._prefetch_hint:
            self._prefetch_hint(label)
        return EngRamResponse(
            adapter_label=label,
            memory_tokens=self._collect_tokens(table),
            salience_max=sal,
            affect_mean=aff,
            evicted_count=evicted,
        )

    def _collect_tokens(self, table) -> list[str]:
        """Return memory text strings for the top-N salience entries."""
        return [p.text for p in table.top_by_salience(self._top_n)]

    # ── PIDX sync path ────────────────────────────────────────────────────────

    def handle_pidx_sync(self, packet: PidxSyncPacket) -> None:
        """
        Apply a PIDX decay tick and optional salience boost to an adapter table.

        decay_delta is in PIDX-native days; convert_pidx_delta() maps it to
        mRAG step counts. Salience boosts are applied by upserting entries with
        elevated salience — not by directly mutating the salience field, so the
        decay clock keeps running from the new baseline.
        """
        steps = convert_pidx_delta(packet.decay_delta)
        if steps > 0:
            self._manager.tick_adapter(packet.adapter_name, steps)

        if packet.salience_boost:
            table = self._manager.mount(packet.adapter_name)
            for key, payload in table.items():
                for tag in payload.tags:
                    if tag in packet.salience_boost:
                        payload.salience = min(
                            1.0,
                            payload.salience + packet.salience_boost[tag]
                        )
                        break  # one boost per memory per sync

    # ── JSON convenience wrappers (the actual wire format) ───────────────────

    def handle_trigger_json(self, raw: str) -> str:
        """Parse ContextTrigger JSON, process, return EngRamResponse JSON."""
        trigger = ContextTrigger.model_validate_json(raw)
        response = self.handle_trigger(trigger)
        return response.model_dump_json()

    def handle_pidx_sync_json(self, raw: str) -> None:
        """Parse PidxSyncPacket JSON and apply to the adapter table."""
        packet = PidxSyncPacket.model_validate_json(raw)
        self.handle_pidx_sync(packet)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Flush all mounted tables to SQLite. Call on clean process exit."""
        self._manager.unmount_all()


if __name__ == "__main__":
    import json
    from mrag.schema.payload import EngRamPayload

    bridge = BridgeInterface(":memory:")

    # Pre-populate blacksmith table with a live memory
    bridge._manager.write_memory(
        "blacksmith", "a3f9b2c1",
        EngRamPayload("Player bought a sword last session.",
                      salience=0.9, affect=0.8, source="blacksmith", age=2)
    )

    # ── trading_positive fixture ──────────────────────────────────────────────
    trigger_json = json.dumps({
        "adapter_hint":   "blacksmith",
        "context_hash":   "a3f9b2c1",
        "prompt_preview": "Player: I want to buy an iron sword.",
    })
    resp_json = bridge.handle_trigger_json(trigger_json)
    resp = json.loads(resp_json)
    assert resp["adapter_label"] == "blacksmith_warm", resp
    assert resp["salience_max"] > 0.0
    assert resp["evicted_count"] == 0

    # ── combat_negative fixture ───────────────────────────────────────────────
    bridge._manager.write_memory(
        "guard", "f1e4d2a8",
        EngRamPayload("Player attacked the market stall yesterday.",
                      salience=0.95, affect=-0.85, source="guard", age=1)
    )
    trigger_json2 = json.dumps({
        "adapter_hint":   "guard",
        "context_hash":   "f1e4d2a8",
        "prompt_preview": "Player attacked the market stall.",
    })
    resp2 = json.loads(bridge.handle_trigger_json(trigger_json2))
    assert resp2["adapter_label"] == "guard_hostile", resp2

    # ── decay_eviction fixture ────────────────────────────────────────────────
    bridge._manager.write_memory(
        "merchant", "b8c3a1d5",
        EngRamPayload("Player haggled poorly three weeks ago.",
                      salience=0.3, affect=-0.3, source="merchant", age=40)
    )
    trigger_json3 = json.dumps({
        "adapter_hint":   "merchant",
        "context_hash":   "b8c3a1d5",
        "prompt_preview": "Player: Do you have any potions?",
    })
    resp3 = json.loads(bridge.handle_trigger_json(trigger_json3))
    assert resp3["adapter_label"] == "merchant_neutral", resp3
    assert resp3["evicted_count"] == 1

    # ── PIDX sync: decay tick ─────────────────────────────────────────────────
    bridge._manager.write_memory(
        "blacksmith", "fresh_key",
        EngRamPayload("Player commissioned a shield.", salience=0.8,
                      affect=0.5, source="blacksmith", age=0, tags=["craft"])
    )
    sync_json = json.dumps({
        "npc_id":       "blacksmith_01",
        "adapter_name": "blacksmith",
        "decay_delta":  5.0,
        "salience_boost": {"craft": 0.1},
    })
    bridge.handle_pidx_sync_json(sync_json)
    p = bridge._manager.get_memory("blacksmith", "fresh_key")
    assert p.age == 5
    assert abs(p.salience - 0.9) < 1e-9   # 0.8 + 0.1 boost

    bridge.shutdown()
    print("interface.py OK")
