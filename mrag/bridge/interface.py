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

from typing import TYPE_CHECKING, Callable, Optional

from mrag.schema.bridge import ContextTrigger, EngRamResponse, PidxSyncPacket
from mrag.store.manager import EngRamManager
from mrag.router.affect_router import AffectRouter
from mrag.router.decay import convert_pidx_delta
from mrag.hash.ngram_hasher import NgramHasher

if TYPE_CHECKING:
    from mrag.router.decay import DecayPolicy

# Weight of trigram similarity vs salience in miss-path re-ranking.
# 0.0 = pure salience, 1.0 = pure similarity.
SIM_WEIGHT = 0.6


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
    decay_policy : DecayPolicy, optional
        Custom DecayPolicy for tables.
    affect_router : AffectRouter, optional
        Custom AffectRouter instance.
    """

    def __init__(
        self,
        tables_dir:     str,
        max_loaded:     int = 3,
        top_n:          int = 5,
        prefetch_hint:  Optional[Callable[[str], None]] = None,
        decay_policy:   Optional[DecayPolicy] = None,
        affect_router:  Optional[AffectRouter] = None,
    ) -> None:
        self._manager       = EngRamManager(tables_dir, max_loaded, decay_policy=decay_policy)
        self._router        = affect_router or AffectRouter()
        self._hasher        = NgramHasher()
        self._top_n         = top_n
        self._prefetch_hint = prefetch_hint

    # ── Core path: mRNA Layer 14 → mRAG → mRNA Layer 25 ─────────────────────

    def handle_trigger(self, trigger: ContextTrigger,
                        query_text: str = "") -> EngRamResponse:
        """
        Main entry point called by mRNA after Layer 14 SAE fires.

        1. Mount the adapter table for trigger.adapter_hint.
        2. Evict any stale entries discovered before pulling top memories.
        3. Look up trigger.context_hash in the table.
        4. Derive adapter_label, salience_max, affect_mean from top-N memories.
        5. Return EngRamResponse for mRNA Layer 25.

        When query_text is provided and the hash path misses, results are
        re-ranked using character-trigram similarity to the query.
        """
        adapter_hint = trigger.adapter_hint
        table = self._manager.mount(adapter_hint)

        # Frontload eviction to ensure we operate on clean, valid table data (O(1) in subsequent hits)
        evicted = table.evict_stale()

        # Direct hash lookup first — O(1), the hot path
        hit = table.get(trigger.context_hash)
        
        # Retrieve top memories exactly once to avoid redundant sorting/heap operations
        top = table.top_by_salience(self._top_n)

        if not top:
            label, salience_max, affect_mean = f"{adapter_hint}_neutral", 0.0, 0.0
        else:
            salience_max = max(p.decayed_salience() for p in top)
            affect_mean  = sum(p.affect for p in top) / len(top)
            label        = self._router.route(affect_mean, adapter_hint)

        if hit is not None:
            # Since eviction ran first, if hit is present, it is guaranteed live and valid
            if self._prefetch_hint:
                self._prefetch_hint(label)
            return EngRamResponse(
                adapter_label=label,
                memory_tokens=self._collect_tokens(table, hit=hit, top=top),
                salience_max=salience_max,
                affect_mean=affect_mean,
                evicted_count=evicted,
            )

        # Miss path
        if self._prefetch_hint:
            self._prefetch_hint(label)
        return EngRamResponse(
            adapter_label=label,
            memory_tokens=self._collect_tokens(table, query_text=query_text, top=top),
            salience_max=salience_max,
            affect_mean=affect_mean,
            evicted_count=evicted,
        )

    def _collect_tokens(self, table, hit=None, query_text: str = "", top: Optional[list[EngRamPayload]] = None) -> list[str]:
        """Return memory text strings for the top-N entries.

        With hash hit: promotes the hit to position 0, fills rest by salience.
        With query_text (hash miss): re-ranks by combined score
        (SIM_WEIGHT * trigram_jaccard + (1-SIM_WEIGHT) * norm_salience).
        Without either: pure top-by-salience.
        """
        if top is None:
            top = table.top_by_salience(self._top_n)
            
        if hit is not None:
            # Promote the hash-matched entry to first position
            tokens = [hit.text]
            seen = {id(hit)}
            for p in top:
                if len(tokens) >= self._top_n:
                    break
                if id(p) not in seen:
                    tokens.append(p.text)
                    seen.add(id(p))
            return tokens

        if query_text and top:
            # Re-rank by combined similarity + salience score
            max_sal = max(p.decayed_salience() for p in top)
            norm = lambda s: s / max_sal if max_sal > 0 else 0.0
            scored = []
            for p in top:
                sim = self._hasher.text_similarity(query_text, p.text)
                score = SIM_WEIGHT * sim + (1.0 - SIM_WEIGHT) * norm(p.decayed_salience())
                scored.append((score, p.text))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [text for _, text in scored[:self._top_n]]

        return [p.text for p in top]

    def _collect_payloads(self, table, hit=None,
                          max_per_table: int = 3,
                          top: Optional[list[EngRamPayload]] = None) -> list:
        """Return top payloads from a table, with hash-hit promotion."""
        if top is None:
            top = table.top_by_salience(max_per_table)
        if hit is None:
            return list(top)
        # Promote the hash-hit payload to first position
        result = [hit]
        seen = {id(hit)}
        for p in top:
            if id(p) not in seen:
                result.append(p)
                seen.add(id(p))
        return result[:max_per_table]

    # ── Cross-table query ─────────────────────────────────────────────────────

    def handle_cross_trigger(self, adapter_hints: list[str],
                             context_hash: str,
                             query_text: str = "") -> EngRamResponse:
        """Query multiple adapters and merge by salience + similarity.

        Pulls top entries from each adapter, deduplicates by text content,
        and returns the top-N across all queried tables. Hash hits receive a
        small salience boost (+0.02). When query_text is provided, the final
        ranking uses a combined score of trigram similarity and salience.
        """
        HIT_BOOST = 0.02
        per_table = max(3, self._top_n)
        evicted_total = 0
        all_payloads: list = []  # (payload, adapter_hint, is_hit)

        for hint in adapter_hints:
            table = self._manager.mount(hint)
            evicted_total += table.evict_stale()
            
            hit = table.get(context_hash)
            top = table.top_by_salience(per_table)
            
            payloads = self._collect_payloads(table, hit=hit,
                                               max_per_table=per_table,
                                               top=top)
            is_hit = hit is not None
            for p in payloads:
                all_payloads.append((p, hint, is_hit and p is hit))

        if not all_payloads:
            return EngRamResponse(
                adapter_label="cross_neutral",
                memory_tokens=[],
                salience_max=0.0,
                affect_mean=0.0,
                evicted_count=evicted_total,
            )

        # Compute combined scores for ranking
        max_sal = max(p.decayed_salience() for p, _, _ in all_payloads)
        norm = lambda s: s / max_sal if max_sal > 0 else 0.0
        scored: list = []  # (score, payload, adapter_hint)
        seen_texts: set = set()

        for p, hint, is_hit in all_payloads:
            text_key = p.text.strip().lower()
            if text_key in seen_texts:
                continue  # dedup — first encountered wins (highest salience per table)
            seen_texts.add(text_key)

            sim = self._hasher.text_similarity(query_text, p.text) if query_text else 0.0
            score = SIM_WEIGHT * sim + (1.0 - SIM_WEIGHT) * norm(p.decayed_salience())
            if is_hit:
                score += HIT_BOOST
            scored.append((score, p, hint))

        ranked = sorted(scored, key=lambda x: x[0], reverse=True)[:self._top_n]

        salience_max = max(p.decayed_salience() for _, p, _ in ranked) if ranked else 0.0
        affect_mean  = sum(p.affect for _, p, _ in ranked) / len(ranked) if ranked else 0.0
        label        = self._router.route(affect_mean, adapter_hint="cross")
        tokens       = [p.text for _, p, _ in ranked]

        return EngRamResponse(
            adapter_label=label,
            memory_tokens=tokens,
            salience_max=salience_max,
            affect_mean=affect_mean,
            evicted_count=evicted_total,
        )

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
            mutated = False
            for key, payload in table.items():
                for tag in payload.tags:
                    if tag in packet.salience_boost:
                        payload.salience = min(
                            1.0,
                            payload.salience + packet.salience_boost[tag]
                        )
                        mutated = True
                        break  # one boost per memory per sync
            if mutated:
                table._needs_eviction = True

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
