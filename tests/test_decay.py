"""
test_decay — Do low-salience memories drop out over time steps?

Pass condition: is_evictable() fires at the correct age for each fixture,
and EngRamTable.evict_stale() removes exactly the expected entries.
"""

import math
import pytest

from mrag.schema.payload import EngRamPayload
from mrag.store.engram_table import EngRamTable
from mrag.store.manager import EngRamManager
from mrag.router.decay import (
    decayed_salience,
    is_evictable,
    steps_until_eviction,
    convert_pidx_delta,
    DEFAULT_DECAY_RATE,
    EVICTION_THRESHOLD,
)


class TestDecayMath:

    def test_decayed_salience_at_zero_age(self):
        """No decay at age 0."""
        assert decayed_salience(0.9, 0) == pytest.approx(0.9)
        assert decayed_salience(0.3, 0) == pytest.approx(0.3)

    def test_decayed_salience_formula(self):
        """salience × (1 − rate)^age matches closed-form."""
        assert decayed_salience(0.9, 2) == pytest.approx(0.9 * 0.95**2)
        assert decayed_salience(0.3, 40) == pytest.approx(0.3 * 0.95**40)

    def test_trading_positive_survives(self, mock_packets):
        """salience=0.9, age=2 — should not be evictable (trading_positive fixture)."""
        packet = next(p for p in mock_packets if p["id"] == "trading_positive")
        mem = packet["memories"][0]
        assert not is_evictable(mem["salience"], mem["age"])

    def test_combat_negative_survives(self, mock_packets):
        """salience=0.95, age=1 — should not be evictable (combat_negative fixture)."""
        packet = next(p for p in mock_packets if p["id"] == "combat_negative")
        mem = packet["memories"][0]
        assert not is_evictable(mem["salience"], mem["age"])

    def test_decay_eviction_fixture_is_evictable(self, mock_packets):
        """salience=0.3, age=40 — must be evictable (decay_eviction fixture)."""
        packet = next(p for p in mock_packets if p["id"] == "decay_eviction")
        mem = packet["memories"][0]
        assert is_evictable(mem["salience"], mem["age"])
        # Verify the math: 0.3 × 0.95^40 ≈ 0.040 < 0.15
        assert decayed_salience(mem["salience"], mem["age"]) == pytest.approx(
            0.3 * 0.95**40, rel=1e-9
        )

    def test_steps_until_eviction_boundary(self):
        """At exactly n steps: evictable. At n-1: not evictable."""
        for salience in [0.9, 0.5, 0.3, 0.2]:
            n = steps_until_eviction(salience)
            assert is_evictable(salience, n), \
                f"salience={salience}: should be evictable at step {n}"
            if n > 0:
                assert not is_evictable(salience, n - 1), \
                    f"salience={salience}: should NOT be evictable at step {n-1}"

    def test_already_below_threshold(self):
        """salience already below threshold → evictable at age 0, steps=0."""
        assert is_evictable(0.1, 0)
        assert steps_until_eviction(0.1) == 0
        assert steps_until_eviction(0.0) == 0

    def test_payload_decayed_salience_matches_standalone(self):
        """EngRamPayload.decayed_salience() == decay.decayed_salience()."""
        p = EngRamPayload("x", salience=0.7, affect=0.0, source="test", age=15)
        assert p.decayed_salience() == pytest.approx(decayed_salience(0.7, 15))

    def test_payload_is_evictable_matches_standalone(self):
        """EngRamPayload.is_evictable() == decay.is_evictable()."""
        for sal, age in [(0.9, 2), (0.3, 40), (0.5, 30)]:
            p = EngRamPayload("x", salience=sal, affect=0.0, source="test", age=age)
            assert p.is_evictable() == is_evictable(sal, age)


class TestEngRamTableDecay:

    def test_tick_increments_age(self):
        """tick(n) adds n to every entry's age."""
        t = EngRamTable("blacksmith")
        p = EngRamPayload("x", 0.9, 0.5, "blacksmith", age=0)
        t.put("k1", p)
        t.tick(10)
        assert p.age == 10
        t.tick(5)
        assert p.age == 15

    def test_evict_stale_removes_only_below_threshold(self):
        """Only sub-threshold entries are removed; live ones survive."""
        t = EngRamTable("test")
        live = EngRamPayload("live",  salience=0.9, affect=0.0, source="test", age=0)
        dead = EngRamPayload("dying", salience=0.3, affect=0.0, source="test", age=0)
        t.put("live", live)
        t.put("dead", dead)

        t.tick(20)  # live: 0.9×0.95^20≈0.322 > 0.15; dead: 0.3×0.95^20≈0.107 < 0.15
        evicted = t.evict_stale()

        assert evicted == 1
        assert t.get("live") is not None
        assert t.get("dead") is None

    def test_evict_stale_returns_correct_count(self):
        """evict_stale() count matches the number of removed entries."""
        t = EngRamTable("test")
        for i in range(5):
            salience = 0.1 if i < 3 else 0.9  # first 3 already below threshold
            t.put(f"k{i}", EngRamPayload("x", salience, 0.0, "test", age=0))

        evicted = t.evict_stale()
        assert evicted == 3
        assert len(t) == 2

    def test_evict_stale_on_empty_table(self):
        """Evicting an empty table returns 0 and doesn't error."""
        t = EngRamTable("empty")
        assert t.evict_stale() == 0

    def test_decay_eviction_fixture_full_pipeline(self, mock_packets):
        """
        decay_eviction fixture end-to-end through EngRamTable:
        memory inserted at age=40, evict_stale() removes it, count=1.
        """
        packet = next(p for p in mock_packets if p["id"] == "decay_eviction")
        mem = packet["memories"][0]

        t = EngRamTable(packet["trigger"]["adapter_hint"])
        t.put(
            packet["trigger"]["context_hash"],
            EngRamPayload(
                text=mem["text"],
                salience=mem["salience"],
                affect=mem["affect"],
                source=mem["source"],
                age=mem["age"],
            ),
        )
        evicted = t.evict_stale()
        assert evicted == packet.get("expected_evicted", 0)
        assert len(t) == 0


class TestEngRamManagerDecay:

    def test_tick_adapter_via_manager(self):
        """tick_adapter() ages entries in the named table."""
        mgr = EngRamManager(":memory:")
        mgr.write_memory("bs", "k1", EngRamPayload("x", 0.9, 0.5, "bs", age=0))
        mgr.tick_adapter("bs", steps=10)
        p = mgr.get_memory("bs", "k1")
        assert p.age == 10

    def test_pidx_decay_delta_conversion(self):
        """convert_pidx_delta maps PIDX days to mRAG steps (1:1 at equal λ/rate)."""
        assert convert_pidx_delta(5.0) == 5
        assert convert_pidx_delta(0.0) == 0
        assert convert_pidx_delta(-2.0) == 0   # boost direction → no aging

    def test_pidx_sync_applies_decay_and_boost(self, bridge):
        """PidxSyncPacket ticks age and applies salience boost via bridge."""
        import json
        bridge._manager.write_memory(
            "blacksmith", "craft_key",
            EngRamPayload("Player commissioned a shield.", 0.8, 0.5,
                          "blacksmith", age=0, tags=["craft"]),
        )
        sync_json = json.dumps({
            "npc_id":       "blacksmith_01",
            "adapter_name": "blacksmith",
            "decay_delta":  3.0,
            "salience_boost": {"craft": 0.1},
        })
        bridge.handle_pidx_sync_json(sync_json)

        p = bridge._manager.get_memory("blacksmith", "craft_key")
        assert p.age == 3
        assert p.salience == pytest.approx(0.9)   # 0.8 + 0.1 boost

    def test_salience_boost_capped_at_one(self, bridge):
        """Salience boost never exceeds 1.0."""
        import json
        bridge._manager.write_memory(
            "guard", "k",
            EngRamPayload("x", 0.95, 0.0, "guard", age=0, tags=["alert"]),
        )
        sync_json = json.dumps({
            "npc_id": "g1", "adapter_name": "guard",
            "decay_delta": 0.0, "salience_boost": {"alert": 0.5},
        })
        bridge.handle_pidx_sync_json(sync_json)
        p = bridge._manager.get_memory("guard", "k")
        assert p.salience <= 1.0
