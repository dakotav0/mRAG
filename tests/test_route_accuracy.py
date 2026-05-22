"""
test_route_accuracy — Does salience + affect combo → correct adapter label?

Pass condition: 100% match on all mock_packets.json fixtures.
No external model dependencies, no GPU required.
"""

import json
import pytest

from mrag.bridge.interface import BridgeInterface
from mrag.schema.bridge import ContextTrigger
from mrag.schema.payload import EngRamPayload


def _trigger_from(packet: dict) -> ContextTrigger:
    return ContextTrigger(**packet["trigger"])


class TestRouteAccuracy:

    def test_all_fixtures_match_expected_adapter(self, populated_bridge, mock_packets):
        """100% adapter label accuracy across all fixtures."""
        for packet in mock_packets:
            trigger = _trigger_from(packet)
            response = populated_bridge.handle_trigger(trigger)
            assert response.adapter_label == packet["expected_adapter"], (
                f"[{packet['id']}] expected {packet['expected_adapter']!r}, "
                f"got {response.adapter_label!r}"
            )

    def test_trading_positive(self, populated_bridge, mock_packets):
        """High positive affect + high salience → warm adapter family."""
        packet = next(p for p in mock_packets if p["id"] == "trading_positive")
        response = populated_bridge.handle_trigger(_trigger_from(packet))

        assert response.adapter_label == "blacksmith_warm"
        assert response.salience_max > 0.0
        assert response.affect_mean > 0.0
        assert response.evicted_count == 0

    def test_combat_negative(self, populated_bridge, mock_packets):
        """High negative affect + high salience → hostile adapter family."""
        packet = next(p for p in mock_packets if p["id"] == "combat_negative")
        response = populated_bridge.handle_trigger(_trigger_from(packet))

        assert response.adapter_label == "guard_hostile"
        assert response.salience_max > 0.0
        assert response.affect_mean < -0.6
        assert response.evicted_count == 0

    def test_decay_eviction_returns_neutral(self, populated_bridge, mock_packets):
        """Sub-threshold memory is evicted; empty table → neutral fallback."""
        packet = next(p for p in mock_packets if p["id"] == "decay_eviction")
        response = populated_bridge.handle_trigger(_trigger_from(packet))

        assert response.adapter_label == "merchant_neutral"
        assert response.evicted_count == packet.get("expected_evicted", 0)
        # After eviction the table is empty — no memory tokens
        assert response.memory_tokens == []
        assert response.salience_max == 0.0

    def test_json_wire_format_round_trip(self, populated_bridge, mock_packets):
        """JSON in → JSON out preserves all fields."""
        for packet in mock_packets:
            trigger_json = json.dumps(packet["trigger"])
            resp_json = populated_bridge.handle_trigger_json(trigger_json)
            resp = json.loads(resp_json)

            assert "adapter_label" in resp
            assert "memory_tokens" in resp
            assert "salience_max" in resp
            assert "affect_mean" in resp
            assert "evicted_count" in resp
            assert 0.0 <= resp["salience_max"] <= 1.0
            assert -1.0 <= resp["affect_mean"] <= 1.0

    def test_cache_miss_returns_neutral(self, bridge):
        """A trigger with no matching memory in an empty table → neutral."""
        trigger = ContextTrigger(
            adapter_hint="unknown_npc",
            context_hash="deadbeef",
            prompt_preview="Hello?",
        )
        response = bridge.handle_trigger(trigger)
        assert response.adapter_label == "unknown_npc_neutral"
        assert response.salience_max == 0.0
        assert response.memory_tokens == []

    def test_prefetch_hook_called(self, mock_packets):
        """Prefetch hint callback fires once per handle_trigger call."""
        called_with = []
        b = BridgeInterface(":memory:", prefetch_hint=called_with.append)

        packet = next(p for p in mock_packets if p["id"] == "trading_positive")
        b._manager.write_memory(
            "blacksmith", "a3f9b2c1",
            EngRamPayload("Sword sale.", 0.9, 0.8, "blacksmith", age=2),
        )
        b.handle_trigger(_trigger_from(packet))
        b.shutdown()

        assert len(called_with) == 1
        assert called_with[0] == "blacksmith_warm"
