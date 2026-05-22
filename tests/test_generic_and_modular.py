"""
Unit tests for decoupled, modular, and generic features of mRAG.

Tests:
1. Generic Engram payloads (storing non-string dict values).
2. Linear decay policy mathematical boundaries.
3. Custom affect router bands and formatting callbacks.
4. End-to-end BridgeInterface integration with customized decay and router.
"""

from typing import Dict

from mrag.schema.payload import Engram, EngRamPayload
from mrag.router.decay import LinearDecay, ExponentialDecay, NoDecay
from mrag.router.affect_router import AffectRouter, _Band
from mrag.bridge.interface import BridgeInterface
from mrag.schema.bridge import ContextTrigger


def test_generic_engram_value():
    """Verify that the generic Engram[T] can store arbitrary value types."""
    # Storing a dictionary mapping concept names to importance scores
    payload_data: Dict[str, int] = {"armor": 80, "weapon": 95, "potion": 30}
    engram = Engram[Dict[str, int]](
        value=payload_data,
        salience=0.85,
        affect=0.4,
        source="blacksmith_inventory",
        tags=["inventory", "stats"],
        metadata={"npc_id": "blacksmith_01"}
    )

    assert engram.value == payload_data
    assert engram.value["weapon"] == 95
    assert engram.salience == 0.85
    assert engram.affect == 0.4
    assert engram.source == "blacksmith_inventory"
    assert "stats" in engram.tags
    assert engram.metadata["npc_id"] == "blacksmith_01"


def test_linear_decay_boundaries():
    """Verify mathematical boundaries for the linear decay policy."""
    # Use 0.0625 (1/16) which is exactly representable in binary float
    policy = LinearDecay(rate=0.0625)

    # Initial decayed salience at age=0 is unchanged
    assert policy.decayed_salience(0.5, age=0) == 0.5

    # Linear subtraction over ages
    assert policy.decayed_salience(0.5, age=2) == 0.375
    assert policy.decayed_salience(0.5, age=4) == 0.25
    assert policy.decayed_salience(0.5, age=8) == 0.0  # clamps to 0.0
    assert policy.decayed_salience(0.5, age=10) == 0.0

    # Eviction boundary: threshold = 0.15
    # At age=5, salience = 0.5 - 5 * 0.0625 = 0.1875 (>= 0.15, NOT evictable yet)
    assert not policy.is_evictable(0.5, age=5, threshold=0.15)

    # At age=6, salience = 0.5 - 6 * 0.0625 = 0.125 (< 0.15, so evictable)
    assert policy.is_evictable(0.5, age=6, threshold=0.15)

    # steps_until_eviction calculation: math.ceil((0.5 - 0.15) / 0.0625) = math.ceil(5.6) = 6 steps
    assert policy.steps_until_eviction(0.5, threshold=0.15) == 6


def test_custom_affect_router():
    """Verify custom affect router formatting and boundary selection."""
    custom_bands = [
        _Band(low=0.1, high=1.01, name="pleased"),
        _Band(low=-0.1, high=0.1, name="indifferent"),
        _Band(low=-1.01, high=-0.1, name="displeased")
    ]
    custom_format = lambda hint, band: f"adapter::{hint}::affect::{band}"

    router = AffectRouter(bands=custom_bands, format_fn=custom_format)

    assert router.route(0.5, "alchemist") == "adapter::alchemist::affect::pleased"
    assert router.route(0.0, "alchemist") == "adapter::alchemist::affect::indifferent"
    assert router.route(-0.8, "alchemist") == "adapter::alchemist::affect::displeased"


def test_end_to_end_modular_bridge():
    """Verify end-to-end memory retrieval, eviction, and routing using custom DI policy and router."""
    # 1. Setup custom policy (linear decay with rate 0.1) and custom router
    custom_policy = LinearDecay(rate=0.1)
    custom_bands = [
        _Band(0.0, 1.01, "positive"),
        _Band(-1.01, 0.0, "negative")
    ]
    custom_router = AffectRouter(bands=custom_bands, format_fn=lambda h, b: f"{h}::{b}")

    # 2. Instantiate bridge interface with injected components
    bridge = BridgeInterface(
        tables_dir=":memory:",
        decay_policy=custom_policy,
        affect_router=custom_router,
        top_n=3
    )

    # 3. Pre-populate custom memories
    # Memory A: highly salience, positive affect, age = 0
    bridge._manager.write_memory(
        "healer", "k1",
        EngRamPayload("Healed successfully.", salience=0.9, affect=0.8, source="healer")
    )
    # Memory B: lower salience, negative affect, age = 5 (will decay to 0.8 - 0.5 = 0.3)
    bridge._manager.write_memory(
        "healer", "k2",
        EngRamPayload("Patient died.", salience=0.8, affect=-0.7, source="healer", age=5)
    )
    # Memory C: low salience, will be evictable under 0.1-rate linear decay at age=3
    # Initial salience = 0.4. Age = 3 -> Decayed = 0.4 - 0.3 = 0.1 < 0.15 (evictable)
    bridge._manager.write_memory(
        "healer", "k3",
        EngRamPayload("Dropped potion bottle.", salience=0.4, affect=-0.2, source="healer", age=3)
    )

    # 4. Trigger lookups and verify eviction and customized routing formats
    trigger = ContextTrigger(
        adapter_hint="healer",
        context_hash="k1",
        prompt_preview="Requesting healing."
    )

    # Trigger handling should:
    # - Run eviction using LinearDecay: Memory C (k3) gets evicted (evicted_count = 1).
    # - Retrieve Memory A (k1) and Memory B (k2).
    # - Average affect of remaining (0.8 + -0.7) / 2 = 0.05.
    # - Route affect 0.05 using custom router -> healer::positive.
    response = bridge.handle_trigger(trigger)

    assert response.adapter_label == "healer::positive"
    assert response.evicted_count == 1
    assert "Healed successfully." in response.memory_tokens
    assert "Patient died." in response.memory_tokens
    assert "Dropped potion bottle." not in response.memory_tokens

    bridge.shutdown()
