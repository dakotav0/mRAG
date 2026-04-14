"""
test_latency — End-to-end < 500ms budget on RTX 4070 Super config.

Measures the full pipeline: ContextTrigger JSON → BridgeInterface →
EngRamResponse JSON. No GPU, no llama.cpp — CPU RAM lookup only.
In practice these run sub-millisecond; the 500ms budget is the spec contract
for when the coordinator is wired into the real prefetch path.

Each test warms the path once before measuring to avoid first-call
import overhead skewing the result.
"""

import json
import time
from statistics import mean, median

import pytest

from mrag.bridge.interface import BridgeInterface
from mrag.hash.ngram_hasher import NgramHasher, NGRAM_N
from mrag.schema.bridge import ContextTrigger
from mrag.schema.payload import EngRamPayload
from mrag.router.affect_router import AffectRouter
from mrag.store.engram_table import EngRamTable

BUDGET_MS = 500.0   # spec contract
REPEAT    = 20      # samples for statistical stability


def _ms(start: float, end: float) -> float:
    return (end - start) * 1000.0


class TestHasherLatency:

    def test_hash_context_under_budget(self):
        """NgramHasher.hash_context() on a 100-token sequence is well under budget."""
        h = NgramHasher()
        ids = list(range(100))
        h.hash_context(ids)  # warm

        times = []
        for _ in range(REPEAT):
            t0 = time.perf_counter()
            h.hash_context(ids)
            times.append(_ms(t0, time.perf_counter()))

        assert median(times) < BUDGET_MS, \
            f"hash_context median={median(times):.2f}ms exceeds {BUDGET_MS}ms"

    def test_lookup_key_under_budget(self):
        """NgramHasher.lookup_key() on a 50-token sequence is well under budget."""
        h = NgramHasher()
        ids = list(range(50))
        h.lookup_key(ids)  # warm

        times = []
        for _ in range(REPEAT):
            t0 = time.perf_counter()
            h.lookup_key(ids)
            times.append(_ms(t0, time.perf_counter()))

        assert median(times) < BUDGET_MS, \
            f"lookup_key median={median(times):.2f}ms exceeds {BUDGET_MS}ms"


class TestRouterLatency:

    def test_affect_router_under_budget(self):
        """AffectRouter.route_from_table() on a 100-entry table is under budget."""
        r = AffectRouter()
        t = EngRamTable("perf_test")
        for i in range(100):
            t.put(f"key{i}", EngRamPayload(f"mem{i}", 0.5 + i * 0.004,
                                            0.0, "perf_test", age=i % 20))
        r.route_from_table("perf_test", t)  # warm

        times = []
        for _ in range(REPEAT):
            t0 = time.perf_counter()
            r.route_from_table("perf_test", t)
            times.append(_ms(t0, time.perf_counter()))

        assert median(times) < BUDGET_MS, \
            f"route_from_table median={median(times):.2f}ms exceeds {BUDGET_MS}ms"


class TestEndToEndLatency:

    def test_handle_trigger_under_budget(self, populated_bridge, mock_packets):
        """
        Full pipeline: ContextTrigger JSON in → EngRamResponse JSON out.
        Median across all fixtures × REPEAT iterations must be < 500ms.
        """
        triggers = [
            json.dumps(p["trigger"]) for p in mock_packets
        ]
        # Warm all fixtures
        for tj in triggers:
            populated_bridge.handle_trigger_json(tj)

        times = []
        for _ in range(REPEAT):
            for tj in triggers:
                t0 = time.perf_counter()
                populated_bridge.handle_trigger_json(tj)
                times.append(_ms(t0, time.perf_counter()))

        p50 = median(times)
        p_max = max(times)
        assert p50 < BUDGET_MS, \
            f"handle_trigger p50={p50:.3f}ms exceeds {BUDGET_MS}ms"
        # Log for visibility — not a hard failure
        print(f"\n  handle_trigger: p50={p50:.3f}ms  max={p_max:.3f}ms  "
              f"n={len(times)}")

    def test_cold_mount_under_budget(self):
        """
        Even a cold table mount (SQLite load → RAM → lookup) is under budget.
        Uses an in-memory SQLite so NVMe latency is excluded from this test.
        """
        b = BridgeInterface(":memory:")
        b._manager.write_memory(
            "cold_npc", "cold_hash",
            EngRamPayload("Cold memory.", 0.8, 0.4, "cold_npc", age=1),
        )
        # Force unmount so the next call re-mounts from SQLite
        b._manager.unmount("cold_npc")

        trigger_json = json.dumps({
            "adapter_hint":   "cold_npc",
            "context_hash":   "cold_hash",
            "prompt_preview": "test",
        })

        # Warm
        b.handle_trigger_json(trigger_json)
        b._manager.unmount("cold_npc")

        times = []
        for _ in range(REPEAT):
            b._manager.unmount("cold_npc")
            t0 = time.perf_counter()
            b.handle_trigger_json(trigger_json)
            times.append(_ms(t0, time.perf_counter()))

        b.shutdown()
        assert median(times) < BUDGET_MS, \
            f"cold mount p50={median(times):.3f}ms exceeds {BUDGET_MS}ms"
