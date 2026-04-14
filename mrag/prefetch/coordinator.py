"""
PrefetchCoordinator — schedule EngRamTable loads before Layer 25 fires.

Phase 6: depends on mRNA's actual layer-timing signals (the gap between
Layer 14 routing and Layer 25 LoRA swap). This module is mocked in tests;
the real PCIe prefetch is wired once llama-server is running and the
layer-timing callback can be registered.

Design
------
When mRNA's Layer 14 SAE fires, there is a short window (the remaining
prefill tokens) before Layer 25 needs the LoRA weights and memory tokens.
The coordinator uses this window to mount the adapter table into RAM
asynchronously, so EngRamManager.mount() returns instantly when the
BridgeInterface calls it at Layer 25 time.

Integration contract
--------------------
1. mRNA registers a timing callback via `register_layer_hook()`.
2. The callback fires with (adapter_hint, context_hash) after Layer 14.
3. The coordinator calls EngRamManager.mount() in a background thread.
4. BridgeInterface.handle_trigger() arrives at Layer 25 — table already warm.

The coordinator is intentionally optional: BridgeInterface accepts it via
the `prefetch_hint` parameter. If not wired, handle_trigger() mounts
on-demand with no functional difference, only a latency cost.

Mocking in tests
----------------
Pass a lambda or any callable as `prefetch_hint` to BridgeInterface.
The coordinator itself is tested here with a fake sleep-based timing signal.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from mrag.store.manager import EngRamManager


class PrefetchCoordinator:
    """
    Schedules background EngRamTable mounts between Layer 14 and Layer 25.

    Parameters
    ----------
    manager : EngRamManager
        Shared manager instance (same one BridgeInterface uses).
    lookahead_ms : float
        How long to wait before triggering the prefetch after a hint arrives.
        Set to 0 for immediate (useful in tests). In production, tune to
        the observed Layer14→Layer25 gap on your llama-server config.
    """

    def __init__(self, manager: EngRamManager, lookahead_ms: float = 0.0) -> None:
        self._manager      = manager
        self._lookahead_s  = lookahead_ms / 1000.0
        self._lock         = threading.Lock()
        self._pending: set[str] = set()  # adapter names queued for prefetch

    # ── Public API ────────────────────────────────────────────────────────────

    def hint(self, adapter_label: str) -> None:
        """
        Called by BridgeInterface.prefetch_hint after routing.

        adapter_label is the composite label ("blacksmith_warm"); strip the
        band suffix to get the raw adapter name for table lookup.
        """
        # Strip affect band suffix: "blacksmith_warm" → "blacksmith"
        adapter_name = _strip_band(adapter_label)
        self._schedule(adapter_name)

    def hint_raw(self, adapter_name: str) -> None:
        """Called with a bare adapter name (no band suffix)."""
        self._schedule(adapter_name)

    def as_prefetch_hook(self) -> Callable[[str], None]:
        """
        Return a callable suitable for BridgeInterface(prefetch_hint=...).

        Usage:
            coordinator = PrefetchCoordinator(manager)
            bridge = BridgeInterface(":memory:", prefetch_hint=coordinator.as_prefetch_hook())
        """
        return self.hint

    # ── Internal ─────────────────────────────────────────────────────────────

    def _schedule(self, adapter_name: str) -> None:
        """Fire-and-forget background mount. Deduplicates in-flight requests."""
        with self._lock:
            if adapter_name in self._pending:
                return   # already queued — don't pile up threads
            self._pending.add(adapter_name)

        t = threading.Thread(
            target=self._prefetch_worker,
            args=(adapter_name,),
            daemon=True,
            name=f"mrag-prefetch-{adapter_name}",
        )
        t.start()

    def _prefetch_worker(self, adapter_name: str) -> None:
        try:
            if self._lookahead_s > 0:
                import time
                time.sleep(self._lookahead_s)
            # mount() is idempotent — safe to call even if already loaded
            self._manager.mount(adapter_name)
        finally:
            with self._lock:
                self._pending.discard(adapter_name)

    # ── Introspection (for tests and monitoring) ──────────────────────────────

    @property
    def pending(self) -> frozenset[str]:
        """Adapter names currently being prefetched."""
        with self._lock:
            return frozenset(self._pending)

    def wait_all(self, timeout: float = 2.0) -> bool:
        """
        Block until all in-flight prefetches complete (or timeout).
        Returns True if all finished, False on timeout.
        Used in tests to avoid racing.
        """
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if not self._pending:
                    return True
            time.sleep(0.001)
        return False


# ── Band suffix helpers ───────────────────────────────────────────────────────

_BAND_SUFFIXES = ("_warm", "_cordial", "_neutral", "_guarded", "_hostile")


def _strip_band(adapter_label: str) -> str:
    """
    Strip the affect band suffix from a composite adapter label.
    "blacksmith_warm" → "blacksmith"
    "blacksmith"      → "blacksmith"  (no-op if already bare)
    """
    for suffix in _BAND_SUFFIXES:
        if adapter_label.endswith(suffix):
            return adapter_label[: -len(suffix)]
    return adapter_label


if __name__ == "__main__":
    import time
    from mrag.store.manager import EngRamManager
    from mrag.schema.payload import EngRamPayload
    from mrag.bridge.interface import BridgeInterface

    # ── _strip_band ───────────────────────────────────────────────────────────
    assert _strip_band("blacksmith_warm")    == "blacksmith"
    assert _strip_band("guard_hostile")      == "guard"
    assert _strip_band("merchant_neutral")   == "merchant"
    assert _strip_band("merchant_cordial")   == "merchant"
    assert _strip_band("blacksmith")         == "blacksmith"   # bare name
    assert _strip_band("my_npc_warm")        == "my_npc"       # compound name

    # ── PrefetchCoordinator with BridgeInterface ──────────────────────────────
    mgr = EngRamManager(":memory:")
    mgr.write_memory("blacksmith", "a3f9b2c1",
                     EngRamPayload("Sword sale.", 0.9, 0.8, "blacksmith", age=2))

    coordinator = PrefetchCoordinator(mgr, lookahead_ms=0)
    bridge = BridgeInterface(":memory:", prefetch_hint=coordinator.as_prefetch_hook())

    # Share the manager so the coordinator's mount touches the same tables
    bridge._manager = mgr

    trigger_json = '{"adapter_hint":"blacksmith","context_hash":"a3f9b2c1","prompt_preview":"sword?"}'
    resp_json = bridge.handle_trigger_json(trigger_json)

    # Allow prefetch thread to complete
    done = coordinator.wait_all(timeout=1.0)
    assert done, "prefetch did not complete in time"

    import json
    resp = json.loads(resp_json)
    assert resp["adapter_label"] == "blacksmith_warm"

    # Verify deduplication: rapid double-hint queues only one thread
    coordinator.hint("guard_hostile")
    coordinator.hint("guard_hostile")
    coordinator.wait_all(timeout=1.0)
    # No assertion beyond "didn't crash" — dedup is internal state

    bridge.shutdown()
    print("coordinator.py OK")
