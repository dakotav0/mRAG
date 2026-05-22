"""
Shared pytest fixtures for mRAG tests.

All fixtures use BridgeInterface(":memory:") — no external model dependencies,
no GPU required. The mock_packets.json file is the only external dependency.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from mrag.bridge.interface import BridgeInterface
from mrag.schema.payload import EngRamPayload

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "mock_packets.json"


@pytest.fixture(scope="session")
def mock_packets() -> list[dict[str, Any]]:
    """Load mock_packets.json once for the entire test session."""
    return json.loads(FIXTURES_PATH.read_text())


@pytest.fixture()
def bridge() -> BridgeInterface:
    """
    Fresh BridgeInterface backed by an in-memory SQLite store.
    Each test gets its own isolated instance.
    """
    b = BridgeInterface(":memory:")
    yield b
    b.shutdown()


@pytest.fixture()
def populated_bridge(mock_packets) -> BridgeInterface:
    """
    BridgeInterface pre-loaded with all mock_packets memories.

    Each memory is stored under its fixture's context_hash key so that
    handle_trigger can find it on an O(1) direct lookup.
    """
    b = BridgeInterface(":memory:")

    for packet in mock_packets:
        adapter_hint = packet["trigger"]["adapter_hint"]
        context_hash = packet["trigger"]["context_hash"]
        for mem in packet["memories"]:
            payload = EngRamPayload(
                text=mem["text"],
                salience=mem["salience"],
                affect=mem["affect"],
                source=mem["source"],
                age=mem.get("age", 0),
                tags=mem.get("tags", []),
            )
            b._manager.write_memory(adapter_hint, context_hash, payload)

    yield b
    b.shutdown()
