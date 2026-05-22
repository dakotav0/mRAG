from dataclasses import dataclass, field
from typing import Generic, TypeVar, Dict, Any, List, Optional

_DECAY_LOOKUP = [0.95 ** i for i in range(1000)]

T = TypeVar("T")


@dataclass
class Engram(Generic[T]):
    """
    Generic engram memory object for sub-millisecond retrieval.
    """
    value: T
    salience: float = 1.0           # 0.0–1.0
    affect:   float = 0.0           # -1.0 to +1.0
    source:   str   = "unknown"
    age:      int   = 0             # simulated time steps
    tags:     List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngRamPayload(Engram[str]):
    """
    The structured object returned on an Engram hash hit.
    Keeps 100% backward compatibility for text-based pipelines.
    """

    def __init__(
        self,
        text: str,
        salience: float = 1.0,
        affect: float = 0.0,
        source: str = "unknown",
        age: int = 0,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            value=text,
            salience=salience,
            affect=affect,
            source=source,
            age=age,
            tags=tags if tags is not None else [],
            metadata=metadata if metadata is not None else {},
        )

    @property
    def text(self) -> str:
        """The memory string injected into the prompt context."""
        return self.value

    @text.setter
    def text(self, val: str) -> None:
        self.value = val

    def decayed_salience(self, decay_rate: float = 0.05) -> float:
        """PIDX-compatible exponential decay over discrete time steps."""
        if decay_rate == 0.05:
            if self.age < 1000:
                return self.salience * _DECAY_LOOKUP[self.age]
            return 0.0
        return self.salience * (1.0 - decay_rate) ** self.age

    def is_evictable(self, threshold: float = 0.15) -> bool:
        """True when decayed salience falls below the KV cache eviction threshold."""
        return self.decayed_salience() < threshold


if __name__ == "__main__":
    # Smoke tests against mock_packets.json fixtures
    trading = EngRamPayload(text="Player bought a sword last session.",
                            salience=0.9, affect=0.8, source="blacksmith", age=2)
    assert not trading.is_evictable(), "trading_positive should not be evictable"
    assert abs(trading.decayed_salience() - 0.9 * 0.95 ** 2) < 1e-9

    decay_case = EngRamPayload(text="Player haggled poorly three weeks ago.",
                               salience=0.3, affect=-0.3, source="merchant", age=40)
    assert decay_case.is_evictable(), "decay_eviction fixture should be evictable"
    # 0.3 * 0.95^40 ≈ 0.040 < 0.15
    assert decay_case.decayed_salience() < 0.15

    # Generic check
    generic_engram = Engram[dict](value={"coord": (10, 20)}, salience=0.8)
    assert generic_engram.value["coord"] == (10, 20)
    assert generic_engram.salience == 0.8

    print("payload.py OK")
