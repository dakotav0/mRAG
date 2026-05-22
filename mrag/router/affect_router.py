"""
AffectRouter — maps affect float + adapter hint → composite adapter label.

This is a lookup table, not a neural network. The base model stays frozen.
Affect bands map directly to .mrna adapter filenames, so the band names
must stay in sync with the LoRA adapter naming convention in mRNA.

Band boundaries are inclusive on the lower end, exclusive on the upper end,
except for the outermost bands which clamp at ±1.0.
"""

from __future__ import annotations

from typing import NamedTuple, Callable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mrag.store.engram_table import EngRamTable
    from mrag.schema.payload import EngRamPayload



class _Band(NamedTuple):
    low:  float
    high: float
    name: str


# Ordered high → low so the first match wins on boundary values.
AFFECT_BANDS: list[_Band] = [
    _Band(+0.6,  1.01, "warm"),      # nostalgic, friendly, generous
    _Band(+0.2,  0.6,  "cordial"),   # neutral-positive, professional
    _Band(-0.2,  0.2,  "neutral"),   # no strong valence
    _Band(-0.6, -0.2,  "guarded"),   # wary, clipped, transactional
    _Band(-1.01,-0.6,  "hostile"),   # betrayal, anger, threatening
]


class AffectRouter:
    """
    Converts affect_mean from retrieved memories into a composite adapter label.
    Supports dependency injection for custom affect bands and composite routing formats.

    Usage
    -----
    router = AffectRouter()
    label = router.route(affect=+0.7, adapter_hint="blacksmith")
    # → "blacksmith_warm"
    """

    def __init__(
        self,
        bands: Optional[List[_Band]] = None,
        format_fn: Optional[Callable[[str, str], str]] = None
    ) -> None:
        self.bands = bands if bands is not None else AFFECT_BANDS
        # Default layout format is legacy "{adapter_hint}_{band}"
        self.format_fn = format_fn if format_fn is not None else (lambda hint, band: f"{hint}_{band}")

    def _band_for(self, affect: float) -> str:
        """Return the band name for a given affect value. Clamps to [-1, 1]."""
        affect = max(-1.0, min(1.0, affect))
        for band in self.bands:
            if band.low <= affect < band.high:
                return band.name
        # Should be unreachable given the clamp and band coverage, but fail loud.
        raise ValueError(f"affect={affect} did not match any band in router configuration")

    def route(self, affect: float, adapter_hint: str) -> str:
        """
        Return composite adapter label: e.g. "{adapter_hint}_{band}".

        Parameters
        ----------
        affect : float
            Mean affect across retrieved EngRamPayloads (from EngRamResponse).
        adapter_hint : str
            SAE concept label from ContextTrigger (e.g. "blacksmith", "guard").
        """
        band = self._band_for(affect)
        return self.format_fn(adapter_hint, band)

    def route_from_table(self, adapter_hint: str,
                         table: EngRamTable,
                         top_n: int = 5) -> tuple[str, float, float]:
        """
        Derive the adapter label directly from a mounted EngRamTable.

        Returns (adapter_label, salience_max, affect_mean) — the three values
        EngRamResponse needs — so the bridge doesn't have to recompute them.

        Parameters
        ----------
        adapter_hint : str
            SAE concept label from ContextTrigger.
        table : EngRamTable
            The mounted in-memory table for this adapter.
        top_n : int
            How many top-salience payloads to use for affect averaging.
        """
        candidates = table.top_by_salience(top_n)

        if not candidates:
            return self.format_fn(adapter_hint, "neutral"), 0.0, 0.0

        # Note: the table's decay_policy is used by top_by_salience, but we also use it here to pull saliences.
        # Check if table has a decay_policy, else fall back to legacy payload decayed_salience
        decay_policy = getattr(table, "decay_policy", None)
        if decay_policy is not None:
            salience_max = max(decay_policy.decayed_salience(p.salience, p.age) for p in candidates)
        else:
            salience_max = max(p.decayed_salience() for p in candidates)

        affect_mean  = sum(p.affect for p in candidates) / len(candidates)
        label        = self.route(affect_mean, adapter_hint)
        return label, salience_max, affect_mean


if __name__ == "__main__":
    r = AffectRouter()

    # Band boundary checks (convention: low inclusive, high exclusive)
    # Band edges:  warm=[0.6,∞), cordial=[0.2,0.6), neutral=[-0.2,0.2),
    #              guarded=[-0.6,-0.2), hostile=(-∞,-0.6)
    assert r.route(+1.0,  "blacksmith") == "blacksmith_warm"
    assert r.route(+0.6,  "blacksmith") == "blacksmith_warm"     # low-inclusive
    assert r.route(+0.59, "blacksmith") == "blacksmith_cordial"
    assert r.route(+0.2,  "blacksmith") == "blacksmith_cordial"  # low-inclusive
    assert r.route(+0.19, "blacksmith") == "blacksmith_neutral"
    assert r.route( 0.0,  "blacksmith") == "blacksmith_neutral"
    assert r.route(-0.19, "blacksmith") == "blacksmith_neutral"
    assert r.route(-0.2,  "blacksmith") == "blacksmith_neutral"  # low-inclusive → neutral
    assert r.route(-0.21, "blacksmith") == "blacksmith_guarded"
    assert r.route(-0.59, "blacksmith") == "blacksmith_guarded"
    assert r.route(-0.6,  "blacksmith") == "blacksmith_guarded"  # low-inclusive → guarded
    assert r.route(-0.61, "blacksmith") == "blacksmith_hostile"
    assert r.route(-1.0,  "blacksmith") == "blacksmith_hostile"

    # Custom router configurations
    custom_bands = [
        _Band(0.0, 1.01, "happy"),
        _Band(-1.01, 0.0, "sad")
    ]
    r_custom = AffectRouter(bands=custom_bands, format_fn=lambda h, b: f"{h}::{b}")
    assert r_custom.route(0.5, "guard") == "guard::happy"
    assert r_custom.route(-0.5, "guard") == "guard::sad"

    print("affect_router.py OK")
