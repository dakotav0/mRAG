from pydantic import BaseModel, field_validator
from typing import Optional


# ── Inbound: mRNA Layer 14 fires this when it detects a semantic concept ──

class ContextTrigger(BaseModel):
    """
    Sent by mRNA Layer 14 SAE → mRAG.

    context_hash is produced by mRNA during prefill (where token IDs are
    available) using the same NgramHasher(n=3) reference implementation.
    mRAG uses it directly as a table lookup key — no re-hashing needed.
    """
    adapter_hint:   str           # SAE latent label, e.g. "trading", "combat"
    context_hash:   str           # Hex digest of N=3 token-level N-gram hash
    prompt_preview: str           # First 128 chars for debugging (never injected)
    layer:          int = 14


# ── Outbound: mRAG returns this to mRNA before Layer 25 ──

class EngRamResponse(BaseModel):
    """Sent by mRAG → mRNA Layer 25."""
    adapter_label:   str          # e.g. "warm_merchant", "hostile_guard"
    memory_tokens:   list[str]    # Short facts to prepend to prompt context
    salience_max:    float        # Highest salience among retrieved memories
    affect_mean:     float        # Mean affect across retrieved memories
    evicted_count:   int = 0      # How many memories were below eviction threshold

    @field_validator("salience_max")
    @classmethod
    def salience_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"salience_max must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("affect_mean")
    @classmethod
    def affect_in_range(cls, v: float) -> float:
        if not (-1.0 <= v <= 1.0):
            raise ValueError(f"affect_mean must be in [-1.0, 1.0], got {v}")
        return v


# ── PIDX sync: PIDX pushes identity/decay updates to mRAG ──

class PidxSyncPacket(BaseModel):
    """
    Sent by PIDX → mRAG on identity change or decay tick.

    decay_delta: PIDX NpcMemory uses continuous λ=0.05/day. mRAG uses discrete
    steps at rate=0.05/step. This float carries the raw PIDX delta; the mRAG
    EngRamManager converts it to step increments. Keep the clocks decoupled.
    """
    npc_id:         str
    adapter_name:   str
    decay_delta:    float         # PIDX-native delta; mRAG converts to step count
    salience_boost: Optional[dict[str, float]] = None  # tag → salience modifier


if __name__ == "__main__":
    import json

    # Round-trip ContextTrigger
    t = ContextTrigger(adapter_hint="blacksmith", context_hash="a3f9b2c1",
                       prompt_preview="Player: I want to buy an iron sword.")
    assert ContextTrigger.model_validate_json(t.model_dump_json()) == t

    # Round-trip EngRamResponse
    r = EngRamResponse(adapter_label="blacksmith_warm",
                       memory_tokens=["Player bought a sword last session."],
                       salience_max=0.9, affect_mean=0.8)
    assert EngRamResponse.model_validate_json(r.model_dump_json()) == r

    # Validator rejects out-of-range salience
    try:
        EngRamResponse(adapter_label="x", memory_tokens=[], salience_max=1.5, affect_mean=0.0)
        assert False, "should have raised"
    except Exception:
        pass

    # Round-trip PidxSyncPacket
    p = PidxSyncPacket(npc_id="blacksmith_01", adapter_name="blacksmith",
                       decay_delta=2.0, salience_boost={"combat": 0.1})
    assert PidxSyncPacket.model_validate_json(p.model_dump_json()) == p

    print("bridge.py OK")
