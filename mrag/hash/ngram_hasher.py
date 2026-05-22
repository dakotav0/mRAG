"""
N-gram hasher for mRAG engram table keys.

N is a CONFIG CONSTANT — it determines table structure and cannot change
after tables are built. Lock it here; never make it a runtime parameter.

Hash function: xxHash (xxh64) for speed on the hot path.
Fallback: first 16 hex chars of SHA-256 if xxhash is not installed.
No security requirement — collision resistance comes from N=3 window diversity.

Why N=3 (token-level trigrams):
- Short turn-based prompt (20 tokens): 18 trigrams — enough to
  discriminate "buy iron sword" from "attack market stall" without the high
  collision rate of bigrams ("do you", "I want").
- Turn-based (100-token prompts): 98 trigrams — the dominant-frequency trigram
  becomes MORE stable with longer context, not less. The highest-frequency
  trigram reliably anchors to the semantic theme of the turn.
- Character trigrams would add 3x entries with no benefit; BPE subword vocab
  already captures morphological units.

lookup_key strategy: Option B — highest-frequency N-gram (tie → first seen).
XOR-fold (Option A) is order-independent and lossy. The most-repeated trigram
captures the dominant semantic concept in both short and long contexts.
"""

import hashlib
import random
import struct

# ── Config constant ─────────────────────────────────────────────────────────
# This determines table column structure. DO NOT make this a runtime parameter.
NGRAM_N: int = 3

# ── Hash backend ─────────────────────────────────────────────────────────────
try:
    import xxhash as _xxhash
    _HAS_XXHASH: bool = True
except ImportError:
    _xxhash = None  # type: ignore[assignment]
    _HAS_XXHASH = False


def _hash_bytes(data: bytes, seed: int) -> str:
    """Return a hex digest for the given byte string."""
    if _HAS_XXHASH and _xxhash is not None:
        return _xxhash.xxh64(data, seed=seed).hexdigest()
    # Fallback: truncate SHA-256 to 16 hex chars (64-bit equivalent length)
    return hashlib.sha256(data).hexdigest()[:16]


class NgramHasher:
    """
    Produces per-N-gram hex digests and a single canonical lookup key
    for a list of token IDs.

    Parameters
    ----------
    n : int
        Gram size. Must equal NGRAM_N (3). Exposed as a parameter only for
        test harness use — production code always passes NGRAM_N.
    seed : int
        xxHash seed for domain separation across different mRAG deployments.
    """

    def __init__(self, n: int = NGRAM_N, seed: int = 42):
        if n != NGRAM_N:
            raise ValueError(
                f"n={n} conflicts with NGRAM_N={NGRAM_N}. "
                "Table structure is locked to NGRAM_N. Changing N requires "
                "rebuilding all adapter tables."
            )
        self.n = n
        self.seed = seed

    def _ngram_bytes(self, gram: tuple[int, ...]) -> bytes:
        """Pack a tuple of token IDs as little-endian int32 bytes."""
        return struct.pack(f"<{len(gram)}i", *gram)

    def hash_context(self, token_ids: list[int]) -> list[str]:
        """
        Returns one hex digest per N-gram window (sliding, step=1).

        An input shorter than N returns an empty list — callers should
        fall back to a unigram or prefix hash in that case.
        """
        if len(token_ids) < self.n:
            return []
        return [
            _hash_bytes(self._ngram_bytes(tuple(token_ids[i:i + self.n])), self.seed)
            for i in range(len(token_ids) - self.n + 1)
        ]

    def lookup_key(self, token_ids: list[int]) -> str:
        """
        Returns the single canonical lookup key for this token context.

        Always uses the full token-ID sequence — the dominant-ngram strategy
        is appropriate for BPE token vocabularies (~32k discrete types) where
        a repeated trigram signals a genuine theme, but character trigrams
        produce catastrophic collisions ("the", "ing", "and" dominate every
        English text). Full-sequence hashing gives stable, unique keys.

        The ngram hash list (hash_context) is preserved for future similarity
        matching.

        Sub-N input: hashes the full token sequence directly.
        """
        packed = self._ngram_bytes(tuple(token_ids)) if token_ids else b""
        return _hash_bytes(packed, self.seed)

    def tokenize_text(self, text: str) -> list[int]:
        """
        Convert raw text to pseudo-token IDs using character trigrams.

        Each character trigram (sliding window of 3 chars) maps to a
        deterministic u32 via byte-packing. Works on any text without
        requiring a BPE tokenizer — the hash space is vocabulary-independent.

        Text shorter than 3 chars returns a single-token list using the
        full string packed as one pseudo-token.

        This is the bridge between free-text input and the N-gram hasher's
        token-ID-based pipeline. Equivalent to _text_to_key's old role but
        produces a proper token-ID sequence instead of word-ordinal-sums.
        """
        chars = text.lower()
        if len(chars) < 3:
            # Sub-trigram text: pack the full string as one pseudo-token
            packed = chars.encode('utf-8', errors='replace')
            val = 0
            for i, b in enumerate(packed[:4]):
                val |= b << (8 * i)
            return [val]

        ids = []
        for i in range(len(chars) - 2):
            trigram = chars[i:i + 3]
            # Pack 3 chars as a u24 (big-endian style): ord0<<16 | ord1<<8 | ord2
            val = (ord(trigram[0]) << 16) | (ord(trigram[1]) << 8) | ord(trigram[2])
            ids.append(val)
        return ids

    def text_similarity(self, query: str, candidate: str) -> float:
        """
        Character-trigram Jaccard similarity between two texts.

        Returns 0.0–1.0. Fast, vocabulary-independent, zero dependencies.
        Used as a lightweight embedding fallback for re-ranking when the
        hash path misses.
        """
        q_ids = set(self.tokenize_text(query))
        c_ids = set(self.tokenize_text(candidate))
        if not q_ids or not c_ids:
            return 0.0
        intersection = q_ids & c_ids
        union = q_ids | c_ids
        return len(intersection) / len(union)


if __name__ == "__main__":
    h = NgramHasher()

    # Determinism: same input → same key
    ids_a = [101, 202, 303, 404, 505]
    assert h.lookup_key(ids_a) == h.lookup_key(ids_a), "lookup_key must be deterministic"

    # Discrimination: distinct short sequences → distinct keys
    ids_buy    = [1000, 2000, 3000, 4000, 5000]
    ids_attack = [9000, 8000, 7000, 6000, 5000]
    assert h.lookup_key(ids_buy) != h.lookup_key(ids_attack), \
        "distinct token sequences should produce distinct keys"

    # Short input (below N) returns a non-empty fallback key
    key_short = h.lookup_key([42, 99])
    assert isinstance(key_short, str) and len(key_short) > 0

    # hash_context length: len(ids) - N + 1
    ids_long = list(range(20))
    hashes = h.hash_context(ids_long)
    assert len(hashes) == 20 - 3 + 1 == 18, f"expected 18 hashes, got {len(hashes)}"

    # Collision check on 100 synthetic sequences
    random.seed(0)
    vocab_size = 32_000  # typical BPE vocab
    sequences = [
        [random.randint(0, vocab_size - 1) for _ in range(random.randint(10, 50))]
        for _ in range(100)
    ]
    keys = [h.lookup_key(seq) for seq in sequences]
    unique = len(set(keys))
    collision_rate = 1.0 - unique / len(keys)
    assert collision_rate < 0.05, \
        f"collision rate {collision_rate:.2%} exceeds 5% threshold"

    backend = "xxhash" if _HAS_XXHASH else "sha256-fallback"
    print(f"ngram_hasher.py OK  [{backend}, N={NGRAM_N}, "
          f"collisions={collision_rate:.1%} on 100 samples]")
