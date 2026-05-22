"""
benchmark_mrag.py — Comprehensive latency and optimization benchmarks for mRAG.

Generates realistic RPG workloads, measures hot, cold, I/O, and decay paths,
and provides profiling details to locate performance bottlenecks.

Run:
    python scripts/benchmark_mrag.py --npcs 5 --memories-per-npc 1000 --queries 500
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import random
import shutil
import sys
import time
from io import StringIO
from pathlib import Path
from statistics import mean, median

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from mrag.bridge.interface import BridgeInterface
from mrag.hash.ngram_hasher import NgramHasher
from mrag.schema.bridge import ContextTrigger
from mrag.schema.payload import EngRamPayload

# Vocabulary of templates for high-realism synthetic memories
TEMPLATES = [
    "Player came to buy a {item}.",
    "Player haggled aggressively over the price of {item}.",
    "Player sold {quantity} {item} to the merchant.",
    "Player was seen loitering near the {place} at {time_of_day}.",
    "Player helped defend the {place} against {enemy}s.",
    "Player was caught stealing {item} from the {place}.",
    "NPC remembers the player talked about {topic}.",
    "NPC is grateful that player defeated {enemy} near the {place}.",
]

ITEMS = ["iron sword", "healing potion", "rare spices", "silver shield", "ancient relic", "bread", "ale"]
PLACES = ["market square", "castle treasury", "tavern hall", "northern gate", "blacksmith shop", "temple sanctuary"]
ENEMIES = ["goblin", "orc", "bandit", "hobgoblin", "shadow beast"]
TIMES = ["midnight", "dawn", "noon", "dusk", "twilight"]
TOPICS = ["the northern war", "ancient dragons", "hidden gold", "the lost king", "elven magic"]
TAGS = ["trade", "crime", "alert", "reputation", "lore", "craft"]

def generate_random_sentence() -> str:
    tpl = random.choice(TEMPLATES)
    return tpl.format(
        item=random.choice(ITEMS),
        quantity=random.randint(1, 20),
        place=random.choice(PLACES),
        enemy=random.choice(ENEMIES),
        time_of_day=random.choice(TIMES),
        topic=random.choice(TOPICS)
    )

class mRAGBenchmark:
    def __init__(self, npcs: int, memories_per_npc: int, queries: int, verbose: bool):
        self.num_npcs = npcs
        self.memories_per_npc = memories_per_npc
        self.num_queries = queries
        self.verbose = verbose
        self.hasher = NgramHasher()

        # Temporary workspace data directory for benchmark SQLite DBs
        self.data_dir = Path(__file__).parent.parent / "data" / "benchmark_tables"
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.npc_names = [f"npc_{i:02d}" for i in range(self.num_npcs)]
        self.stored_keys: dict[str, list[str]] = {name: [] for name in self.npc_names}
        self.stored_memories: dict[str, list[str]] = {name: [] for name in self.npc_names}

    def cleanup(self):
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)

    def _ms(self, start: float) -> float:
        return (time.perf_counter() - start) * 1000.0

    def populate_database(self):
        print(f"Populating DBs with {self.num_npcs} NPCs, {self.memories_per_npc} memories each...")
        t0 = time.perf_counter()
        
        # We write directly using a BridgeInterface temporary instance to build the SQLite DBs
        bridge = BridgeInterface(str(self.data_dir), max_loaded=self.num_npcs + 2)
        
        for npc in self.npc_names:
            for i in range(self.memories_per_npc):
                text = generate_random_sentence()
                
                # Make pseudo-tokens and lookup key
                words = text.lower().split()
                # word-to-id mapping formula matching demo
                ids = [(sum(ord(c) * (31 ** idx) for idx, c in enumerate(w)) % 32000) for w in words]
                key = self.hasher.lookup_key(ids) + f"_{i}"
                
                payload = EngRamPayload(
                    text=text,
                    salience=random.uniform(0.2, 1.0),
                    affect=random.uniform(-1.0, 1.0),
                    source=npc,
                    age=random.randint(0, 50),
                    tags=random.sample(TAGS, random.randint(1, 3))
                )
                
                bridge._manager.write_memory(npc, key, payload)
                self.stored_keys[npc].append(key)
                self.stored_memories[npc].append(text)
                
        bridge.shutdown()
        print(f"Populated database in {self._ms(t0):.2f}ms.\n")

    def run_hashing_benchmark(self) -> dict:
        print("Benchmarking NgramHasher...")
        
        # Test sequences
        seq_short = list(range(20))
        seq_long = list(range(150))
        text_query = "Player: I want to buy a masterwork steel sword and some potions from the shop."
        
        # 1. Hashing short sequence
        t0 = time.perf_counter()
        iters = 10000
        for _ in range(iters):
            self.hasher.hash_context(seq_short)
        short_seq_time = self._ms(t0) / iters

        # 2. Hashing long sequence
        t0 = time.perf_counter()
        for _ in range(iters):
            self.hasher.hash_context(seq_long)
        long_seq_time = self._ms(t0) / iters

        # 3. Canonical lookup key
        t0 = time.perf_counter()
        for _ in range(iters):
            self.hasher.lookup_key(seq_short)
        lookup_key_time = self._ms(t0) / iters

        # 4. Tokenization (pseudo-trigram)
        t0 = time.perf_counter()
        for _ in range(iters):
            self.hasher.tokenize_text(text_query)
        tokenize_time = self._ms(t0) / iters

        # 5. Jaccard text similarity
        cand = "Player bought a masterwork sword."
        t0 = time.perf_counter()
        for _ in range(iters):
            self.hasher.text_similarity(text_query, cand)
        jaccard_time = self._ms(t0) / iters

        print(f"  hash_context (20 tokens):  {short_seq_time * 1000:.3f} us")
        print(f"  hash_context (150 tokens): {long_seq_time * 1000:.3f} us")
        print(f"  lookup_key:                {lookup_key_time * 1000:.3f} us")
        print(f"  tokenize_text:             {tokenize_time * 1000:.3f} us")
        print(f"  text_similarity:           {jaccard_time * 1000:.3f} us\n")
        
        return {
            "hash_short": short_seq_time,
            "hash_long": long_seq_time,
            "lookup_key": lookup_key_time,
            "tokenize": tokenize_time,
            "similarity": jaccard_time
        }

    def run_hot_path_benchmark(self, bridge: BridgeInterface) -> tuple[list[float], float]:
        """Direct hash hits on already warm mounted tables."""
        latencies = []
        
        # Warm up manager cache by triggering once on each NPC table
        for npc in self.npc_names:
            key = self.stored_keys[npc][0]
            trigger = ContextTrigger(adapter_hint=npc, context_hash=key, prompt_preview="warm")
            bridge.handle_trigger(trigger)
            
        print("Benchmarking Hot Path (Direct Hash Hit)...")
        t_start = time.perf_counter()
        
        for _ in range(self.num_queries):
            npc = random.choice(self.npc_names)
            key = random.choice(self.stored_keys[npc])
            
            trigger = ContextTrigger(
                adapter_hint=npc,
                context_hash=key,
                prompt_preview="bench_hot"
            )
            
            t0 = time.perf_counter()
            resp = bridge.handle_trigger(trigger)
            latencies.append(self._ms(t0))
            
            assert resp.salience_max > 0, "Hot path hit failed!"
            
        total_time = self._ms(t_start)
        
        p50 = median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        throughput = self.num_queries / (total_time / 1000.0)
        
        print(f"  p50:        {p50:.4f} ms")
        print(f"  p95:        {p95:.4f} ms")
        print(f"  p99:        {p99:.4f} ms")
        print(f"  avg:        {mean(latencies):.4f} ms")
        print(f"  throughput: {throughput:.1f} queries/sec")
        print(f"  total time: {total_time:.2f} ms for {self.num_queries} queries\n")
        
        return latencies, throughput

    def run_cold_path_benchmark(self, bridge: BridgeInterface) -> tuple[list[float], float]:
        """Hash misses, forcing text similarity Jaccard re-ranking of all memories."""
        latencies = []
        
        print("Benchmarking Cold Path (Hash Miss + Jaccard Re-ranking)...")
        
        # Verify first that we indeed miss
        miss_key = "non_existent_hash"
        
        t_start = time.perf_counter()
        
        for _ in range(self.num_queries // 5):  # Fewer samples since this is much slower
            npc = random.choice(self.npc_names)
            query_text = generate_random_sentence()
            
            trigger = ContextTrigger(
                adapter_hint=npc,
                context_hash=miss_key,
                prompt_preview="bench_cold"
            )
            
            t0 = time.perf_counter()
            resp = bridge.handle_trigger(trigger, query_text=query_text)
            latencies.append(self._ms(t0))
            
        total_time = self._ms(t_start)
        
        p50 = median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        throughput = (self.num_queries // 5) / (total_time / 1000.0)
        
        print(f"  p50:        {p50:.4f} ms")
        print(f"  p95:        {p95:.4f} ms")
        print(f"  p99:        {p99:.4f} ms")
        print(f"  avg:        {mean(latencies):.4f} ms")
        print(f"  throughput: {throughput:.1f} queries/sec")
        print(f"  total time: {total_time:.2f} ms for {self.num_queries // 5} queries\n")
        
        return latencies, throughput

    def run_io_benchmark(self) -> dict:
        print("Benchmarking SQLite I/O (Mount / Unmount tables)...")
        
        bridge = BridgeInterface(str(self.data_dir), max_loaded=1)  # force eviction every mount
        
        # We will measure the duration of mounting an unmounted table from disk
        mount_times = []
        
        for npc in self.npc_names[:5]:
            # Evict everyone else by mounting something else, or unmount explicitly
            bridge._manager.unmount_all()
            
            t0 = time.perf_counter()
            bridge._manager.mount(npc)
            mount_times.append(self._ms(t0))
            
        avg_mount = mean(mount_times)
        
        # Measure save times (unmount)
        save_times = []
        for npc in self.npc_names[:5]:
            t0 = time.perf_counter()
            bridge._manager.unmount(npc)
            save_times.append(self._ms(t0))
            
        avg_save = mean(save_times)
        
        bridge.shutdown()
        
        print(f"  Average cold mount: {avg_mount:.2f} ms")
        print(f"  Average unmount (SQLite save): {avg_save:.2f} ms\n")
        
        return {"avg_mount": avg_mount, "avg_save": avg_save}

    def run_profiler(self):
        print("══════════════════════════════════════════════════════════════")
        print("                  PROFILER RUN (cProfile)")
        print("══════════════════════════════════════════════════════════════")
        
        bridge = BridgeInterface(str(self.data_dir), max_loaded=self.num_npcs + 2)
        
        # Warm tables
        for npc in self.npc_names:
            bridge._manager.mount(npc)
            
        # We will profile 200 hot path queries and 50 cold path queries
        pr = cProfile.Profile()
        pr.enable()
        
        # Hot queries
        for _ in range(200):
            npc = random.choice(self.npc_names)
            key = random.choice(self.stored_keys[npc])
            trigger = ContextTrigger(adapter_hint=npc, context_hash=key, prompt_preview="profiler")
            bridge.handle_trigger(trigger)
            
        # Cold queries
        for _ in range(50):
            npc = random.choice(self.npc_names)
            query_text = generate_random_sentence()
            trigger = ContextTrigger(adapter_hint=npc, context_hash="non_existent", prompt_preview="profiler")
            bridge.handle_trigger(trigger, query_text=query_text)
            
        pr.disable()
        bridge.shutdown()
        
        s = StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(30)
        print(s.getvalue())

def main():
    parser = argparse.ArgumentParser(description="mRAG Core Performance Benchmark")
    parser.add_argument("--npcs", type=int, default=5, help="Number of NPCs to generate tables for")
    parser.add_argument("--memories-per-npc", type=int, default=1000, help="Number of memories per NPC table")
    parser.add_argument("--queries", type=int, default=500, help="Number of queries to run during hot benchmark")
    parser.add_argument("--verbose", action="store_true", help="Print verbose scenario statements")
    
    args = parser.parse_args()
    
    print("══════════════════════════════════════════════════════════════")
    print(f"             mRAG PERFORMANCE BENCHMARK ENGINE")
    print("══════════════════════════════════════════════════════════════")
    print(f"  Configuration:")
    print(f"    - NPCs:             {args.npcs}")
    print(f"    - Memories per NPC: {args.memories_per_npc} (Total: {args.npcs * args.memories_per_npc})")
    print(f"    - Query Samples:    {args.queries}")
    print("══════════════════════════════════════════════════════════════\n")
    
    benchmark = mRAGBenchmark(args.npcs, args.memories_per_npc, args.queries, args.verbose)
    
    try:
        benchmark.populate_database()
        benchmark.run_hashing_benchmark()
        
        # Load up a BridgeInterface with the generated DBs
        bridge = BridgeInterface(str(benchmark.data_dir), max_loaded=args.npcs + 2)
        
        # Run Benchmarks
        benchmark.run_hot_path_benchmark(bridge)
        benchmark.run_cold_path_benchmark(bridge)
        
        bridge.shutdown()
        
        benchmark.run_io_benchmark()
        benchmark.run_profiler()
        
    finally:
        benchmark.cleanup()
        print("Benchmark complete and clean.")

if __name__ == "__main__":
    main()
