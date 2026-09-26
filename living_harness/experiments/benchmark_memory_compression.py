import time
import json
import random
import os
from living_harness.memory.consolidation import MemoryConsolidator

def run_benchmark():
    num_episodes = 1000
    vector_dim = 64
    num_clusters = 10

    # Generate mock episodic memories with some clusters explicitly created
    memories = []
    for i in range(num_episodes):
        cluster_id = i % num_clusters
        base_vector = [1.0 if j % num_clusters == cluster_id else 0.0 for j in range(vector_dim)]
        vector = [v + random.uniform(-0.1, 0.1) for v in base_vector]

        memories.append({
            "id": i,
            "content": f"Memory about event {i}",
            "vector": vector,
            "timestamp": time.time()
        })

    consolidator = MemoryConsolidator(num_clusters=num_clusters)

    start_time = time.time()
    abstract_rules = consolidator.consolidate_memories(memories)
    end_time = time.time()

    original_size = len(memories)
    compressed_size = len(abstract_rules)
    compression_ratio = (original_size - compressed_size) / original_size * 100

    results = {
        "num_episodes": original_size,
        "vector_dim": vector_dim,
        "num_clusters": num_clusters,
        "compressed_size": compressed_size,
        "compression_ratio_percent": compression_ratio,
        "execution_time_sec": end_time - start_time,
        "abstract_rules_preview": [rule["content"][:100] + "..." for rule in abstract_rules[:3]]
    }

    # Use path relative to the script location assuming it's in living_harness/experiments
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    out_path = os.path.join(data_dir, "memory_compression_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Benchmark completed. Compression ratio: {compression_ratio:.2f}%. Results saved to {out_path}")

if __name__ == "__main__":
    run_benchmark()
