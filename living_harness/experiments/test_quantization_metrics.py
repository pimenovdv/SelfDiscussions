import sys
import os
import json
import random
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from living_harness.memory.vector_store import VectorStore

def run_experiment():
    random.seed(42)
    # Generate 100 random vectors of size 128
    vectors = [[random.uniform(-1.0, 1.0) for _ in range(128)] for _ in range(100)]
    queries = [[random.uniform(-1.0, 1.0) for _ in range(128)] for _ in range(10)]

    # Non-quantized store
    store_normal = VectorStore(db_path="living_harness/data/temp_normal.json")
    store_normal.clear()
    for i, vec in enumerate(vectors):
        store_normal.add_item(f"doc_{i}", vec, quantize=False)

    size_normal = os.path.getsize("living_harness/data/temp_normal.json")

    # Quantized store
    store_quantized = VectorStore(db_path="living_harness/data/temp_quantized.json")
    store_quantized.clear()
    for i, vec in enumerate(vectors):
        store_quantized.add_item(f"doc_{i}", vec, quantize=True)

    size_quantized = os.path.getsize("living_harness/data/temp_quantized.json")

    # Calculate precision (top 5 overlap)
    total_overlap = 0
    for q in queries:
        res_normal = [item['text'] for item in store_normal.search(q, top_k=5)]
        res_quantized = [item['text'] for item in store_quantized.search(q, top_k=5)]
        overlap = len(set(res_normal) & set(res_quantized))
        total_overlap += overlap

    precision = total_overlap / (len(queries) * 5)

    metrics = {
        "size_normal_bytes": size_normal,
        "size_quantized_bytes": size_quantized,
        "compression_ratio": size_normal / size_quantized if size_quantized else 0,
        "top_5_precision": precision
    }

    with open("living_harness/data/quantization_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics: {metrics}")

    store_normal.clear()
    store_quantized.clear()

if __name__ == "__main__":
    run_experiment()
