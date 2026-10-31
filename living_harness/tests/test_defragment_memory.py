import pytest
import time
import numpy as np
from living_harness.memory.vector_store import VectorStore
import os

def test_defragmentation_metrics():
    db_path = "data/test_defrag_metrics.json"
    if os.path.exists(db_path):
        os.remove(db_path)

    store = VectorStore(db_path=db_path)
    store.clear()

    # Generate many items with slight variations (duplicates)
    base_vectors = [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5]
    ]

    np.random.seed(42)
    for i in range(500):
        base_idx = i % len(base_vectors)
        base_vec = base_vectors[base_idx]

        # Add slight noise to make them near-duplicates
        noise = np.random.normal(0, 0.01, 3)
        noisy_vec = [float(v + n) for v, n in zip(base_vec, noise)]

        store.add_item(f"Item {i} type {base_idx}", noisy_vec)

    initial_count = len(store.memory)

    # Measure search time before defragmentation
    start_time = time.time()
    for _ in range(100):
        store.search([0.8, 0.1, 0.1], top_k=5)
    search_time_before = time.time() - start_time

    # Defragment
    defrag_start_time = time.time()
    store.defragment_memory(similarity_threshold=0.98)
    defrag_time = time.time() - defrag_start_time

    final_count = len(store.memory)

    # Measure search time after defragmentation
    start_time = time.time()
    for _ in range(100):
        store.search([0.8, 0.1, 0.1], top_k=5)
    search_time_after = time.time() - start_time

    print(f"\n--- Defragmentation Metrics ---")
    print(f"Initial count: {initial_count}")
    print(f"Final count: {final_count}")
    print(f"Reduction: {(initial_count - final_count) / initial_count * 100:.1f}%")
    print(f"Defragmentation time: {defrag_time:.4f}s")
    print(f"Search time before (100 queries): {search_time_before:.4f}s")
    print(f"Search time after (100 queries): {search_time_after:.4f}s")
    print(f"Search speedup: {search_time_before / search_time_after:.2f}x")
    print(f"-------------------------------")

    assert final_count <= initial_count / 2

    store.clear()
    if os.path.exists(db_path):
        os.remove(db_path)
