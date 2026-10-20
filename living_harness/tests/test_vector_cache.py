import pytest
import time
from living_harness.memory.vector_store import VectorStore
from typing import List, Dict, Any, Tuple
import json
import os

def test_vector_cache():
    # Use a proper path for the test database
    db_path = "data/test_vector_cache.json"
    store = VectorStore(db_path=db_path)
    store.clear()

    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    query = [1.0, 0.0, 0.0]

    # First add a small amount
    for i in range(100):
        store.add_item(f"item_v1_{i}", v1, domain="test")
        store.add_item(f"item_v2_{i}", v2, domain="test")

    # First search, should calculate and cache
    start_time = time.perf_counter()
    results1 = store.search(query_vector=query, top_k=10, domain="test")
    first_duration = time.perf_counter() - start_time

    assert len(results1) == 10

    # Verify cache is populated
    cache_key = (tuple(query), 10, "test")
    assert cache_key in store._search_cache

    # Second search, should use cache
    start_time = time.perf_counter()
    results2 = store.search(query_vector=query, top_k=10, domain="test")
    second_duration = time.perf_counter() - start_time

    assert len(results2) == 10

    # Ensure results are identical
    assert [r["text"] for r in results1] == [r["text"] for r in results2]

    assert cache_key in store._search_cache

    # Invalidate cache by adding new item
    store.add_item("item3", [0.0, 0.0, 1.0], domain="test")
    assert len(store._search_cache) == 0

    # Calculate speedup
    speedup = first_duration / second_duration if second_duration > 0 else float('inf')

    # Record numerical metrics
    metrics = {
        "first_duration_sec": first_duration,
        "second_duration_sec": second_duration,
        "speedup": speedup
    }

    os.makedirs("data", exist_ok=True)
    metrics_path = "data/vector_cache_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    store.clear()
