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

def test_cache_eviction_lru():
    db_path = "data/test_cache_eviction.json"
    # Small cache size to test eviction
    store = VectorStore(db_path=db_path, max_cache_size=5)
    store.clear()

    v1 = [1.0, 0.0, 0.0]
    for i in range(20):
        store.add_item(f"item_v1_{i}", v1, domain="test")

    # Do 5 searches with different queries to fill the cache
    queries = []
    for i in range(5):
        q = [1.0, float(i)/10.0, 0.0]
        queries.append(q)
        store.search(query_vector=q, top_k=3, domain="test")

    assert len(store._search_cache) == 5

    first_query = tuple(queries[0])
    first_cache_key = (first_query, 3, "test")
    assert first_cache_key in store._search_cache

    # Do a 6th search with a new query to evict the first one
    q6 = [1.0, 0.6, 0.0]
    store.search(query_vector=q6, top_k=3, domain="test")

    assert len(store._search_cache) == 5
    assert first_cache_key not in store._search_cache

    # Check that LRU works: access the 2nd query to make it newest
    second_query = tuple(queries[1])
    second_cache_key = (second_query, 3, "test")
    store.search(query_vector=list(second_query), top_k=3, domain="test")

    # Now do a 7th search to evict the oldest one (which is now the 3rd query)
    q7 = [1.0, 0.7, 0.0]
    store.search(query_vector=q7, top_k=3, domain="test")

    assert len(store._search_cache) == 5
    assert second_cache_key in store._search_cache # 2nd query is kept

    third_query = tuple(queries[2])
    third_cache_key = (third_query, 3, "test")
    assert third_cache_key not in store._search_cache # 3rd query is evicted

    store.clear()
