import pytest
import time
from living_harness.memory.vector_store import VectorStore
import os
import json

def test_predictive_caching_latency():
    db_path = "living_harness/data/test_predictive_caching.json"
    if os.path.exists(db_path):
        os.remove(db_path)

    store = VectorStore(db_path=db_path)

    # Populate store with some data
    for i in range(50):
        store.add_item(f"Item {i} A", [1.0, 0.0, 0.0], domain="domain_a")
    for i in range(50):
        store.add_item(f"Item {i} B", [0.0, 1.0, 0.0], domain="domain_b")

    # Establish transition A -> B
    store.search([1.0, 0.0, 0.0], domain="domain_a")
    store.search([0.0, 1.0, 0.0], domain="domain_b")

    # Query in domain_a to trigger transition to B contextually next
    store.search([1.0, 0.0, 0.0], domain="domain_a")

    # Measure time without cache for domain_b
    store._search_cache.clear() # Ensure cache is empty

    query_b = [0.0, 0.0, 0.0]

    # Cold start (cache miss)
    start_time = time.perf_counter()
    store.search(query_b, domain="domain_b", top_k=5)
    time_cold = time.perf_counter() - start_time

    store._search_cache.clear()

    # Prefetch domain B
    store.prefetch_predicted_domain("domain_a", top_k=5)

    # Warm start (cache hit)
    start_time = time.perf_counter()
    store.search(query_b, domain="domain_b", top_k=5)
    time_warm = time.perf_counter() - start_time

    print(f"Cold time: {time_cold}, Warm time: {time_warm}")

    cache_key = (tuple(query_b), 5, "domain_b", None)
    assert cache_key in store._search_cache

    # Optional: ensure we can benchmark this properly
    metrics = {
        "cold_time": time_cold,
        "warm_time": time_warm,
        "speedup": time_cold / time_warm if time_warm > 0 else float('inf')
    }

    with open("living_harness/data/predictive_caching_metrics.json", "w") as f:
        json.dump(metrics, f)
