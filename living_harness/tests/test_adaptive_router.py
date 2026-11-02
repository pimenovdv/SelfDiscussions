import pytest
import os
import time
from living_harness.memory.vector_store import VectorStore
from living_harness.memory.router import AdaptiveRouter

@pytest.fixture
def store():
    db_path = "living_harness/data/test_adaptive_router.json"
    if os.path.exists(db_path):
        os.remove(db_path)
    store = VectorStore(db_path=db_path)
    yield store
    if os.path.exists(db_path):
        os.remove(db_path)

def test_adaptive_router_correctness(store):
    router = AdaptiveRouter(store)

    vector_a1 = [1.0] * 192 + [0.0] * 192
    vector_b1 = [0.0] * 192 + [1.0] * 192

    store.add_item("A1", vector_a1, domain="domain_a")
    store.add_item("B1", vector_b1, domain="domain_b")

    query_vector_a = [0.95] * 192 + [0.05] * 192
    routed_memory = router.route(query_vector_a, base_similarity_threshold=0.8)

    assert len(routed_memory) == 1
    domains = {item["domain"] for item in routed_memory}
    assert "domain_a" in domains
    assert "domain_b" not in domains

def test_adaptive_router_performance(store):
    router = AdaptiveRouter(store)

    # Populate store with many items
    for i in range(100):
        store.add_item(f"A{i}", [0.9] * 192 + [0.1] * 192, domain="domain_a")
    for i in range(400):
        store.add_item(f"B{i}", [0.1] * 192 + [0.9] * 192, domain="domain_b")

    query_vector = [0.95] * 192 + [0.05] * 192

    # Warmup
    store.search(query_vector, top_k=5)
    router.route(query_vector, base_similarity_threshold=0.8)

    # Measure full search time
    start_time = time.perf_counter()
    full_search_results = store.search(query_vector, top_k=5)
    full_search_time = time.perf_counter() - start_time

    # Measure routed search time
    start_time = time.perf_counter()
    routed_memory = router.route(query_vector, base_similarity_threshold=0.8)
    original_memory = store.memory
    store.memory = routed_memory
    routed_search_results = store.search(query_vector, top_k=5)
    store.memory = original_memory
    routed_search_time = time.perf_counter() - start_time

    # Assert time reduction
    assert routed_search_time < full_search_time, f"Routed search ({routed_search_time}) not faster than full search ({full_search_time})"

    # Assert accuracy remains comparable (both find top items from domain_a)
    assert len(full_search_results) == 5
    assert len(routed_search_results) == 5
    assert all(item["domain"] == "domain_a" for item in full_search_results)
    assert all(item["domain"] == "domain_a" for item in routed_search_results)
