import pytest
import os
from living_harness.memory.vector_store import VectorStore

@pytest.fixture
def store():
    db_path = "living_harness/data/test_contextual_routing.json"
    if os.path.exists(db_path):
        os.remove(db_path)
    store = VectorStore(db_path=db_path)
    yield store
    if os.path.exists(db_path):
        os.remove(db_path)

def test_contextual_routing(store):
    # Setup domains with distinctly different vectors
    # Domain A: High values in first half
    vector_a1 = [1.0] * 192 + [0.0] * 192
    vector_a2 = [0.9] * 192 + [0.1] * 192

    # Domain B: High values in second half
    vector_b1 = [0.0] * 192 + [1.0] * 192
    vector_b2 = [0.1] * 192 + [0.9] * 192

    store.add_item("A1", vector_a1, domain="domain_a")
    store.add_item("A2", vector_a2, domain="domain_a")
    store.add_item("B1", vector_b1, domain="domain_b")
    store.add_item("B2", vector_b2, domain="domain_b")

    # Query clearly aligned with Domain A
    query_vector_a = [0.95] * 192 + [0.05] * 192

    # Route context
    routed_memory = store.contextual_routing(query_vector_a, similarity_threshold=0.8)

    # Should only contain items from domain_a
    assert len(routed_memory) == 2
    domains = {item["domain"] for item in routed_memory}
    assert "domain_a" in domains
    assert "domain_b" not in domains

    # Query clearly aligned with Domain B
    query_vector_b = [0.05] * 192 + [0.95] * 192

    # Route context
    routed_memory_b = store.contextual_routing(query_vector_b, similarity_threshold=0.8)

    # Should only contain items from domain_b
    assert len(routed_memory_b) == 2
    domains_b = {item["domain"] for item in routed_memory_b}
    assert "domain_b" in domains_b
    assert "domain_a" not in domains_b

    # Query completely unrelated (e.g. all zeros) - should fallback to all memory
    query_vector_unrelated = [0.0] * 384
    routed_memory_unrelated = store.contextual_routing(query_vector_unrelated, similarity_threshold=0.8)
    assert len(routed_memory_unrelated) == 4
