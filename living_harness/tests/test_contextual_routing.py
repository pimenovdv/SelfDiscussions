import pytest
import os
from living_harness.memory.vector_store import VectorStore
from unittest.mock import patch

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
    vector_a1 = [1.0] * 192 + [0.0] * 192
    vector_a2 = [0.9] * 192 + [0.1] * 192
    vector_b1 = [0.0] * 192 + [1.0] * 192
    vector_b2 = [0.1] * 192 + [0.9] * 192

    store.add_item("A1", vector_a1, domain="domain_a")
    store.add_item("A2", vector_a2, domain="domain_a")
    store.add_item("B1", vector_b1, domain="domain_b")
    store.add_item("B2", vector_b2, domain="domain_b")

    query_vector_a = [0.95] * 192 + [0.05] * 192
    routed_memory = store.contextual_routing(query_vector_a, base_similarity_threshold=0.8)

    assert len(routed_memory) == 2
    domains = {item["domain"] for item in routed_memory}
    assert "domain_a" in domains
    assert "domain_b" not in domains

    query_vector_b = [0.05] * 192 + [0.95] * 192
    routed_memory_b = store.contextual_routing(query_vector_b, base_similarity_threshold=0.8)

    assert len(routed_memory_b) == 2
    domains_b = {item["domain"] for item in routed_memory_b}
    assert "domain_b" in domains_b
    assert "domain_a" not in domains_b

    query_vector_unrelated = [0.0] * 384
    routed_memory_unrelated = store.contextual_routing(query_vector_unrelated, base_similarity_threshold=0.8)
    assert len(routed_memory_unrelated) == 4

def test_contextual_routing_entropy_expansion(store):
    store.add_item("C1", [0.5] * 192 + [0.0] * 192, domain="domain_c")
    store.add_item("C2", [0.6] * 192 + [0.0] * 192, domain="domain_c")

    # E is our anchor to prevent fallback
    store.add_item("E1", [1.0] * 384, domain="domain_e")

    # Query matches E perfectly, and C partially (sim ~ 0.55)
    query = [1.0] * 384

    # 1. Low entropy -> dynamic threshold remains near 0.8
    # Domain C sim ~0.55 < 0.8 -> C is not routed.
    with patch.object(store, 'calculate_domain_entropy', return_value=0.0):
        routed = store.contextual_routing(query, base_similarity_threshold=0.8, use_dynamic_threshold=True)
        domains = {item["domain"] for item in routed}
        assert "domain_e" in domains
        assert "domain_c" not in domains

    # 2. High entropy -> dynamic threshold drops (e.g. 0.8 - 2.0*0.15 = 0.5)
    # Domain C sim ~0.55 >= 0.5 -> C is routed!
    with patch.object(store, 'calculate_domain_entropy', return_value=2.0):
        routed2 = store.contextual_routing(query, base_similarity_threshold=0.8, use_dynamic_threshold=True)
        domains2 = {item["domain"] for item in routed2}
        assert "domain_e" in domains2
        assert "domain_c" in domains2
