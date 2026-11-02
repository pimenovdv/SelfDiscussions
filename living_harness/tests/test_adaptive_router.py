import pytest
import os
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

def test_adaptive_router(store):
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
