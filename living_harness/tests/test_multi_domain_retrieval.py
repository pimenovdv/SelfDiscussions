import pytest
from living_harness.memory.vector_store import VectorStore

def test_multi_domain_retrieval(tmp_path):
    db_file = tmp_path / "test_vector_memory.json"
    store = VectorStore(db_path=str(db_file))

    # Add items to different domains
    store.add_item("physics fact", [1.0, 0.0, 0.0], domain="physics")
    store.add_item("biology fact", [0.0, 1.0, 0.0], domain="biology")
    store.add_item("philosophy fact", [0.0, 0.0, 1.0], domain="philosophy")
    store.add_item("chemistry fact", [0.5, 0.5, 0.0], domain="chemistry")

    # Test single domain string
    results = store.search([1.0, 0.0, 0.0], top_k=5, domain="physics")
    assert len(results) == 1
    assert results[0]["domain"] == "physics"

    # Test list of domains
    results = store.search([1.0, 1.0, 1.0], top_k=5, domain=["physics", "philosophy"])
    assert len(results) == 2
    domains_retrieved = {res["domain"] for res in results}
    assert domains_retrieved == {"physics", "philosophy"}

    # Test none domain (all)
    results = store.search([1.0, 1.0, 1.0], top_k=5, domain=None)
    assert len(results) == 4
