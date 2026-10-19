import pytest
from living_harness.memory.vector_store import VectorStore

def test_spatial_memory_retrieval():
    store = VectorStore(db_path="living_harness/data/test_spatial_vector_memory.json")
    store.clear()

    # Add items to different domains
    store.add_item("Physics is the study of matter.", [1.0, 0.0, 0.0], domain="science")
    store.add_item("Painting is a form of expression.", [0.9, 0.1, 0.0], domain="art")
    store.add_item("Biology is the study of life.", [0.0, 1.0, 0.0], domain="science")

    # Query vector close to 'art' but we search in 'science'
    query_vector = [0.9, 0.1, 0.0]
    results = store.search(query_vector, top_k=5, domain="science")

    # Assert
    assert len(results) == 2
    for item in results:
        assert item["domain"] == "science"
        assert "Painting" not in item["text"]

    store.clear()
