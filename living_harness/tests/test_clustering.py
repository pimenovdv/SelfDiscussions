import pytest
from living_harness.memory.vector_store import VectorStore

def test_clustering_abstractions():
    store = VectorStore(db_path="data/test_clustering.json")
    store.clear()

    # 2 clusters: [1, 0, 0] and [0, 1, 0]
    for i in range(5):
        store.add_item(f"Cat {i}", [1.0, 0.1 * i, 0.0])
        store.add_item(f"Dog {i}", [0.0, 1.0, 0.1 * i])

    initial_len = len(store.memory)

    # Run clustering
    store.cluster_and_abstract(n_clusters=2)

    final_len = len(store.memory)

    assert final_len == initial_len + 2

    abstract_items = [item for item in store.memory if item.get("metadata", {}).get("is_abstract")]
    assert len(abstract_items) == 2
    assert abstract_items[0]["domain"] == "abstract"
    assert "Abstract Concept" in abstract_items[0]["text"]
