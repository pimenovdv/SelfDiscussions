import pytest
import os
from living_harness.memory.vector_store import VectorStore

def test_search_increases_importance_weight(tmp_path):
    db_path = tmp_path / "test_memory.json"
    store = VectorStore(db_path=str(db_path))

    store.add_item("item 1", [1.0, 0.0], {"importance_weight": 1.0})
    store.add_item("item 2", [0.0, 1.0], {"importance_weight": 1.0})

    # search for item 1
    results = store.search([1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0]["text"] == "item 1"
    # The search method should have increased the weight
    assert results[0]["metadata"]["importance_weight"] == pytest.approx(1.1)

    # verify persistence
    store_reloaded = VectorStore(db_path=str(db_path))
    assert store_reloaded.memory[0]["metadata"]["importance_weight"] == pytest.approx(1.1)
    assert store_reloaded.memory[1]["metadata"]["importance_weight"] == pytest.approx(1.0)
