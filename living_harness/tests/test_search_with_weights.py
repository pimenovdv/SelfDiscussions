import pytest
import os
from living_harness.memory.vector_store import VectorStore

def test_search_with_importance_weight():
    db_path = "living_harness/data/test_vector_memory_weights.json"
    if os.path.exists(db_path):
        os.remove(db_path)

    try:
        vector_store = VectorStore(db_path=db_path)

        vector = [1.0, 0.0, 0.0]
        vector_store.add_item("Pattern A (low weight)", vector, metadata={"importance_weight": 1.0})
        vector_store.add_item("Pattern A (high weight)", vector, metadata={"importance_weight": 5.0})
        vector_store.add_item("Pattern A (medium weight)", vector, metadata={"importance_weight": 3.0})

        results = vector_store.search(vector, top_k=3)

        assert len(results) == 3
        assert results[0]["metadata"]["importance_weight"] == 5.0
        assert results[1]["metadata"]["importance_weight"] == 3.0
        assert results[2]["metadata"]["importance_weight"] == 1.0
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
