import os
import time
import math
from living_harness.memory.vector_store import VectorStore

def test_recency_boost():
    test_db = "living_harness/data/test_recency_boost.json"
    if os.path.exists(test_db):
        os.remove(test_db)

    store = VectorStore(db_path=test_db)

    # We create two vectors that are identical in content and importance
    vector1 = [1.0, 0.0, 0.0]
    vector2 = [1.0, 0.0, 0.0]

    current_time = time.time()

    # Vector 1 is older (e.g., 10 days old)
    store.add_item("Older item", vector1, {"importance_weight": 1.0, "last_accessed": current_time - (10 * 24 * 3600)})

    # Vector 2 is fresh (just accessed)
    store.add_item("Fresher item", vector2, {"importance_weight": 1.0, "last_accessed": current_time})

    # Search for an identical vector
    results = store.search([1.0, 0.0, 0.0], top_k=2)

    # Check that both are returned
    assert len(results) == 2

    # The fresher item should be ranked higher due to recency boost
    assert results[0]["text"] == "Fresher item"
    assert results[1]["text"] == "Older item"

    # Clean up
    if os.path.exists(test_db):
        os.remove(test_db)

if __name__ == "__main__":
    test_recency_boost()
