import os
import time
import pytest
from living_harness.memory.vector_store import VectorStore

def test_last_accessed_updates_on_search(tmp_path):
    db_path = str(tmp_path / "test_vector_memory.json")
    store = VectorStore(db_path=db_path)

    # Add an item
    vector = [1.0, 0.0, 0.0]
    store.add_item("Test item", vector)

    # Get the original last_accessed
    assert len(store.memory) == 1
    original_last_accessed = store.memory[0]["metadata"]["last_accessed"]

    # Wait a bit to ensure time difference
    time.sleep(0.01)

    # Search for the item
    results = store.search(vector, top_k=1)
    assert len(results) == 1

    # Get the new last_accessed
    new_last_accessed = results[0]["metadata"]["last_accessed"]

    assert new_last_accessed > original_last_accessed

    # Check if it was saved
    store_reloaded = VectorStore(db_path=db_path)
    assert store_reloaded.memory[0]["metadata"]["last_accessed"] == new_last_accessed
