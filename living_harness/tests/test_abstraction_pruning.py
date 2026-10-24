import pytest
import time
import os
from living_harness.memory.vector_store import VectorStore

def test_prune_stale_abstractions():
    # Use a specific test file to avoid conflicts
    test_db = "data/test_abstraction_pruning.json"
    if os.path.exists(test_db):
        os.remove(test_db)

    store = VectorStore(db_path=test_db)

    # Add regular items
    store.add_item(text="Item 1", vector=[0.1, 0.2], domain="test")
    store.add_item(text="Item 2", vector=[0.2, 0.1], domain="test")

    # Add abstract concepts with different weights
    store.add_item(text="Abstract Concept: cluster_0", vector=[0.15, 0.15], domain="abstract", metadata={"is_abstract": True, "importance_weight": 1.5})
    store.add_item(text="Abstract Concept: cluster_1", vector=[0.5, 0.5], domain="abstract", metadata={"is_abstract": True, "importance_weight": 0.9})

    assert len(store.memory) == 4

    # Prune abstractions with weight < 1.05
    store.prune_stale_abstractions(threshold=1.05)

    assert len(store.memory) == 3

    # Verify the remaining items
    remaining_texts = [item.get("text") for item in store.memory]
    assert "Item 1" in remaining_texts
    assert "Item 2" in remaining_texts
    assert "Abstract Concept: cluster_0" in remaining_texts
    assert "Abstract Concept: cluster_1" not in remaining_texts

    if os.path.exists(test_db):
        os.remove(test_db)

def test_prune_stale_abstractions_leaves_regular_items():
    test_db = "data/test_abstraction_pruning_2.json"
    if os.path.exists(test_db):
        os.remove(test_db)

    store = VectorStore(db_path=test_db)

    # Add regular items with low weight
    store.add_item(text="Low Weight Item", vector=[0.1, 0.2], domain="test", metadata={"importance_weight": 0.5})

    assert len(store.memory) == 1

    # Prune abstractions
    store.prune_stale_abstractions(threshold=1.05)

    # Regular item should not be pruned even though its weight is < 1.05
    assert len(store.memory) == 1
    assert store.memory[0].get("text") == "Low Weight Item"

    if os.path.exists(test_db):
        os.remove(test_db)
