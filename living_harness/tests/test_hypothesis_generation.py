import pytest
import os
from living_harness.memory.vector_store import VectorStore

def test_hypothesis_generation():
    test_path = "living_harness/data/test_hypothesis_memory.json"
    if os.path.exists(test_path):
        os.remove(test_path)

    store = VectorStore(db_path=test_path)
    store.clear()

    # Create memory items to establish domains
    store.add_item("Data 1", [1.0, 0.0], domain="math")
    store.add_item("Data 2", [0.0, 1.0], domain="physics")
    store.add_item("Data 3", [0.5, 0.5], domain="biology") # Isolated

    # Trigger transitions via search to connect math and physics
    store.search([1.0, 0.0], domain="math")
    store.search([0.0, 1.0], domain="physics")
    store.search([0.0, 1.0], domain="physics")
    store.search([1.0, 0.0], domain="math")

    # Generate hypotheses
    hypotheses = store.generate_curiosity_hypothesis()

    # Biology is isolated, so it should be in the hypothesis
    assert any("biology" in h.lower() for h in hypotheses), "Should generate hypothesis for isolated domain 'biology'"

    # We should have at least one hypothesis if there are isolated domains
    assert len(hypotheses) > 0, "Should generate at least one hypothesis"

    if os.path.exists(test_path):
        os.remove(test_path)
