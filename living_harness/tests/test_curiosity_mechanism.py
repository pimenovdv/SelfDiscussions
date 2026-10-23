import pytest
import os
from living_harness.memory.vector_store import VectorStore

def test_curiosity_targets_identification():
    test_path = "living_harness/data/test_curiosity_memory.json"
    if os.path.exists(test_path):
        os.remove(test_path)

    store = VectorStore(db_path=test_path)
    store.clear()

    # Create memory items to establish domains
    store.add_item("Data 1", [1.0, 0.0], domain="math")
    store.add_item("Data 2", [0.0, 1.0], domain="physics")
    store.add_item("Data 3", [0.5, 0.5], domain="chemistry")
    store.add_item("Data 4", [0.2, 0.8], domain="biology")

    # Simulate some transitions to build a graph
    # math <-> physics (strong connection)
    # physics -> chemistry (weak connection)
    # biology is completely isolated (no transitions involve it)

    # Trigger transitions via search
    # This sets transitions['math']['physics'] = 1
    store.search([1.0, 0.0], domain="math")
    store.search([0.0, 1.0], domain="physics")

    # This sets transitions['physics']['math'] = 1
    store.search([0.0, 1.0], domain="physics")
    store.search([1.0, 0.0], domain="math")

    # This sets transitions['math']['chemistry'] = 1 (to add some connection)
    store.search([1.0, 0.0], domain="math")
    store.search([0.5, 0.5], domain="chemistry")


    targets = store.identify_curiosity_targets()

    # Expected:
    # math: out=2 (physics, chemistry), in=1 (physics) -> total=3 (not a target)
    # physics: out=1 (math), in=1 (math) -> total=2 (not a target)
    # chemistry: out=0, in=1 (math) -> total=1 (target)
    # biology: out=0, in=0 -> total=0 (target)

    assert "biology" in targets, "Biology should be a target as it is isolated."
    assert "chemistry" in targets, "Chemistry should be a target as it has low connectivity."
    assert "math" not in targets, "Math has high connectivity, should not be a target."

    if os.path.exists(test_path):
        os.remove(test_path)
