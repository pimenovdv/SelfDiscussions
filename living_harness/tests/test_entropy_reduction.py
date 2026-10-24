import pytest
import os
import json
import random
import time
from living_harness.memory.vector_store import VectorStore

@pytest.fixture
def temp_memory_file(tmp_path):
    file_path = tmp_path / "test_entropy_memory.json"
    yield str(file_path)
    if os.path.exists(file_path):
        os.remove(file_path)

def test_entropy_reduction_triggers_abstraction(temp_memory_file):
    store = VectorStore(db_path=temp_memory_file)

    # Simulate high entropy (random vectors)
    for i in range(10):
        random_vec = [random.random() for _ in range(5)]
        store.add_item(f"Item {i}", random_vec, domain="test_domain")

    # Calculate initial entropy
    initial_entropy = store.calculate_domain_entropy("test_domain")

    # We expect entropy to be relatively high for random data
    assert initial_entropy > 0.0

    # If entropy is high, trigger abstraction
    if initial_entropy > 0.05:  # Arbitrary threshold for test
        store.cluster_and_abstract(n_clusters=2)

    # Check if abstract items were created
    abstract_items = [item for item in store.memory if item.get("metadata", {}).get("is_abstract")]

    # Verify abstraction happened
    assert len(abstract_items) == 2
    assert "abstract" in [item.get("domain") for item in abstract_items]

def test_low_entropy_no_action(temp_memory_file):
    store = VectorStore(db_path=temp_memory_file)

    # Simulate low entropy (identical or very similar vectors)
    base_vec = [0.5] * 5
    for i in range(5):
        # Very small variations
        vec = [v + random.uniform(-0.01, 0.01) for v in base_vec]
        store.add_item(f"Similar Item {i}", vec, domain="stable_domain")

    entropy = store.calculate_domain_entropy("stable_domain")

    # Entropy should be very low
    assert entropy < 0.01
