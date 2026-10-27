import pytest
import os
import numpy as np
from living_harness.memory.vector_store import VectorStore

@pytest.fixture
def vector_store():
    store = VectorStore(db_path="test_boredom_vector_memory.json")
    yield store
    if os.path.exists("test_boredom_vector_memory.json"):
        os.remove("test_boredom_vector_memory.json")

def test_boredom_signal(vector_store):
    # Test low boredom (high variance)
    for i in range(5):
        vector_store.add_item(f"Diverse idea {i}", np.random.rand(3).tolist())

    boredom_low = vector_store.calculate_boredom(recent_k=5)

    # Test high boredom (low variance, repeated identical vectors)
    vec = [1.0, 0.0, 0.0]
    for i in range(5):
        vector_store.add_item(f"Same idea {i}", vec)

    boredom_high = vector_store.calculate_boredom(recent_k=5)

    assert boredom_high > boredom_low
    assert boredom_high > 0.9  # Should be close to 1.0 for identical vectors
    assert 0 <= boredom_low <= 1.0
