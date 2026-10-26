import pytest
import os
from living_harness.memory.vector_store import VectorStore

@pytest.fixture
def vector_store():
    # Use a temporary file for the test store to avoid polluting main data
    store = VectorStore(db_path="test_spontaneous_vector_memory.json")
    yield store
    # Cleanup after test
    if os.path.exists("test_spontaneous_vector_memory.json"):
        os.remove("test_spontaneous_vector_memory.json")

def test_spontaneous_activation(vector_store):
    # Add items with different weights
    # 1. High weight (safe)
    vector_store.add_item("High weight memory", [0.1, 0.2, 0.3], metadata={"importance_weight": 1.0})
    # 2. Borderline weight (should be activated)
    vector_store.add_item("Borderline memory", [0.4, 0.5, 0.6], metadata={"importance_weight": 0.65})
    # 3. Too low weight (below threshold, won't be saved by activation alone)
    vector_store.add_item("Low weight memory", [0.2, 0.2, 0.2], metadata={"importance_weight": 0.2})
    # 4. Abstract concept (should be ignored)
    vector_store.add_item("Abstract concept", [0.7, 0.8, 0.9], metadata={"importance_weight": 0.7, "is_abstract": True})
    # 5. Meta knowledge (should be ignored)
    vector_store.add_item("Meta knowledge", [0.1, 0.9, 0.9], metadata={"importance_weight": 0.65, "type": "meta_knowledge"})

    # Run spontaneous activation with threshold 0.6 and boost 0.2
    vector_store.spontaneous_activation(threshold=0.6, boost_amount=0.2)

    # Check updated weights
    memories = vector_store.memory

    for item in memories:
        if item["text"] == "High weight memory":
            assert item["metadata"]["importance_weight"] == 1.0 # Unchanged
        elif item["text"] == "Borderline memory":
            # 0.65 + 0.2 = 0.85
            assert pytest.approx(item["metadata"]["importance_weight"], 0.01) == 0.85
        elif item["text"] == "Low weight memory":
            assert item["metadata"]["importance_weight"] == 0.2 # Unchanged
        elif item["text"] == "Abstract concept":
            assert item["metadata"]["importance_weight"] == 0.7 # Unchanged because abstract
        elif item["text"] == "Meta knowledge":
            assert item["metadata"]["importance_weight"] == 0.65 # Unchanged because meta_knowledge
