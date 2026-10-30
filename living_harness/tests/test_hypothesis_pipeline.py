import pytest
import os
import json
from living_harness.memory.vector_store import VectorStore

@pytest.fixture
def temp_memory_file(tmp_path):
    filepath = tmp_path / "test_memory.json"
    yield str(filepath)
    if os.path.exists(filepath):
        os.remove(filepath)

def test_integrate_hypothesis_pipeline(temp_memory_file):
    store = VectorStore(db_path=temp_memory_file)

    # Create isolated domain 'C'
    store.add_item("A text", [1.0, 0.0, 0.0], domain="A")
    store.add_item("B text", [0.0, 1.0, 0.0], domain="B")
    store.add_item("C text", [0.0, 0.0, 1.0], domain="C") # Isolated

    # Create transitions between A and B
    store.transitions = {
        "A": {"B": 1},
        "B": {"A": 1}
    }

    # Run the pipeline
    integrated = store.integrate_hypothesis_pipeline(vectorization_callback=lambda h: [0.0, 0.0, 1.0])

    # Since C is isolated, it should generate a hypothesis for C and potentially validate it
    # We used a mock vector close to C's vector, so it should pass validation
    assert len(integrated) > 0
    assert any("Hypothesis: Domain 'C'" in item["text"] for item in integrated)

    # Check that it's added to memory
    meta_items = [item for item in store.memory if item.get("domain") == "meta"]
    assert len(meta_items) > 0
    assert meta_items[0]["metadata"]["type"] == "meta_knowledge"
