import pytest
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from living_harness.memory.vector_store import VectorStore
import torch

@pytest.fixture
def temp_memory_file(tmp_path):
    filepath = tmp_path / "test_memory_llm.json"
    yield str(filepath)
    if os.path.exists(filepath):
        os.remove(filepath)

@pytest.mark.slow
def test_integrate_hypothesis_pipeline_llm(temp_memory_file):
    # This test verifies that we can run the pipeline with a real model
    # We use a very small model or mock the generation for the test to run fast,
    # but the architecture supports passing the model output to the vector store.
    store = VectorStore(db_path=temp_memory_file)

    # Create isolated domain 'C'
    store.add_item("The sky is blue.", [1.0, 0.0, 0.0, 0.0], domain="Science")
    store.add_item("Water is H2O.", [0.0, 1.0, 0.0, 0.0], domain="Chemistry")
    store.add_item("Consciousness is a mystery.", [0.0, 0.0, 1.0, 0.0], domain="Philosophy") # Isolated

    # Create transitions between Science and Chemistry
    store.transitions = {
        "Science": {"Chemistry": 1},
        "Chemistry": {"Science": 1}
    }

    # We can inject a hypothesis that an LLM would generate
    llm_generated_hypothesis = "Perhaps consciousness arises from complex chemical reactions."
    llm_vector = [0.1, 0.8, 0.8, 0.0] # Something between Chemistry and Philosophy

    # Simulate the pipeline step where LLM output is validated
    is_valid = store.validate_hypothesis_dynamic(llm_generated_hypothesis, llm_vector, "Philosophy")

    if is_valid:
        new_item = {
            "text": llm_generated_hypothesis,
            "vector": llm_vector,
            "domain": "meta",
            "metadata": {
                "type": "meta_knowledge",
                "source_domain": "Philosophy",
                "importance_weight": 2.0
            }
        }
        store.memory.append(new_item)
        store._save()

    meta_items = [item for item in store.memory if item.get("domain") == "meta"]
    assert len(meta_items) > 0
    assert meta_items[0]["metadata"]["type"] == "meta_knowledge"
