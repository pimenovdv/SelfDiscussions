import os
from living_harness.memory.context_manager import ContextManager

def test_meta_knowledge_integration():
    cm = ContextManager(system_prompt="Test System", enable_hybrid_compression=False)
    # mock vector search
    cm.vector_store.add_item("My meta insight", [0.5, 0.5], {"type": "meta_knowledge"})
    cm.vector_store.add_item("Some episodic data", [0.5, 0.5], {"type": "episodic"})

    prompt = cm.build_prompt(query_vector=[0.5, 0.5])

    assert "<|meta_reflection|>" in prompt
    assert "My meta insight" in prompt
    assert "Some episodic data" not in prompt

    if os.path.exists("living_harness/data/vector_memory.json"):
        os.remove("living_harness/data/vector_memory.json")
