from living_harness.memory.context_manager import ContextManager
import pytest

def test_semantic_rules_integration():
    cm = ContextManager(system_prompt="Test System", max_window_tokens=100, enable_hybrid_compression=False)

    # Add fake rules to vector store
    cm.vector_store.add_item("Rule 1: Always be polite.", [0.1, 0.2], metadata={"type": "semantic"})
    cm.vector_store.add_item("Fact 1: The sky is blue.", [0.9, 0.8], metadata={"type": "episodic"})
    cm.vector_store.add_item("Rule 2: Think step by step.", [0.2, 0.3], metadata={"type": "semantic"})

    prompt = cm.build_prompt(query_vector=[0.15, 0.25])

    assert "<|rules|>" in prompt
    assert "[СЕМАНТИЧЕСКОЕ ПРАВИЛО]" in prompt
    assert "Always be polite." in prompt or "Think step by step." in prompt
    assert "The sky is blue." not in prompt # it's episodic
