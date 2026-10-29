import pytest
from living_harness.memory.vector_store import VectorStore

def test_validate_hypothesis():
    vs = VectorStore(db_path=":memory:")

    # Add some domain knowledge
    vs.add_item("Dogs are mammals.", domain="biology", vector=[0.9, 0.1, 0.0])
    vs.add_item("Cats are mammals.", domain="biology", vector=[0.8, 0.2, 0.0])
    vs.add_item("Birds lay eggs.", domain="biology", vector=[0.2, 0.8, 0.0])

    # Test valid hypothesis (close to dogs/cats)
    valid_hypo_vec = [0.85, 0.15, 0.0]
    # Cosine similarity between [0.85, 0.15, 0] and [0.9, 0.1, 0]: ~0.99
    # Cosine similarity between [0.85, 0.15, 0] and [0.8, 0.2, 0]: ~0.99
    is_valid = vs.validate_hypothesis("Wolves are mammals.", valid_hypo_vec, threshold=0.5)
    assert is_valid == True

    # Test invalid hypothesis (far from existing)
    invalid_hypo_vec = [0.0, 0.0, 0.9]
    is_valid = vs.validate_hypothesis("Stars are hot.", invalid_hypo_vec, threshold=0.5)
    assert is_valid == False

if __name__ == "__main__":
    pytest.main(["-v", "living_harness/tests/test_hypothesis_validation.py"])
