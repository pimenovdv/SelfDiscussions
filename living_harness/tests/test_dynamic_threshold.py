import pytest
from living_harness.memory.vector_store import VectorStore
from unittest.mock import patch

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_memory.json"
    return str(db_file)

def test_calculate_domain_entropy(temp_db):
    store = VectorStore(db_path=temp_db)

    # Empty domain
    assert store.calculate_domain_entropy("empty") == 0.0

    # 1 item domain
    store.add_item("item", [1.0, 0.0], domain="one_item")
    assert store.calculate_domain_entropy("one_item") == 0.0

    # Low variance
    store.add_item("low1", [1.0, 0.0], domain="low")
    store.add_item("low2", [0.99, 0.01], domain="low")
    low_entropy = store.calculate_domain_entropy("low")

    # High variance
    store.add_item("high1", [1.0, 0.0], domain="high")
    store.add_item("high2", [-1.0, 0.0], domain="high")
    high_entropy = store.calculate_domain_entropy("high")

    assert high_entropy > low_entropy
    assert low_entropy > 0.0

def test_validate_hypothesis_dynamic(temp_db):
    store = VectorStore(db_path=temp_db)
    store.add_item("item1", [1.0, 0.0], domain="test_domain")
    store.add_item("item2", [1.0, 0.0], domain="test_domain")

    hyp_vector = [0.8, 0.6]  # Cosine sim with [1.0, 0.0] is 0.8

    with patch.object(store, 'calculate_domain_entropy', return_value=0.0):
        # Entropy 0 -> dynamic threshold = max(0.1, min(0.9, 0.9 - 0.0)) = 0.9
        # avg score 0.8 < 0.9 -> False
        assert not store.validate_hypothesis_dynamic("hyp", hyp_vector, "test_domain", base_threshold=0.9)

    with patch.object(store, 'calculate_domain_entropy', return_value=2.0):
        # Entropy 2.0 -> dynamic threshold = max(0.1, min(0.9, 0.9 - 0.2)) = 0.7
        # avg score 0.8 >= 0.7 -> True
        assert store.validate_hypothesis_dynamic("hyp", hyp_vector, "test_domain", base_threshold=0.9)
