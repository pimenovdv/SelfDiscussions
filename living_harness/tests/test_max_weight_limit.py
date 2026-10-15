import pytest
import os
from living_harness.memory.vector_store import VectorStore

@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path / "test_max_weight.json")

def test_max_weight_limit(temp_db_path):
    store = VectorStore(db_path=temp_db_path)

    # Add an item with an initial weight close to the limit
    store.add_item("Test item", [1.0, 0.0], {"importance_weight": 4.8})

    # Search repeatedly to trigger reinforcement multiple times
    for _ in range(5):
        store.search([1.0, 0.0], top_k=1)

    # Retrieve the item to check its weight
    # 4.8 + 0.1 * 5 = 5.3, but it should be capped at 5.0
    final_weight = store.memory[0]["metadata"]["importance_weight"]
    assert final_weight == 5.0, f"Expected weight to be capped at 5.0, but got {final_weight}"

def test_max_weight_limit_already_at_max(temp_db_path):
    store = VectorStore(db_path=temp_db_path)

    # Add an item already at the maximum weight
    store.add_item("Test item 2", [0.0, 1.0], {"importance_weight": 5.0})

    # Search to trigger reinforcement
    store.search([0.0, 1.0], top_k=1)

    # The weight should remain at 5.0
    final_weight = store.memory[0]["metadata"]["importance_weight"]
    assert final_weight == 5.0, f"Expected weight to remain 5.0, but got {final_weight}"
