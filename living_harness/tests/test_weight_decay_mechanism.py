import os
import time
import pytest
from living_harness.memory.vector_store import VectorStore

@pytest.fixture
def vector_store():
    test_db = "living_harness/data/test_vector_memory_decay_mechanism.json"
    if os.path.exists(test_db):
        os.remove(test_db)
    store = VectorStore(db_path=test_db)
    yield store
    if os.path.exists(test_db):
        os.remove(test_db)

def test_weight_decay_mechanism(vector_store):
    current_time = time.time()

    # 1. Old vector with high weight (10 days ago)
    vector_store.add_item("old_heavy", [1.0, 0.0], metadata={
        "importance_weight": 5.0,
        "last_accessed": current_time - 10 * 24 * 3600
    })

    # 2. Old vector with basic weight (10 days ago)
    vector_store.add_item("old_light", [0.0, 1.0], metadata={
        "importance_weight": 1.0,
        "last_accessed": current_time - 10 * 24 * 3600
    })

    # 3. New vector with high weight (0.5 days ago)
    vector_store.add_item("new_heavy", [1.0, 1.0], metadata={
        "importance_weight": 5.0,
        "last_accessed": current_time - 0.5 * 24 * 3600
    })

    # Run decay
    vector_store.decay_weights(decay_rate=0.1, days_threshold=1.0)

    items = {item["text"]: item for item in vector_store.memory}

    # Old heavy should lose weight
    old_heavy_weight = items["old_heavy"]["metadata"]["importance_weight"]
    assert 1.0 < old_heavy_weight < 5.0, f"Expected weight between 1.0 and 5.0, got {old_heavy_weight}"

    # Old light should remain 1.0
    old_light_weight = items["old_light"]["metadata"]["importance_weight"]
    assert old_light_weight == 1.0, f"Expected basic weight to remain 1.0, got {old_light_weight}"

    # New heavy should remain 5.0 because it's newer than days_threshold
    new_heavy_weight = items["new_heavy"]["metadata"]["importance_weight"]
    assert new_heavy_weight == 5.0, f"Expected new heavy weight to remain 5.0, got {new_heavy_weight}"
