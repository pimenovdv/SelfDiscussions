import pytest
import os
from living_harness.memory.vector_store import VectorStore

def test_consolidate_abstractions():
    test_db = "living_harness/data/test_consolidation_abstractions.json"
    if os.path.exists(test_db):
        os.remove(test_db)

    vs = VectorStore(db_path=test_db)

    # Add memory items that can be clustered
    item1 = {"text": "A cat", "vector": [1.0, 0.0, 0.0], "domain": "animals"}
    item2 = {"text": "A dog", "vector": [0.9, 0.1, 0.0], "domain": "animals"}
    item3 = {"text": "A car", "vector": [0.0, 1.0, 0.0], "domain": "vehicles"}
    item4 = {"text": "A truck", "vector": [0.0, 0.9, 0.1], "domain": "vehicles"}
    item5 = {"text": "A bike", "vector": [0.0, 0.8, 0.2], "domain": "vehicles"}

    vs.memory.extend([item1, item2, item3, item4, item5])
    vs._save()

    initial_count = len(vs.memory)

    # Consolidate abstractions
    vs.consolidate_abstractions(max_depth=2, use_cosine=True)

    # Verify new elements were added
    assert len(vs.memory) > initial_count, "No abstract items were added."

    abstract_items = [item for item in vs.memory if item.get("metadata", {}).get("is_abstract") == True]
    assert len(abstract_items) > 0, "No abstract items with is_abstract=True found."

    for item in abstract_items:
        assert item["metadata"]["type"] == "hierarchical_abstraction"
        assert item["metadata"]["importance_weight"] == 2.0
        assert "vector" in item
        assert isinstance(item["vector"], list)

    if os.path.exists(test_db):
        os.remove(test_db)

if __name__ == "__main__":
    pytest.main(["-v", "living_harness/tests/test_consolidation_abstractions.py"])
